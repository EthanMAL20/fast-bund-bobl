"""Portfolio simulation (sheet ``Portefeuille`` of the workbook).

One row per Eurex session. Each day:

1. P&L of the contracts held since yesterday's settlement;
2. circuit breaker (position cut after N consecutive losing sessions);
3. target contracts = NAV x global leverage x weight x signal leverage / contract notional;
4. roll: on the last trading day the position is closed at the settlement and
   re-opened on the next contract the following session;
5. brokerage and slippage on every contract traded, cash remunerated at the
   ECB deposit rate (ACT/360), management fee (ACT/365), optional performance
   fee with a year-end high-water mark.

Execution is assumed at the settlement price that is also used to compute the
signal (slightly favourable bias), margin calls are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import roll_dates
from .config import Config
from .data import MarketData
from .signals import build_bobl, build_bund


def excel_round(x: float) -> float:
    """Excel ROUND(x, 0): halves rounded away from zero (Python rounds to even)."""
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


@dataclass
class BacktestResult:
    config: Config
    portfolio: pd.DataFrame  # one row per simulated session
    bund: pd.DataFrame  # signal details, full history
    bobl: pd.DataFrame

    @property
    def nav(self) -> pd.Series:
        return self.portfolio["nav"]


def run_backtest(md: MarketData, cfg: Config) -> BacktestResult:
    bund = build_bund(md, cfg)
    bobl = build_bobl(md, cfg)
    dates = md.futures.index
    sim = dates >= pd.Timestamp(cfg.start_date)
    idx = np.flatnonzero(sim)
    if len(idx) == 0:
        raise ValueError("no data after the start date")

    rolls = set(roll_dates(cfg.roll_offset_days))
    ecb = md.ecb.sort_index()
    pv, tv = cfg.point_value, cfg.tick_value
    c = cfg.costs

    legs = {
        "bund": (bund, cfg.bund.weight if cfg.bund.active else 0.0, cfg.bund.stop_loss_days),
        "bobl": (bobl, cfg.bobl.weight if cfg.bobl.active else 0.0, cfg.bobl.stop_loss_days),
    }
    n = len(idx)
    rec = {k: np.zeros(n) for k in ("days", "nav_open", "cash_rate", "interest", "mgmt_fee", "perf_fee", "nav", "hwm")}
    leg_rec = {
        name: {k: np.zeros(n) for k in ("price", "leverage", "contracts_open", "pnl", "loss_streak", "contracts", "roll_close", "traded", "brokerage", "slippage")}
        for name in legs
    }
    roll_flag = np.zeros(n, dtype=bool)

    prev_nav = cfg.initial_aum
    hwm = cfg.initial_aum
    for k, i in enumerate(idx):
        first = k == 0
        d = dates[i]
        days = 0 if first else (d - dates[i - 1]).days
        nav_open = cfg.initial_aum if first else prev_nav
        is_roll = d in rolls
        roll_flag[k] = is_roll

        total_pnl = total_costs = 0.0
        for name, (sig, weight, stop) in legs.items():
            r = leg_rec[name]
            price = sig["price"].iat[i]
            lev = sig["leverage"].iat[i]
            held = 0.0 if first else r["contracts"][k - 1] + r["roll_close"][k - 1]
            pnl = 0.0 if first else held * (price - sig["price"].iat[i - 1]) * pv
            streak = (r["loss_streak"][k - 1] + 1 if k else 1) if pnl < 0 else 0
            if stop > 0 and streak >= stop:
                target = 0.0
            else:
                target = excel_round(nav_open * cfg.global_leverage * weight * lev * 100 / (price * cfg.nominal))
            roll_close = -target if is_roll else 0.0
            traded = abs(target - held) + abs(roll_close)
            brokerage = traded * c.brokerage_per_contract
            slippage = traded * c.slippage_ticks * tv
            for key, val in (
                ("price", price), ("leverage", lev), ("contracts_open", held), ("pnl", pnl),
                ("loss_streak", streak), ("contracts", target), ("roll_close", roll_close),
                ("traded", traded), ("brokerage", brokerage), ("slippage", slippage),
            ):
                r[key][k] = val
            total_pnl += pnl
            total_costs += brokerage + slippage

        rate = float(ecb[ecb.index <= d].iloc[-1]) + cfg.cash.margin if (ecb.index <= d).any() else 0.0
        if not cfg.cash.apply_negative_rates:
            rate = max(0.0, rate)
        interest = nav_open * cfg.cash.share * rate * days / cfg.cash.day_count
        mgmt = nav_open * c.management_fee * days / 365
        nav_before_perf = nav_open + total_pnl - total_costs + interest - mgmt

        year_end = k + 1 < n and dates[idx[k + 1]].year != d.year
        perf = c.performance_fee * max(0.0, nav_before_perf - hwm) if year_end else 0.0
        nav = nav_before_perf - perf
        if year_end:
            hwm = max(hwm, nav)

        for key, val in (
            ("days", days), ("nav_open", nav_open), ("cash_rate", rate), ("interest", interest),
            ("mgmt_fee", mgmt), ("perf_fee", perf), ("nav", nav), ("hwm", hwm),
        ):
            rec[key][k] = val
        prev_nav = nav

    pf = pd.DataFrame(rec, index=dates[idx])
    pf["roll"] = roll_flag
    for name in legs:
        for key, arr in leg_rec[name].items():
            pf[f"{name}_{key}"] = arr
    pf["ret"] = pf["nav"] / pf["nav_open"] - 1
    pf["gross_exposure"] = (
        (pf["bund_contracts"].abs() * pf["bund_price"] + pf["bobl_contracts"].abs() * pf["bobl_price"]) * pv / pf["nav"]
    )
    return BacktestResult(cfg, pf, bund, bobl)
