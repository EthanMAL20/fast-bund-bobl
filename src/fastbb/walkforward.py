"""Walk-forward validation (rolling in-sample calibration, out-of-sample evaluation).

For each test year Y, every configuration of a parameter grid is scored on the
``train_years`` calendar years before Y (Sharpe ratio over the ECB deposit
rate, same definition as :func:`fastbb.risk.summary`). The best one is frozen
and its returns for year Y are kept. The kept years, chained together, form an
out-of-sample track record: no return in it was seen when its parameters were
chosen.

The signals are causal (they only use data up to each settlement), so scoring a
configuration on a sub-period of one full-history backtest gives the same
returns as a backtest run on that sub-period with its warm-up history.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data import MarketData
from .indicators import TRADING_DAYS


def set_param(cfg: Config, path: str, value: Any) -> Config:
    """Return a copy of ``cfg`` with a dotted attribute (``"bund.signal.rsi_period"``) changed."""
    cfg = copy.deepcopy(cfg)
    obj = cfg
    *parents, last = path.split(".")
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, last):
        raise AttributeError(f"unknown parameter {path!r}")
    setattr(obj, last, value)
    return cfg


def period_stats(pf: pd.DataFrame) -> dict[str, float]:
    """Headline statistics of a slice of the portfolio table (daily ``ret`` and ``cash_rate``)."""
    r = pf["ret"]
    years = (pf.index[-1] - pf.index[0]).days / 365.25
    growth = float((1 + r).prod())
    ann = growth ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    nav = (1 + r).cumprod()
    return {
        "cumulative_return": growth - 1,
        "annualised_return": ann,
        "annualised_vol": vol,
        "sharpe": float((ann - pf["cash_rate"].mean()) / vol) if vol > 0 else 0.0,
        "max_drawdown": float((nav / nav.cummax().clip(lower=1.0) - 1).min()),
    }


@dataclass
class WalkForwardResult:
    returns: pd.DataFrame  # out-of-sample sessions: ret, cash_rate, chosen configuration
    choices: pd.DataFrame  # one row per test year: training window, chosen parameters, OOS return
    grid: pd.DataFrame  # full-period statistics of every configuration (in-sample, for reference)

    @property
    def stats(self) -> dict[str, float]:
        return period_stats(self.returns)


def walk_forward(
    md: MarketData,
    base: Config,
    grid: dict[str, list[Any]],
    first_test_year: int = 2018,
    last_test_year: int | None = None,
    train_years: int = 3,
) -> WalkForwardResult:
    """Run every configuration of ``grid`` once, then select year by year on the trailing window."""
    names = list(grid)
    combos = list(itertools.product(*grid.values()))
    runs: dict[tuple, pd.DataFrame] = {}
    for combo in combos:
        cfg = base
        for path, value in zip(names, combo, strict=True):
            cfg = set_param(cfg, path, value)
        runs[combo] = run_backtest(md, cfg).portfolio[["ret", "cash_rate"]]

    any_pf = next(iter(runs.values()))
    last_test_year = last_test_year or int(any_pf.index[-1].year)
    if any_pf.index[0].year > first_test_year - train_years:
        raise ValueError("not enough history before the first test year")

    kept, rows = [], []
    for year in range(first_test_year, last_test_year + 1):
        train = slice(f"{year - train_years}", f"{year - 1}")
        best = max(combos, key=lambda c: period_stats(runs[c].loc[train])["sharpe"])
        test = runs[best].loc[str(year)].copy()
        for path, value in zip(names, best, strict=True):
            test[path] = value
        kept.append(test)
        rows.append({
            "test_year": year,
            "train": f"{year - train_years}-{year - 1}",
            **dict(zip(names, best, strict=True)),
            "train_sharpe": period_stats(runs[best].loc[train])["sharpe"],
            "oos_return": float((1 + test["ret"]).prod() - 1),
        })

    table = pd.DataFrame(
        [{**dict(zip(names, c, strict=True)), **period_stats(runs[c])} for c in combos]
    ).sort_values("sharpe", ascending=False, ignore_index=True)
    return WalkForwardResult(pd.concat(kept), pd.DataFrame(rows), table)
