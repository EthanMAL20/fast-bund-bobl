"""Performance and risk analytics.

* headline statistics (sheet ``Synthèse``): cumulative / annualised return,
  volatility, Sharpe ratio over the ECB deposit rate, maximum drawdown;
* calendar-year table;
* historical 1-day VaR at 99 % on the contracts actually held, its UCITS
  20-day scaling, and a Kupiec proportion-of-failures backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import BacktestResult
from .indicators import TRADING_DAYS


# --------------------------------------------------------------------- headline
def summary(res: BacktestResult) -> dict[str, float | pd.Timestamp]:
    pf = res.portfolio
    nav0 = res.config.initial_aum
    years = (pf.index[-1] - pf.index[0]).days / 365.25
    cum = pf["nav"].iloc[-1] / nav0 - 1
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0.0
    vol = pf["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    cash = pf["cash_rate"].mean()
    peak = pf["nav"].cummax().clip(lower=nav0)
    dd = pf["nav"] / peak - 1
    return {
        "start": pf.index[0],
        "end": pf.index[-1],
        "final_nav": pf["nav"].iloc[-1],
        "cumulative_return": cum,
        "annualised_return": ann,
        "annualised_vol": vol,
        "sharpe": (ann - cash) / vol if vol > 0 else 0.0,
        "max_drawdown": dd.min(),
        "max_drawdown_date": dd.idxmin(),
        "avg_gross_exposure": pf["gross_exposure"].mean(),
        "max_gross_exposure": pf["gross_exposure"].max(),
        "pnl_bund": pf["bund_pnl"].sum(),
        "pnl_bobl": pf["bobl_pnl"].sum(),
        "costs": -(pf[["bund_brokerage", "bund_slippage", "bobl_brokerage", "bobl_slippage"]].sum().sum()),
        "contracts_traded": pf["bund_traded"].sum() + pf["bobl_traded"].sum(),
    }


# ----------------------------------------------------------------- historical VaR
@dataclass
class VaRResult:
    var: pd.Series  # 1-day VaR as a fraction of start-of-day NAV (0 when flat)
    pnl: pd.Series  # realised daily futures P&L (EUR)
    exceptions: pd.Series  # True when the realised loss exceeded the VaR
    invested: pd.Series  # True when contracts were held during the session


def _scenario_changes(res: BacktestResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily price changes (points) with contract-change sessions removed."""
    d_b = res.bund["price"].diff().mask(res.bund["change_day"]).to_numpy()
    d_o = res.bobl["price"].diff().mask(res.bobl["change_day"]).to_numpy()
    valid = ~np.isnan(d_b) & ~np.isnan(d_o)
    return d_b, d_o, valid


def historical_var(
    res: BacktestResult, confidence: float = 0.99, lookback: int = 250
) -> VaRResult:
    """Full-revaluation historical VaR on the contracts held during each session.

    For session t, the contracts held since the previous settlement are applied
    to the futures price changes of the ``lookback`` previous valid sessions
    (contract-change sessions, where the generic series jumps, are skipped and
    the window extended). The loss quantile (linear interpolation, as Excel
    PERCENTILE.INC) is expressed as a fraction of the start-of-day NAV.
    """
    pf = res.portfolio
    pv = res.config.point_value
    d_b, d_o, valid = _scenario_changes(res)
    pos = pd.Series(np.arange(len(res.bund.index)), index=res.bund.index)

    var = pd.Series(0.0, index=pf.index)
    held_b = pf["bund_contracts_open"].to_numpy()
    held_o = pf["bobl_contracts_open"].to_numpy()
    nav_open = pf["nav_open"].to_numpy()
    for k, t in enumerate(pf.index):
        if held_b[k] == 0 and held_o[k] == 0:
            continue
        hist = np.flatnonzero(valid[: pos[t]])[-lookback:]
        scen = (held_b[k] * d_b[hist] + held_o[k] * d_o[hist]) * pv
        var.iat[k] = max(0.0, -np.quantile(scen, 1 - confidence)) / nav_open[k]

    pnl = pf["bund_pnl"] + pf["bobl_pnl"]
    invested = (pf["bund_contracts_open"] != 0) | (pf["bobl_contracts_open"] != 0)
    exceptions = invested & (-pnl > var * pf["nav_open"])
    return VaRResult(var=var, pnl=pnl, exceptions=exceptions, invested=invested)


def current_var(res: BacktestResult, confidence: float = 0.99, lookback: int = 250) -> tuple[float, float]:
    """VaR of tonight's position (end-of-day contracts), in % of NAV and in EUR."""
    pf, pv = res.portfolio, res.config.point_value
    d_b, d_o, valid = _scenario_changes(res)
    hist = np.flatnonzero(valid)[-lookback:]
    scen = (pf["bund_contracts"].iloc[-1] * d_b[hist] + pf["bobl_contracts"].iloc[-1] * d_o[hist]) * pv
    eur = max(0.0, -np.quantile(scen, 1 - confidence))
    return eur / pf["nav"].iloc[-1], eur


def kupiec_pof(n_obs: int, n_exceptions: int, confidence: float = 0.99) -> dict[str, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the exception rate equals 1 - confidence. LR ~ chi2(1).
    """
    p = 1 - confidence
    x, n = n_exceptions, n_obs
    phat = x / n
    def ll(prob: float) -> float:
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)
    lr = -2 * (ll(p) - ll(phat))
    return {"n_obs": n, "exceptions": x, "rate": phat, "expected": p, "lr_stat": lr, "p_value": 1 - stats.chi2.cdf(lr, 1)}


# ------------------------------------------------------------------ annual table
def annual_table(res: BacktestResult, var: VaRResult | None = None) -> pd.DataFrame:
    pf = res.portfolio.copy()
    nav0 = res.config.initial_aum
    pf["year"] = pf.index.year
    rows = []
    prev_close = nav0
    for year, g in pf.groupby("year"):
        start = g["nav_open"].iloc[0]
        perf = g["nav"].iloc[-1] / start - 1
        vol = g["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS) if len(g) > 1 else np.nan
        peak = g["nav"].cummax().clip(lower=start)
        dd = (g["nav"] / peak - 1).min()
        worst_vs_jan1 = min(0.0, (g["nav"] / prev_close - 1).min())
        row = {
            "year": year,
            "return": perf,
            "vol": vol,
            "return_over_vol": perf / vol if vol else np.nan,
            "max_drawdown": dd,
            "worst_vs_jan1": worst_vs_jan1,
            "avg_exposure": g["gross_exposure"].mean(),
            "pnl_bund": g["bund_pnl"].sum(),
            "pnl_bobl": g["bobl_pnl"].sum(),
            "nav_end": g["nav"].iloc[-1],
        }
        if var is not None:
            v = var.var.loc[g.index]
            row.update(
                var99_avg=v.mean(),
                var99_max=v.max(),
                ucits_breach_days=int((v * np.sqrt(20) > 0.20).sum()),
                var_exceptions=int(var.exceptions.loc[g.index].sum()),
            )
        rows.append(row)
        prev_close = g["nav"].iloc[-1]
    return pd.DataFrame(rows).set_index("year")
