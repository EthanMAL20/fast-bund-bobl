"""Technical indicators, written to reproduce the Excel formulas exactly.

Signals are encoded as strings: ``"Long"``, ``"Short"`` or ``""`` (neutral),
as in the workbook, which keeps the parity tests readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LONG, SHORT, NEUTRAL = "Long", "Short", ""
TRADING_DAYS = 252


# --------------------------------------------------------------------- volatility
def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualised volatility of the ``window`` sessions *before* today.

    Excel: ``STDEV(Q[r-N] : Q[r-1]) * SQRT(252)``. Blank returns (excluded
    contract-change days) are skipped by STDEV, hence ``min_periods=2``.
    """
    vol = returns.rolling(window, min_periods=2).std(ddof=1).shift(1) * np.sqrt(TRADING_DAYS)
    vol.iloc[: window + 1] = np.nan  # Excel: IF(ROW()-N<7, "", ...)
    return vol


# ------------------------------------------------------------------ Student t-test
def t_threshold(proba: float, n_obs: int) -> float:
    """One-sided critical value, Excel ``TINV(2*p, N-1)``."""
    return float(stats.t.ppf(1.0 - proba, n_obs - 1))


def rolling_tstat(changes: pd.Series, n_obs: int) -> pd.Series:
    """t = sqrt(N) * mean / std of the last N daily price changes (today included).

    A zero standard deviation gives t = 0 (Excel ``IFERROR(..., 0)``).
    """
    roll = changes.rolling(n_obs, min_periods=n_obs)
    mean, std = roll.mean(), roll.std(ddof=1)
    t = np.sqrt(n_obs) * mean / std
    t[(std == 0) & mean.notna()] = 0.0
    return t


def threshold_signal(t: pd.Series, threshold: float, persistence: bool) -> np.ndarray:
    """Long if t >= threshold, Short if t <= -threshold, else keep or neutral."""
    out = np.empty(len(t), dtype=object)
    prev = NEUTRAL
    for i, x in enumerate(t.to_numpy()):
        if np.isnan(x):
            cur = NEUTRAL
        elif x >= threshold:
            cur = LONG
        elif x <= -threshold:
            cur = SHORT
        else:
            cur = prev if persistence else NEUTRAL
        out[i] = prev = cur
    return out


# ---------------------------------------------------------- triple moving average
def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window, min_periods=window).mean()


def triple_ma_signal(
    prices: pd.Series,
    windows: tuple[int, int, int],
    persistence: bool,
    inverted: bool = False,
) -> np.ndarray:
    """Long if short MA > mid MA > long MA, Short if the order is reversed."""
    f, g, h = (sma(prices, w).to_numpy() for w in windows)
    up, down = (SHORT, LONG) if inverted else (LONG, SHORT)
    out = np.empty(len(prices), dtype=object)
    prev = NEUTRAL
    for i in range(len(prices)):
        if np.isnan(f[i]) or np.isnan(g[i]) or np.isnan(h[i]):
            cur = NEUTRAL
        elif f[i] > g[i]:
            cur = up if g[i] > h[i] else (prev if persistence else NEUTRAL)
        else:
            cur = down if g[i] < h[i] else (prev if persistence else NEUTRAL)
        out[i] = prev = cur
    return out


# ------------------------------------------------------------------------- MACD
def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential moving average seeded with the first price (Excel recursion)."""
    return prices.ewm(span=span, adjust=False).mean()


def macd_signal(prices: pd.Series, fast: int, slow: int, signal: int, rule: str) -> np.ndarray:
    macd = ema(prices, fast) - ema(prices, slow)
    line = macd.ewm(span=signal, adjust=False).mean()
    ref = 0.0 if rule == "sign" else line
    return np.where(macd > ref, LONG, np.where(macd < ref, SHORT, NEUTRAL)).astype(object)


# -------------------------------------------------------------------------- RSI
def rsi_wilder(prices: pd.Series, period: int) -> pd.Series:
    """Wilder RSI; the smoothing is seeded with the first daily move (as in Excel)."""
    delta = prices.diff()
    up = delta.clip(lower=0).iloc[1:]
    down = (-delta).clip(lower=0).iloc[1:]
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_up / avg_down)
    rsi[avg_down == 0] = 100.0
    return rsi.reindex(prices.index)


def rsi_signal(
    prices: pd.Series,
    period: int,
    rule: str,
    high: float = 70,
    low: float = 30,
    persistence: bool = False,
) -> np.ndarray:
    rsi = rsi_wilder(prices, period).to_numpy()
    out = np.empty(len(rsi), dtype=object)
    prev = NEUTRAL
    for i, x in enumerate(rsi):
        if np.isnan(x):
            cur = NEUTRAL
        elif rule == "trend":
            cur = LONG if x > 50 else SHORT if x < 50 else NEUTRAL
        else:  # mean reversion
            cur = LONG if x < low else SHORT if x > high else (prev if persistence else NEUTRAL)
        out[i] = prev = cur
    return out
