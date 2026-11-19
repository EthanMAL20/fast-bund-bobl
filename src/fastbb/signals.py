"""Signal construction for the two legs (sheets ``Calc_BUND``, ``Calc_BOBL``,
``Calc_Indices`` of the workbook).

Pipeline for one leg::

    main trend signal (Student / MA / MACD / RSI)
      -> confirmations (Hang Seng, Dow Jones or NFP for the Bund; none, 2y-5y slope or NFP for the Bobl)
      -> label: Long / Mid-Long / Short / Mid-Short / NOK
      -> volatility filter (NOK when realised vol is outside [min, max])
      -> leverage table (multiple of NAV)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import contract_change_days
from .config import Config, LegParams, MAConfirmation, SlopeConfirmation
from .data import MarketData
from .indicators import (
    LONG,
    NEUTRAL,
    SHORT,
    macd_signal,
    realized_vol,
    rolling_tstat,
    rsi_signal,
    t_threshold,
    threshold_signal,
    triple_ma_signal,
)

NOK = "NOK"


# ------------------------------------------------------------------ main signal
def main_signal(prices: pd.Series, leg: LegParams) -> np.ndarray:
    p = leg.signal
    changes = prices.diff()
    if p.method == "student":
        thr = t_threshold(p.t_proba, p.t_obs)
        return threshold_signal(rolling_tstat(changes, p.t_obs), thr, p.persistence)
    if p.method == "moving_averages":
        return triple_ma_signal(prices, p.ma, p.persistence)
    if p.method == "macd":
        return macd_signal(prices, *p.macd, rule=p.macd_rule)
    if p.method == "rsi":
        return rsi_signal(prices, p.rsi_period, p.rsi_rule, p.rsi_high, p.rsi_low, p.rsi_persistence)
    raise ValueError(f"unknown signal method {p.method!r}")


def multi_window_signals(prices: pd.Series, leg: LegParams) -> list[np.ndarray]:
    """Student test on several windows (robustness switch « plusieurs durées »)."""
    p = leg.signal
    changes = prices.diff()
    out = []
    for n in p.windows:
        if n is None or n < 2:
            continue
        out.append(threshold_signal(rolling_tstat(changes, n), t_threshold(p.t_proba, n), p.persistence))
    return out


# ---------------------------------------------------------------- confirmations
def asof_map(signal: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> np.ndarray:
    """Value of an index signal on its own calendar, seen from each futures date.

    Excel: ``INDEX(signal, MATCH(date - lag, index_dates, 1))`` -> last
    observation on or before ``date - lag`` (calendar days).
    """
    target = dates - pd.Timedelta(days=lag_days)
    pos = signal.index.searchsorted(target, side="right") - 1
    vals = signal.to_numpy()
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], NEUTRAL).astype(object)


def ma_confirmation(index: pd.Series, conf: MAConfirmation, dates: pd.DatetimeIndex) -> np.ndarray:
    sig = pd.Series(triple_ma_signal(index, conf.ma, conf.persistence, conf.inverted), index=index.index)
    return asof_map(sig, dates, conf.lag_days)


def slope_confirmation(slope: pd.Series, conf: SlopeConfirmation, dates: pd.DatetimeIndex) -> np.ndarray:
    thr = t_threshold(conf.t_proba, conf.t_obs)
    raw = threshold_signal(rolling_tstat(slope.diff(), conf.t_obs), thr, conf.persistence)
    if conf.inverted:
        raw = np.array([SHORT if s == LONG else LONG if s == SHORT else s for s in raw], dtype=object)
    return asof_map(pd.Series(raw, index=slope.index), dates, conf.lag_days)


def nfp_signal(
    prices: pd.Series,
    nfp: pd.DataFrame,
    change_day: np.ndarray,
    method: str,
    hold_sessions: int,
) -> np.ndarray:
    """US payrolls signal.

    ``day_reaction``: direction of the futures move on the release session
    (14:30 Paris, before the 17:15 settlement), kept ``hold_sessions`` sessions
    (0 = until the next release). The two other methods use revised figures
    (look-ahead bias) and are kept for research only.
    """
    dates = prices.index
    px = prices.to_numpy()
    releases = pd.DatetimeIndex(nfp["release_date"]).to_numpy()
    out = np.full(len(dates), NEUTRAL, dtype=object)
    if method != "day_reaction":
        change = nfp["change"].astype(float)
        if method == "vs_3m_pace":
            metric = change - change.shift(1).rolling(3).mean()
        else:
            metric = change.rolling(3).mean() - change.rolling(12).mean()
        lab = np.where(metric > 0, SHORT, np.where(metric < 0, LONG, NEUTRAL))
    for i, d in enumerate(dates.to_numpy()):
        k = np.searchsorted(releases, d, side="right") - 1
        if k < 0:
            continue
        if method != "day_reaction":
            out[i] = lab[k]
            continue
        # first futures session on or after the release date
        j = np.searchsorted(dates.to_numpy(), releases[k] - np.timedelta64(1, "D"), side="right") - 1 + 1
        if j < 1 or j > i:
            continue
        if hold_sessions > 0 and (i - j) >= hold_sessions:
            continue
        if change_day[j]:
            continue
        out[i] = LONG if px[j] > px[j - 1] else SHORT if px[j] < px[j - 1] else NEUTRAL
    return out


# --------------------------------------------------------------- leg assembling
def _rebalance_mask(dates: pd.DatetimeIndex, leg: LegParams) -> np.ndarray:
    if leg.rebalance == "daily":
        return np.ones(len(dates), dtype=bool)
    return np.asarray(dates.weekday) + 1 == leg.rebalance_weekday


def _vol_filter(returns: pd.Series, leg: LegParams, cfg: Config) -> tuple[pd.Series, np.ndarray]:
    vol = realized_vol(returns, leg.signal.vol_window)
    if cfg.risk_method == "vol_target":
        ok = np.ones(len(vol), dtype=bool)
    else:
        ok = (vol >= leg.signal.vol_min).to_numpy() & (vol <= leg.signal.vol_max).to_numpy()
    return vol, ok


def _apply_vol_target(lev: np.ndarray, rebal: np.ndarray, vol60: pd.Series, leg: LegParams, cfg: Config) -> np.ndarray:
    if cfg.risk_method == "vol_filter":
        return lev
    out = np.zeros(len(lev))
    v = vol60.to_numpy()
    for i in range(len(lev)):
        if lev[i] == 0:
            out[i] = 0.0
        elif not rebal[i]:
            out[i] = out[i - 1] if i else 0.0
        elif np.isnan(v[i]):
            out[i] = 0.0
        else:
            out[i] = np.sign(lev[i]) * min(leg.lev_max, abs(lev[i]) * leg.vol_ref / v[i])
    return out


def _leverage_path(label, rebal, vol_ok, table, multi_fn=None) -> np.ndarray:
    """Leverage table lookup, Excel columns O (Bund) / L (Bobl).

    Single window: NOK -> 0, else hold yesterday's leverage on non-rebalancing
    days, else table[label]. Multi-window: vol filter -> 0, else hold, else the
    average leverage returned by ``multi_fn(i)``.
    """
    out = np.zeros(len(label))
    for i in range(len(label)):
        if multi_fn is None:
            if label[i] == NOK:
                out[i] = 0.0
            elif not rebal[i]:
                out[i] = out[i - 1] if i else 0.0
            else:
                out[i] = table[label[i]]
        else:
            if not vol_ok[i]:
                out[i] = 0.0
            elif not rebal[i]:
                out[i] = out[i - 1] if i else 0.0
            else:
                out[i] = multi_fn(i)
    return out


def build_bund(md: MarketData, cfg: Config) -> pd.DataFrame:
    leg = cfg.bund
    prices = md.futures["bund"]
    dates = prices.index
    change_day = contract_change_days(dates)
    returns = prices.pct_change()
    if cfg.exclude_roll_from_vol:
        returns = returns.mask(change_day)

    vol, vol_ok = _vol_filter(returns, leg, cfg)
    trend = main_signal(prices, leg)
    hsi = ma_confirmation(md.hsi, leg.hsi, dates)
    if leg.second_confirmation == "dow_jones":
        conf2 = ma_confirmation(md.dow_jones, leg.dow_jones, dates)
    elif leg.second_confirmation == "nfp":
        conf2 = nfp_signal(prices, md.nfp, change_day, cfg.nfp.method, cfg.nfp.hold_sessions)
    else:
        conf2 = np.full(len(dates), NEUTRAL, dtype=object)

    # Combination Bund trend + Hang Seng, then second confirmation
    label = np.empty(len(dates), dtype=object)
    for i in range(len(dates)):
        t, h, k = trend[i], hsi[i], conf2[i]
        if not vol_ok[i] or t == NEUTRAL:
            lab = NOK
        elif t == h:
            lab = t
        elif h == NEUTRAL:
            lab = "Mid-" + t
        else:
            lab = NOK
        if lab == "Mid-Long":
            lab = LONG if k == LONG else NOK if k == SHORT else lab
        elif lab == "Mid-Short":
            lab = SHORT if k == SHORT else NOK if k == LONG else lab
        label[i] = lab

    rebal = _rebalance_mask(dates, leg)
    table = leg.leverage
    multi_fn = None
    if leg.signal.multi_windows and leg.signal.method == "student":
        sigs = multi_window_signals(prices, leg)

        def multi_fn(i: int) -> float:
            if not sigs:
                return 0.0
            n_long = sum(s[i] == LONG for s in sigs)
            n_short = sum(s[i] == SHORT for s in sigs)
            h, k = hsi[i], conf2[i]
            if h == NEUTRAL:
                long_lab = LONG if k == LONG else NOK if k == SHORT else "Mid-Long"
                short_lab = SHORT if k == SHORT else NOK if k == LONG else "Mid-Short"
            else:
                long_lab = LONG if h == LONG else NOK
                short_lab = SHORT if h == SHORT else NOK
            return (n_long * table[long_lab] + n_short * table[short_lab]) / len(sigs)

    lev = _leverage_path(label, rebal, vol_ok, table, multi_fn)

    vol60 = realized_vol(returns, cfg.vol_target_window)
    final = _apply_vol_target(lev, rebal, vol60, leg, cfg)
    return pd.DataFrame(
        {
            "price": prices,
            "vol": vol,
            "vol_ok": vol_ok,
            "trend": trend,
            "hsi": hsi,
            "conf2": conf2,
            "label": label,
            "leverage": final,
            "change_day": change_day,
        },
        index=dates,
    )


def build_bobl(md: MarketData, cfg: Config) -> pd.DataFrame:
    leg = cfg.bobl
    prices = md.futures["bobl"]
    dates = prices.index
    change_day = contract_change_days(dates)
    returns = prices.pct_change()
    if cfg.exclude_roll_from_vol:
        returns = returns.mask(change_day)

    vol, vol_ok = _vol_filter(returns, leg, cfg)
    main = main_signal(prices, leg)
    if leg.confirmation == "slope":
        conf = slope_confirmation(md.slope, leg.slope, dates)
    elif leg.confirmation == "nfp":
        conf = nfp_signal(prices, md.nfp, change_day, cfg.nfp.method, cfg.nfp.hold_sessions)
    else:
        conf = np.full(len(dates), NEUTRAL, dtype=object)

    label = np.empty(len(dates), dtype=object)
    for i in range(len(dates)):
        m, c = main[i], conf[i]
        if not vol_ok[i] or m == NEUTRAL:
            label[i] = NOK
        elif leg.confirmation == "none" or m == c:
            label[i] = m
        elif c == NEUTRAL:
            label[i] = "Mid-" + m
        else:
            label[i] = NOK

    rebal = _rebalance_mask(dates, leg)
    table = leg.leverage
    multi_fn = None
    if leg.signal.multi_windows and leg.signal.method == "student":
        sigs = multi_window_signals(prices, leg)

        def multi_fn(i: int) -> float:
            if not sigs:
                return 0.0
            n_long = sum(s[i] == LONG for s in sigs)
            n_short = sum(s[i] == SHORT for s in sigs)
            c = conf[i]
            if leg.confirmation == "none":
                lv_long, lv_short = table[LONG], table[SHORT]
            else:
                lv_long = table["Mid-Long" if c == NEUTRAL else LONG if c == LONG else NOK]
                lv_short = table["Mid-Short" if c == NEUTRAL else SHORT if c == SHORT else NOK]
            return (n_long * lv_long + n_short * lv_short) / len(sigs)

    lev = _leverage_path(label, rebal, vol_ok, table, multi_fn)

    vol60 = realized_vol(returns, cfg.vol_target_window)
    final = _apply_vol_target(lev, rebal, vol60, leg, cfg)
    return pd.DataFrame(
        {
            "price": prices,
            "vol": vol,
            "vol_ok": vol_ok,
            "trend": main,
            "conf": conf,
            "label": label,
            "leverage": final,
            "change_day": change_day,
        },
        index=dates,
    )
