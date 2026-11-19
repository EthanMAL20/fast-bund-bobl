import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fastbb.calendar import eurex_last_trading_days, roll_dates
from fastbb.indicators import (
    LONG,
    NEUTRAL,
    SHORT,
    ema,
    realized_vol,
    rolling_tstat,
    rsi_wilder,
    t_threshold,
    threshold_signal,
    triple_ma_signal,
)

rng = np.random.default_rng(0)
PRICES = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 300)), index=pd.bdate_range("2020-01-01", periods=300))


def test_tstat_matches_scipy_one_sample_test():
    changes = PRICES.diff()
    t = rolling_tstat(changes, 12)
    window = changes.iloc[100 - 11 : 101]
    expected = stats.ttest_1samp(window, 0.0).statistic
    assert t.iloc[100] == pytest.approx(expected, rel=1e-12)
    assert t.iloc[:12].isna().all() and not np.isnan(t.iloc[12])


def test_t_threshold_equals_excel_tinv():
    # Excel: TINV(2*0.5%, 11) = 3.105806516...
    assert t_threshold(0.005, 12) == pytest.approx(3.1058065155, rel=1e-9)


def test_threshold_signal_persistence():
    t = pd.Series([np.nan, 0.5, 4.0, 1.0, -0.2, -5.0, 0.0])
    assert list(threshold_signal(t, 3.0, True)) == [NEUTRAL, NEUTRAL, LONG, LONG, LONG, SHORT, SHORT]
    assert list(threshold_signal(t, 3.0, False)) == [NEUTRAL, NEUTRAL, LONG, NEUTRAL, NEUTRAL, SHORT, NEUTRAL]


def test_zero_volatility_gives_zero_t():
    flat = pd.Series([1.0] * 20)
    assert rolling_tstat(flat.diff(), 5).iloc[10] == 0.0


def test_triple_ma_alignment():
    up = pd.Series(np.arange(1.0, 200.0))
    sig = triple_ma_signal(up, (3, 30, 100), persistence=False)
    assert sig[98] == NEUTRAL and sig[99] == LONG
    assert triple_ma_signal(up, (3, 30, 100), False, inverted=True)[150] == SHORT


def test_ema_seeded_with_first_value():
    e = ema(PRICES, 12)
    alpha = 2 / 13
    manual = PRICES.iloc[0]
    for x in PRICES.iloc[1:6]:
        manual = alpha * x + (1 - alpha) * manual
    assert e.iloc[5] == pytest.approx(manual)


def test_rsi_bounds_and_extremes():
    rsi = rsi_wilder(PRICES, 14).dropna()
    assert rsi.between(0, 100).all()
    assert rsi_wilder(pd.Series(np.arange(50.0)), 14).iloc[-1] == 100.0


def test_realized_vol_uses_previous_sessions_only():
    rets = PRICES.pct_change()
    vol = realized_vol(rets, 13)
    assert vol.iloc[:14].isna().all()
    expected = rets.iloc[1:14].std(ddof=1) * np.sqrt(252)
    assert vol.iloc[14] == pytest.approx(expected)


def test_eurex_last_trading_days():
    ltd = eurex_last_trading_days(2013, 2014)
    assert pd.Timestamp("2013-03-07") in ltd  # delivery Mon 11/03
    assert pd.Timestamp("2014-09-08") in ltd  # delivery Wed 10/09
    assert len(ltd) == 8
    assert (roll_dates(0, 2013, 2014) == ltd).all()
