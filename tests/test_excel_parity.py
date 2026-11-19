"""Parity with the Excel model (Modele_FAST_Bund_Bobl_v10.xlsx).

Runs only when the licensed market data has been imported locally:

    fastbb import-excel Modele_FAST_Bund_Bobl_v10.xlsx --out data/raw

Expected values were produced by Excel (v10 settings) and by a LibreOffice
recalculation of the same workbook with the settings of each variant.
Variants A-C together exercise every signal method and every switch.
"""

import copy

import numpy as np
import pytest

from fastbb import MarketData, run_backtest
from fastbb.risk import annual_table, current_var, historical_var

from .conftest import RAW

pytestmark = pytest.mark.skipif(not (RAW / "futures.csv").exists(), reason="licensed data not available")

VARIANTS = {
    "v10": ({}, 5_126_272.34),
    "A: Student multi-window, Dow Jones, roll excluded, weekly Bobl": (
        {"bund.signal.method": "student", "bund.second_confirmation": "dow_jones",
         "bund.signal.multi_windows": True, "bobl.signal.multi_windows": True,
         "exclude_roll_from_vol": True, "bobl.rebalance": "weekly", "bobl.rebalance_weekday": 5},
        1_265_345.52,
    ),
    "B: MA / MACD sign, vol target, weekly Bund, perf fee, no negative rates": (
        {"bund.signal.method": "moving_averages", "bund.second_confirmation": "dow_jones",
         "bobl.signal.method": "macd", "bobl.signal.macd_rule": "sign", "risk_method": "vol_target",
         "bund.rebalance": "weekly", "bund.rebalance_weekday": 1, "costs.performance_fee": 0.2,
         "cash.apply_negative_rates": False, "cash.margin": -0.001},
        304_553.95,
    ),
    "C: MACD cross / RSI mean reversion, slope confirmation, NFP held 5 sessions": (
        {"bobl.confirmation": "slope", "bund.signal.method": "macd", "bund.signal.macd_rule": "cross",
         "bobl.signal.method": "rsi", "bobl.signal.rsi_rule": "mean_reversion",
         "bobl.signal.rsi_persistence": True, "nfp.hold_sessions": 5},
        1_401_663.88,
    ),
}


@pytest.fixture(scope="module")
def raw():
    return MarketData.from_csv(RAW)


def _apply(cfg, mods):
    cfg = copy.deepcopy(cfg)
    for path, value in mods.items():
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return cfg


@pytest.mark.parametrize("name", list(VARIANTS))
def test_final_nav_matches_excel(raw, v10, name):
    mods, expected = VARIANTS[name]
    res = run_backtest(raw, _apply(v10, mods))
    assert res.nav.iloc[-1] == pytest.approx(expected, abs=0.01)


def test_risk_report_matches_pdf(raw, v10):
    """Figures of 'Risque_consomme_FAST_v10.pdf'."""
    res = run_backtest(raw, v10)
    var = historical_var(res)
    table = annual_table(res, var)
    assert int(var.invested.sum()) == 2400 and int(var.exceptions.sum()) == 38
    np.testing.assert_allclose(
        (table["var99_max"] * 100).round(1).to_numpy(),
        [7.5, 7.6, 6.0, 5.9, 4.0, 7.0, 7.0, 8.5, 16.1, 15.6, 8.3, 5.7],
    )
    assert list(table["ucits_breach_days"]) == [12, 67, 89, 14, 0, 56, 14, 52, 108, 121, 30, 46]
    pct, eur = current_var(res)
    assert round(pct, 3) == 0.034 and round(eur / 1e3) == 173
