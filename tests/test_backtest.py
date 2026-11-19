import copy

import numpy as np
import pytest

from fastbb import Config, run_backtest
from fastbb.backtest import excel_round
from fastbb.risk import annual_table, historical_var, kupiec_pof, summary


def test_excel_round_half_away_from_zero():
    assert excel_round(2.5) == 3 and excel_round(-2.5) == -3 and excel_round(0.49) == 0


def test_accounting_identity(sample_result):
    """Daily NAV change = futures P&L - costs + interest - fees, to the cent."""
    pf = sample_result.portfolio
    costs = pf[["bund_brokerage", "bund_slippage", "bobl_brokerage", "bobl_slippage"]].sum(axis=1)
    change = pf["bund_pnl"] + pf["bobl_pnl"] - costs + pf["interest"] - pf["mgmt_fee"] - pf["perf_fee"]
    np.testing.assert_allclose(pf["nav"] - pf["nav_open"], change, atol=1e-6)
    np.testing.assert_allclose(pf["nav_open"].iloc[1:].to_numpy(), pf["nav"].iloc[:-1].to_numpy())


def test_flat_after_roll(sample_result):
    """No position is carried through the generic-series jump."""
    pf = sample_result.portfolio
    after_roll = pf["roll"].shift(1, fill_value=False)
    assert (pf.loc[after_roll, ["bund_contracts_open", "bobl_contracts_open"]] == 0).all().all()


def test_position_sizing(sample_result, v10):
    pf = sample_result.portfolio
    row = pf[pf["bobl_contracts"] != 0].iloc[0]
    expected = row["nav_open"] * v10.global_leverage * v10.bobl.weight * row["bobl_leverage"] * 100 / (row["bobl_price"] * v10.nominal)
    assert row["bobl_contracts"] == excel_round(expected)


def test_no_strategy_means_cash_only(sample_data, v10):
    cfg = copy.deepcopy(v10)
    cfg.bund.active = cfg.bobl.active = False
    pf = run_backtest(sample_data, cfg).portfolio
    assert (pf["bund_contracts"] == 0).all() and (pf["bobl_contracts"] == 0).all()
    np.testing.assert_allclose(pf["nav"] - pf["nav_open"], pf["interest"] - pf["mgmt_fee"], atol=1e-9)


def test_circuit_breaker(sample_result, v10):
    pf = sample_result.portfolio
    hit = pf["bobl_loss_streak"] >= v10.bobl.stop_loss_days
    assert hit.any() and (pf.loc[hit, "bobl_contracts"] == 0).all()


def test_all_methods_run(sample_data, v10):
    for method in ("student", "moving_averages", "macd", "rsi"):
        cfg = copy.deepcopy(v10)
        cfg.bund.signal.method = cfg.bobl.signal.method = method
        cfg.bund.second_confirmation = "dow_jones"
        cfg.bobl.confirmation = "slope"
        assert np.isfinite(run_backtest(sample_data, cfg).nav.iloc[-1])
    cfg = copy.deepcopy(v10)
    cfg.bund.signal.method = "student"
    cfg.bund.signal.multi_windows = cfg.bobl.signal.multi_windows = True
    cfg.risk_method = "vol_target"
    assert np.isfinite(run_backtest(sample_data, cfg).nav.iloc[-1])


def test_risk_outputs(sample_result):
    s = summary(sample_result)
    assert s["max_drawdown"] <= 0 and s["annualised_vol"] > 0
    var = historical_var(sample_result)
    assert (var.var >= 0).all()
    assert (var.var[~var.invested] == 0).all()
    table = annual_table(sample_result, var)
    assert table.index[0] == 2015 and {"var99_avg", "ucits_breach_days"} <= set(table.columns)


def test_kupiec():
    assert kupiec_pof(1000, 10)["lr_stat"] == pytest.approx(0.0, abs=1e-9)
    assert kupiec_pof(2400, 38)["p_value"] < 0.01


def test_config_roundtrip(tmp_path, v10):
    path = tmp_path / "cfg.yaml"
    v10.to_yaml(path)
    assert Config.from_yaml(path) == v10


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("bund:\n  wieght: 0.5\n")
    with pytest.raises(ValueError, match="wieght"):
        Config.from_yaml(path)
