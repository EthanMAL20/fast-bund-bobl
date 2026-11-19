import pandas as pd
import pytest

from fastbb import Config, run_backtest
from fastbb.walkforward import period_stats, set_param, walk_forward

from .conftest import RAW, ROOT

GRID = {"bund.signal.rsi_period": [10, 14, 20], "bund.signal.vol_max": [0.04, 0.05, 0.06]}


def test_set_param_copies_and_validates(v10):
    cfg = set_param(v10, "bund.signal.rsi_period", 20)
    assert cfg.bund.signal.rsi_period == 20 and v10.bund.signal.rsi_period == 14
    with pytest.raises(AttributeError):
        set_param(v10, "bund.signal.no_such_param", 1)


def test_walk_forward_uses_only_past_data(sample_data, v10):
    grid = {"bund.signal.rsi_period": [10, 20]}
    wf = walk_forward(sample_data, v10, grid, first_test_year=2018, last_test_year=2020)
    assert list(wf.choices.test_year) == [2018, 2019, 2020]
    assert set(wf.returns.index.year) == {2018, 2019, 2020}
    for row in wf.choices.itertuples(index=False):
        cfg = set_param(v10, "bund.signal.rsi_period", row[2])
        full = run_backtest(sample_data, cfg).portfolio["ret"]
        kept = wf.returns.loc[str(row.test_year), "ret"]
        pd.testing.assert_series_equal(kept, full.loc[str(row.test_year)], check_names=False)


def test_period_stats_on_flat_returns():
    idx = pd.bdate_range("2020-01-01", periods=300)
    pf = pd.DataFrame({"ret": 0.0001, "cash_rate": 0.0}, index=idx)
    s = period_stats(pf)
    assert s["max_drawdown"] == 0 and s["annualised_return"] > 0


@pytest.mark.skipif(not (RAW / "futures.csv").exists(), reason="licensed data not available")
def test_walk_forward_real_data():
    from fastbb import MarketData

    md = MarketData.from_csv(RAW)
    v11 = Config.from_yaml(ROOT / "config" / "v11.yaml")
    assert run_backtest(md, v11).nav.iloc[-1] == pytest.approx(1_676_950.30, abs=0.01)
    wf = walk_forward(md, v11, GRID)
    assert list(wf.choices["bund.signal.rsi_period"]) == [20, 14, 14, 14, 20, 20, 10, 20, 10]
    assert wf.stats["sharpe"] == pytest.approx(0.683, abs=1e-3)
