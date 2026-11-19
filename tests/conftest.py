from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fastbb import Config, MarketData, run_backtest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def sample_data() -> MarketData:
    if not (SAMPLE / "futures.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py"), "--out", str(SAMPLE)], check=True)
    return MarketData.from_csv(SAMPLE)


@pytest.fixture(scope="session")
def v10() -> Config:
    return Config.from_yaml(ROOT / "config" / "v10.yaml")


@pytest.fixture(scope="session")
def sample_result(sample_data, v10):
    return run_backtest(sample_data, v10)
