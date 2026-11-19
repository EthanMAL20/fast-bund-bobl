"""FAST — systematic directional strategy on Eurex Bund and Bobl futures."""

from .backtest import BacktestResult, run_backtest
from .config import Config
from .data import MarketData

__all__ = ["BacktestResult", "Config", "MarketData", "run_backtest"]
__version__ = "1.0.0"
