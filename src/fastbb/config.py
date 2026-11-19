"""Strategy configuration.

Every parameter of the Excel ``Paramètres`` sheet has a typed counterpart here.
Configurations are stored as YAML files (see ``config/v10.yaml``) so that each
backtest is fully reproducible from a versioned file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

SignalMethod = Literal["student", "moving_averages", "macd", "rsi"]
Label = Literal["Long", "Mid-Long", "Short", "Mid-Short", "NOK"]

DEFAULT_WINDOWS = (5, 6, 8, 10, 12, 14, 16, 20, 25, 30)


@dataclass
class SignalParams:
    """Main trend signal computed on the futures price."""

    method: SignalMethod = "student"
    # Student t-test on the last N daily price changes
    t_obs: int = 12
    t_proba: float = 0.005  # one-sided significance level
    persistence: bool = True  # keep the previous signal while |t| < threshold
    multi_windows: bool = False  # average the leverage over several windows
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    # Triple moving average
    ma: tuple[int, int, int] = (3, 30, 100)
    # MACD
    macd: tuple[int, int, int] = (12, 26, 9)
    macd_rule: Literal["cross", "sign"] = "cross"
    # Wilder RSI
    rsi_period: int = 14
    rsi_rule: Literal["trend", "mean_reversion"] = "trend"
    rsi_high: float = 70.0
    rsi_low: float = 30.0
    rsi_persistence: bool = False
    # Volatility filter (annualised, computed on the N previous sessions)
    vol_window: int = 13
    vol_min: float = 0.0
    vol_max: float = 0.05


@dataclass
class MAConfirmation:
    """Triple moving-average confirmation on an external index (Hang Seng, Dow Jones)."""

    ma: tuple[int, int, int] = (3, 30, 100)
    persistence: bool = False
    inverted: bool = False
    lag_days: int = 0  # calendar days subtracted from the futures date


@dataclass
class SlopeConfirmation:
    """Student t-test on the EUR 2y-5y swap slope (Bobl confirmation)."""

    t_obs: int = 8
    t_proba: float = 0.08
    persistence: bool = False
    inverted: bool = False
    lag_days: int = 0


@dataclass
class NFPParams:
    """US Non-Farm Payrolls signal."""

    method: Literal["day_reaction", "vs_3m_pace", "trend_3m_12m"] = "day_reaction"
    hold_sessions: int = 0  # 0 = valid until the next release


def _default_leverage() -> dict[str, float]:
    return {"Long": 3.0, "Mid-Long": 2.0, "Short": -3.0, "Mid-Short": -1.0, "NOK": 0.0}


@dataclass
class LegParams:
    """Parameters common to both legs (Bund, Bobl)."""

    active: bool = True
    weight: float = 1.0
    signal: SignalParams = field(default_factory=SignalParams)
    leverage: dict[str, float] = field(default_factory=_default_leverage)
    stop_loss_days: int = 3  # circuit breaker after N consecutive losing sessions
    rebalance: Literal["daily", "weekly"] = "daily"
    rebalance_weekday: int = 1  # 1 = Monday ... 5 = Friday
    vol_ref: float = 0.046  # used by the volatility-target method only
    lev_max: float = 4.5


@dataclass
class BundParams(LegParams):
    hsi: MAConfirmation = field(default_factory=lambda: MAConfirmation(ma=(3, 30, 120)))
    second_confirmation: Literal["dow_jones", "nfp", "none"] = "dow_jones"
    dow_jones: MAConfirmation = field(
        default_factory=lambda: MAConfirmation(ma=(3, 30, 100), lag_days=1)
    )


@dataclass
class BoblParams(LegParams):
    confirmation: Literal["none", "slope", "nfp"] = "none"
    slope: SlopeConfirmation = field(default_factory=SlopeConfirmation)


@dataclass
class CostParams:
    brokerage_per_contract: float = 2.0  # EUR per contract per side
    slippage_ticks: float = 0.5  # ticks per contract per side
    management_fee: float = 0.005  # per year, ACT/365 on start-of-day NAV
    performance_fee: float = 0.0  # on year-end NAV above the high-water mark


@dataclass
class CashParams:
    share: float = 1.0  # share of NAV earning the ECB deposit rate
    margin: float = 0.0  # spread over the ECB deposit rate
    apply_negative_rates: bool = True
    day_count: int = 360  # ACT/360


@dataclass
class Config:
    name: str = "FAST v10"
    initial_aum: float = 1_000_000.0
    start_date: date = date(2015, 1, 2)
    nominal: float = 100_000.0
    tick: float = 0.01
    global_leverage: float = 2.5
    risk_method: Literal["vol_filter", "vol_target"] = "vol_filter"
    exclude_roll_from_vol: bool = False
    vol_target_window: int = 60
    roll_offset_days: int = 0  # business days before the Eurex last trading day
    bund: BundParams = field(default_factory=BundParams)
    bobl: BoblParams = field(default_factory=BoblParams)
    nfp: NFPParams = field(default_factory=NFPParams)
    costs: CostParams = field(default_factory=CostParams)
    cash: CashParams = field(default_factory=CashParams)

    # ------------------------------------------------------------------ derived
    @property
    def point_value(self) -> float:
        """EUR value of one price point for one contract (1 000 EUR)."""
        return self.nominal / 100

    @property
    def tick_value(self) -> float:
        """EUR value of one tick for one contract (10 EUR)."""
        return self.tick * self.point_value

    # ---------------------------------------------------------------------- I/O
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return _from_dict(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_to_plain(asdict(self)), fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------- helpers
def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build nested dataclasses from a plain dict (YAML)."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown parameter(s) for {cls.__name__}: {sorted(unknown)}")
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint) and isinstance(value, dict):
            # merge with defaults so YAML files only need to state what changes
            default = next(f for f in fields(cls) if f.name == name).default_factory()  # type: ignore[misc]
            merged = {**_to_plain(asdict(default)), **value}
            kwargs[name] = _from_dict(hint, merged)
        elif get_origin(hint) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        elif hint is date and isinstance(value, str):
            kwargs[name] = date.fromisoformat(value)
        elif get_origin(hint) is Literal and value not in get_args(hint):
            raise ValueError(f"{cls.__name__}.{name} must be one of {get_args(hint)}, got {value!r}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj
