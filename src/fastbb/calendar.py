"""Eurex roll calendar for Bund (FGBL) and Bobl (FGBM) futures.

Quarterly cycle (March, June, September, December). Delivery day is the 10th
of the month (or the next business day); the last trading day is two business
days before delivery. Exchange holidays are ignored, as in the Excel model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eurex_last_trading_days(first_year: int = 2013, last_year: int = 2031) -> pd.DatetimeIndex:
    out = []
    for year in range(first_year, last_year + 1):
        for month in (3, 6, 9, 12):
            tenth = np.datetime64(f"{year}-{month:02d}-10")
            delivery = np.busday_offset(tenth, 0, roll="forward")
            out.append(np.busday_offset(delivery, -2, roll="backward"))
    return pd.DatetimeIndex(out)


def roll_dates(offset_days: int = 0, first_year: int = 2013, last_year: int = 2031) -> pd.DatetimeIndex:
    """Dates on which the position is closed and re-opened on the next contract.

    ``offset_days = 0`` (default) rolls on the last trading day, which is the
    only correct choice with generic RX1 / OE1 series: they only switch to the
    next contract on the following session.
    """
    ltd = eurex_last_trading_days(first_year, last_year).values.astype("datetime64[D]")
    return pd.DatetimeIndex(np.busday_offset(ltd, -offset_days, roll="backward"))


def contract_change_days(dates: pd.DatetimeIndex, first_year: int = 2013, last_year: int = 2031) -> np.ndarray:
    """Boolean mask: True on the session following a last trading day.

    On that session the generic series jumps from one contract to the next; the
    jump is not a market move.
    """
    ltd = set(eurex_last_trading_days(first_year, last_year))
    prev = dates.to_series().shift(1)
    return prev.isin(ltd).to_numpy()
