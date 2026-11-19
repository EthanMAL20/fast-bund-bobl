"""Market data loading.

Two sources are supported:

* the Excel workbook ``Modele_FAST_Bund_Bobl_vX.xlsx`` (data sheets only), and
* a directory of CSV files (``futures.csv``, ``hsi.csv``, ...), which is the
  format used by the engine and by the synthetic sample in ``data/sample``.

Raw Bloomberg data must never be committed to a public repository: keep it in
``data/raw/`` (ignored by git).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CSV_FILES = {
    "futures": "futures.csv",  # date, bund, bobl  (RX1 / OE1 settlement prices)
    "hsi": "hsi.csv",  # date, value
    "dow_jones": "dow_jones.csv",  # date, value
    "slope": "slope_2y5y.csv",  # date, value  (EUSA2 - EUSA5, in %)
    "nfp": "nfp.csv",  # ref_month, release_date, change
    "ecb": "ecb_deposit_rate.csv",  # effective_date, rate
    "eurostoxx_tr": "eurostoxx50_tr.csv",  # optional benchmark
    "cac_tr": "cac40_tr.csv",  # optional benchmark
}


@dataclass
class MarketData:
    futures: pd.DataFrame  # index: date ; columns: bund, bobl
    hsi: pd.Series
    dow_jones: pd.Series
    slope: pd.Series
    nfp: pd.DataFrame  # columns: ref_month, release_date, change
    ecb: pd.Series  # index: effective date ; value: deposit rate (decimal)
    eurostoxx_tr: pd.Series | None = None
    cac_tr: pd.Series | None = None

    def validate(self) -> None:
        """Basic sanity checks, mirroring the 'Contrôles des données' block."""
        f = self.futures
        if not f.index.is_monotonic_increasing or f.index.has_duplicates:
            raise ValueError("futures dates must be strictly increasing")
        if f[["bund", "bobl"]].isna().any().any():
            raise ValueError("missing Bund/Bobl settlement prices")
        same_as_previous = (f.diff().abs().sum(axis=1) == 0).iloc[1:]
        if same_as_previous.any():
            days = list(same_as_previous[same_as_previous].index.date[:5])
            warnings.warn(f"Bund and Bobl settlements unchanged vs previous day (holiday copied?): {days}", stacklevel=2)
        for name in ("hsi", "dow_jones", "slope"):
            s = getattr(self, name)
            if not s.index.is_monotonic_increasing:
                raise ValueError(f"{name} dates must be increasing")

    # ------------------------------------------------------------------ CSV I/O
    @classmethod
    def from_csv(cls, directory: str | Path) -> MarketData:
        d = Path(directory)

        def series(key: str, required: bool = True) -> pd.Series | None:
            path = d / CSV_FILES[key]
            if not path.exists():
                if required:
                    raise FileNotFoundError(path)
                return None
            df = pd.read_csv(path, parse_dates=["date"])
            return df.set_index("date")["value"].astype(float).sort_index()

        futures = pd.read_csv(d / CSV_FILES["futures"], parse_dates=["date"]).set_index("date")
        nfp = pd.read_csv(d / CSV_FILES["nfp"], parse_dates=["ref_month", "release_date"])
        ecb = pd.read_csv(d / CSV_FILES["ecb"], parse_dates=["effective_date"])
        return cls(
            futures=futures[["bund", "bobl"]].astype(float).sort_index(),
            hsi=series("hsi"),  # type: ignore[arg-type]
            dow_jones=series("dow_jones"),  # type: ignore[arg-type]
            slope=series("slope"),  # type: ignore[arg-type]
            nfp=nfp.sort_values("release_date").reset_index(drop=True),
            ecb=ecb.set_index("effective_date")["rate"].astype(float).sort_index(),
            eurostoxx_tr=series("eurostoxx_tr", required=False),
            cac_tr=series("cac_tr", required=False),
        )

    def to_csv(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.futures.rename_axis("date").to_csv(d / CSV_FILES["futures"])
        for key in ("hsi", "dow_jones", "slope", "eurostoxx_tr", "cac_tr"):
            s = getattr(self, key)
            if s is not None:
                s.rename("value").rename_axis("date").to_csv(d / CSV_FILES[key])
        self.nfp.to_csv(d / CSV_FILES["nfp"], index=False)
        self.ecb.rename("rate").rename_axis("effective_date").to_csv(d / CSV_FILES["ecb"])

    # ------------------------------------------------------------- Excel import
    @classmethod
    def from_workbook(cls, path: str | Path) -> MarketData:
        """Read the data sheets of the Excel model (values only, no formulas)."""
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

        def block(sheet: str, date_col: int, value_cols: list[int], first_row: int = 6):
            out = []
            for row in wb[sheet].iter_rows(min_row=first_row, values_only=True):
                d = row[date_col] if date_col < len(row) else None
                if d is None or d == "":
                    continue
                vals = [row[c] if c < len(row) else None for c in value_cols]
                out.append([pd.Timestamp(d), *vals])
            return out

        fut = pd.DataFrame(block("Données_Futures", 0, [1, 2]), columns=["date", "bund", "bobl"])
        fut = fut.dropna().set_index("date")

        def idx_series(date_col: int, value_col: int) -> pd.Series:
            rows = block("Données_Indices", date_col, [value_col])
            s = pd.Series({d: v for d, v in rows}, dtype="float64")
            return s.dropna().sort_index()

        hsi = idx_series(0, 1)
        dj = idx_series(3, 4)
        # Slope = EUSA2 - EUSA5 when both swaps are available, else historical .SEUR2-5
        rows = block("Données_Indices", 6, [7, 8, 9])
        slope = pd.Series(
            {
                d: (s2 - s5) if _num(s2) and _num(s5) else hist
                for d, s2, s5, hist in rows
            },
            dtype="float64",
        ).sort_index()
        sx5t = idx_series(11, 12)
        cacr = idx_series(14, 15)

        nfp_rows = block("Données_NFP", 0, [1, 2])
        nfp = pd.DataFrame(nfp_rows, columns=["ref_month", "release_date", "change"])
        nfp["release_date"] = pd.to_datetime(nfp["release_date"])
        nfp = nfp.dropna(subset=["release_date"])

        ecb_rows = block("Taux_BCE", 0, [1])
        ecb = pd.Series({d: v for d, v in ecb_rows}, dtype="float64").sort_index()

        return cls(
            futures=fut.astype(float),
            hsi=hsi,
            dow_jones=dj,
            slope=slope,
            nfp=nfp.reset_index(drop=True),
            ecb=ecb,
            eurostoxx_tr=sx5t,
            cac_tr=cacr,
        )


def _num(x: object) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x))
