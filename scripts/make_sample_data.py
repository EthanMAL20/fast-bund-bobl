"""Generate a synthetic data set with the same format as the real one.

The real inputs (Bloomberg RX1 / OE1 settlements, Hang Seng, Dow Jones, EUR
swaps) are licensed and must not be published. This script writes random-walk
series with realistic volatilities to ``data/sample`` so that anyone can clone
the repository and run the full pipeline and the test suite.

    python scripts/make_sample_data.py --out data/sample --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ECB deposit facility rate (public data, ECB website)
ECB_DEPOSIT_RATE = [
    ("2012-07-11", 0.0), ("2014-06-11", -0.001), ("2014-09-10", -0.002), ("2015-12-09", -0.003),
    ("2016-03-16", -0.004), ("2019-09-18", -0.005), ("2022-07-27", 0.0), ("2022-09-14", 0.0075),
    ("2022-11-02", 0.015), ("2022-12-21", 0.02), ("2023-02-08", 0.025), ("2023-03-22", 0.03),
    ("2023-05-10", 0.0325), ("2023-06-21", 0.035), ("2023-08-02", 0.0375), ("2023-09-20", 0.04),
    ("2024-06-12", 0.0375), ("2024-09-18", 0.035), ("2024-10-23", 0.0325), ("2024-12-18", 0.03),
    ("2025-02-05", 0.0275), ("2025-03-12", 0.025), ("2025-04-23", 0.0225), ("2025-06-11", 0.02),
]


def random_walk(rng: np.random.Generator, n: int, start: float, daily_vol: float, drift: float = 0.0) -> np.ndarray:
    # regime-switching volatility so that the volatility filter is exercised
    regime = np.repeat(rng.choice([0.7, 1.0, 1.8], size=n // 60 + 1, p=[0.4, 0.45, 0.15]), 60)[:n]
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # fat tails
    trend = np.cumsum(rng.normal(0, 0.02, n)) * daily_vol  # slow persistent drift
    rets = drift + trend * 0.05 + shocks * daily_vol * regime
    return start * np.exp(np.cumsum(rets))


def main(out: Path, seed: int, start: str = "2014-01-02", end: str = "2026-10-20") -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, end)

    pd.DataFrame(
        {
            "date": days,
            "bund": np.round(random_walk(rng, len(days), 140.0, 0.0035), 2),
            "bobl": np.round(random_walk(rng, len(days), 124.0, 0.0016), 2),
        }
    ).to_csv(out / "futures.csv", index=False)
    for name, level, vol in (("hsi", 23000, 0.012), ("dow_jones", 16500, 0.010)):
        pd.DataFrame({"date": days, "value": np.round(random_walk(rng, len(days), level, vol, 0.0002), 2)}).to_csv(
            out / f"{name}.csv", index=False
        )
    slope = -0.7 + np.cumsum(rng.normal(0, 0.01, len(days)))
    pd.DataFrame({"date": days, "value": np.round(slope, 3)}).to_csv(out / "slope_2y5y.csv", index=False)

    months = pd.date_range("2014-01-01", end, freq="MS")
    releases = [(m + pd.offsets.MonthBegin(1)) + pd.offsets.Week(weekday=4) for m in months]  # first Friday after
    nfp = pd.DataFrame(
        {"ref_month": months, "release_date": releases, "change": rng.normal(180, 90, len(months)).round()}
    )
    nfp[nfp["release_date"] <= pd.Timestamp(end)].to_csv(out / "nfp.csv", index=False)
    pd.DataFrame(ECB_DEPOSIT_RATE, columns=["effective_date", "rate"]).to_csv(out / "ecb_deposit_rate.csv", index=False)
    print(f"synthetic data written to {out} ({len(days)} sessions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    a = ap.parse_args()
    main(a.out, a.seed)
