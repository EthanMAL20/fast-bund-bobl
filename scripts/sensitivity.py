"""Parameter sensitivity of the Bobl Student signal (overfitting check).

Runs the full backtest on a grid of (window, significance level) for the Bobl
leg and plots the cumulative return. A robust signal shows a smooth surface;
an isolated peak means the result depends on one lucky setting.

    python scripts/sensitivity.py --data data/raw --config config/v10.yaml --out docs
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from fastbb import Config, MarketData, run_backtest
from fastbb.report import INK, INK2, SURFACE

WINDOWS = [6, 8, 10, 12, 14, 16, 20, 25, 30]
PROBAS = [0.0025, 0.005, 0.01, 0.025, 0.05]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config/v10.yaml")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    md = MarketData.from_csv(args.data)
    base = Config.from_yaml(args.config)
    grid = pd.DataFrame(index=PROBAS, columns=WINDOWS, dtype=float)
    for p in PROBAS:
        for n in WINDOWS:
            cfg = copy.deepcopy(base)
            cfg.bobl.signal.t_obs, cfg.bobl.signal.t_proba = n, p
            res = run_backtest(md, cfg)
            grid.loc[p, n] = res.nav.iloc[-1] / cfg.initial_aum - 1
            print(f"window {n:2d}  p={p:<6}  {grid.loc[p, n]:+.0%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "sensitivity_bobl.csv")

    # diverging red <-> gray <-> blue, centred on 0 % (loss vs gain)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    vmax = np.nanmax(np.abs(grid.to_numpy()))
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(grid.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(WINDOWS)), [f"{w} d" for w in WINDOWS])
    ax.set_yticks(range(len(PROBAS)), [f"{p:.2%}" for p in PROBAS])
    ax.set_xlabel("Student test window (sessions)", color=INK2, fontsize=9)
    ax.set_ylabel("One-sided significance level", color=INK2, fontsize=9)
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=8, color=INK)
    r = PROBAS.index(base.bobl.signal.t_proba) if base.bobl.signal.t_proba in PROBAS else None
    c = WINDOWS.index(base.bobl.signal.t_obs) if base.bobl.signal.t_obs in WINDOWS else None
    if r is not None and c is not None:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=INK2)
    ax.set_title("Cumulative return by Bobl signal setting (framed: v10 setting)", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, format=lambda v, _: f"{v:+.0%}", shrink=0.8).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_bobl.png", facecolor=SURFACE)
    print(f"written {out / 'sensitivity_bobl.png'}")


if __name__ == "__main__":
    main()
