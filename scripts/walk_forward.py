"""Walk-forward validation of the v11 Bund strategy, and the Bobl attribution check.

    python scripts/walk_forward.py --data data/raw --config config/v11.yaml --out docs

Writes docs/walk_forward.md, docs/walk_forward.csv and docs/walk_forward.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastbb import Config, MarketData, run_backtest
from fastbb.report import AXIS, BLUE, INK2, MUTED, ORANGE, SURFACE, _fig, _style, buy_and_hold
from fastbb.walkforward import period_stats, set_param, walk_forward

GRID = {"bund.signal.rsi_period": [10, 14, 20], "bund.signal.vol_max": [0.04, 0.05, 0.06]}
REGIME_SWITCH = "2022-01-07"  # first session of the v10 Bobl short signal


def bobl_vs_naive(md: MarketData, v10: Config) -> pd.DataFrame:
    """Yearly sum of daily returns: v10 Bobl leg vs 'long until the switch, short after' at constant leverage."""
    res = run_backtest(md, v10)
    pf, bobl = res.portfolio, res.bobl
    leg = pf["bobl_pnl"] / pf["nav_open"]
    change = bobl["price"].diff().mask(bobl["change_day"], 0.0)
    ret = (change / bobl["price"].shift(1)).reindex(pf.index)
    lev = v10.bobl.leverage["Long"] * v10.global_leverage * v10.bobl.weight
    naive = lev * np.where(pf.index < pd.Timestamp(REGIME_SWITCH), 1.0, -1.0) * ret
    out = pd.DataFrame({"v10_bobl_leg": leg, "naive_regime_bet": naive}).groupby(pf.index.year).sum()
    out.loc["Total"] = out.sum()
    return out


def plot_walk_forward(oos: pd.DataFrame, fixed: pd.DataFrame, bund_bh: pd.Series, path: Path) -> None:
    """Out-of-sample NAV vs fixed v11 settings and Bund buy & hold, base 100."""
    fig, ax = _fig()
    series = [
        ("Walk-forward (out-of-sample)", 100 * (1 + oos["ret"]).cumprod(), BLUE, 2.0),
        ("Fixed v11 settings", 100 * (1 + fixed["ret"]).cumprod(), MUTED, 1.4),
        ("Bund buy & hold", 100 * bund_bh / bund_bh.iloc[0], ORANGE, 1.4),
    ]
    for label, s, color, lw in series:
        ax.plot(s.index, s, color=color, linewidth=lw, label=label)
        ax.annotate(f"{s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.axhline(100, color=AXIS, linewidth=1)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    _style(ax, "Walk-forward validation, base 100",
           f"Bund strategy · {oos.index[0]:%d/%m/%Y} – {oos.index[-1]:%d/%m/%Y} · parameters recalibrated each year on the 3 previous years · net of costs")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{x:+.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config/v11.yaml")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = MarketData.from_csv(args.data)
    v11 = Config.from_yaml(args.config)
    wf = walk_forward(md, v11, GRID)
    res11 = run_backtest(md, v11)
    first = str(wf.choices.test_year.min())
    fixed = res11.portfolio.loc[first:]
    plot_walk_forward(wf.returns, fixed, buy_and_hold(res11)["bund"].loc[first:], out / "walk_forward.png")
    oos, ins = wf.stats, period_stats(fixed)
    ex2020 = period_stats(wf.returns[wf.returns.index.year != 2020])
    wf.choices.to_csv(out / "walk_forward.csv", index=False)

    v10 = set_param(set_param(v11, "bobl.active", True), "name", "v10")
    attrib = bobl_vs_naive(md, v10)

    lines = [
        "# Walk-forward validation — v11 (Bund only)",
        "",
        f"Grid: RSI period {GRID['bund.signal.rsi_period']} x volatility-filter cap "
        f"{GRID['bund.signal.vol_max']}. Each test year, the configuration with the best Sharpe ratio "
        "over the 3 previous calendar years is frozen and traded for that year. Net of all costs.",
        "",
        "| Test year | Calibration | RSI | Vol cap | Out-of-sample return |",
        "|---|---|---:|---:|---:|",
    ]
    for r in wf.choices.itertuples(index=False):
        lines.append(f"| {r.test_year} | {r.train} | {r[2]} | {r[3]:.0%} | {pct(r.oos_return)} |")
    lines += [
        "",
        f"| {wf.choices.test_year.min()}–{wf.choices.test_year.max()} | Out-of-sample | Fixed v11 settings |",
        "|---|---:|---:|",
    ]
    for key, label in [("annualised_return", "Annualised return"), ("annualised_vol", "Volatility"),
                       ("max_drawdown", "Max drawdown")]:
        lines.append(f"| {label} | {oos[key]:.1%} | {ins[key]:.1%} |")
    lines.append(f"| Sharpe ratio | {oos['sharpe']:.2f} | {ins['sharpe']:.2f} |")
    lines += [
        "",
        f"Excluding 2020, the out-of-sample Sharpe ratio is {ex2020['sharpe']:.2f}: the edge is concentrated "
        "in stress periods, as is typical of trend signals. The grid is centred on settings already known "
        "to work, so the test is somewhat optimistic.",
        "",
        "## Why the Bobl leg was removed",
        "",
        f"Naive rule: long Bobl until {REGIME_SWITCH}, short afterwards, constant leverage, no filter. "
        "Sum of daily returns (not compounded):",
        "",
        "| Year | v10 Bobl leg | Naive regime bet |",
        "|---|---:|---:|",
    ]
    for year, r in attrib.iterrows():
        lines.append(f"| {year} | {pct(r.v10_bobl_leg)} | {pct(r.naive_regime_bet)} |")
    (out / "walk_forward.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OOS Sharpe {oos['sharpe']:.2f} vs fixed {ins['sharpe']:.2f}; report written to {out}/walk_forward.md")


if __name__ == "__main__":
    main()
