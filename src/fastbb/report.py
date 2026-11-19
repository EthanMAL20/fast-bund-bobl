"""Charts and report generation (matplotlib, static PNG for the README)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .risk import VaRResult, annual_table, current_var, historical_var, kupiec_pof, summary

# Reference palette (light mode): categorical slots in fixed order + chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL_GRAY = "#b9b8b1"


def _style(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def _fig(w: float = 9, h: float = 4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def buy_and_hold(res: BacktestResult) -> pd.DataFrame:
    """Bund / Bobl held long at 1x NAV, rolled quarterly, cash remunerated, no costs."""
    pf, cfg = res.portfolio, res.config
    out = {}
    for leg in ("bund", "bobl"):
        px = pf[f"{leg}_price"]
        ret = px.pct_change().where(~pf["roll"].shift(1, fill_value=False), 0.0).fillna(0.0)
        cash = cfg.cash.share * pf["cash_rate"] * pf["days"] / cfg.cash.day_count
        r = ret + cash
        r.iloc[0] = 0.0
        out[leg] = 100 * (1 + r).cumprod()
    return pd.DataFrame(out)


def plot_nav(res: BacktestResult, path: Path) -> None:
    fig, ax = _fig()
    nav = 100 * res.nav / res.config.initial_aum
    bh = buy_and_hold(res)
    series = [("FAST (net of costs)", nav, BLUE), ("Bund buy & hold", bh["bund"], ORANGE), ("Bobl buy & hold", bh["bobl"], AQUA)]
    # end-of-line labels, nudged apart when two series finish close together
    ends = sorted(((s.iloc[-1], label) for label, s, _ in series), reverse=True)
    offsets, last_y = {}, None
    for value, label in ends:
        y = np.log10(value)
        offsets[label] = -7 if last_y is not None and last_y - y < 0.03 else 0
        last_y = y
    for label, s, color in series:
        ax.plot(s.index, s, color=color, linewidth=2 if label.startswith("FAST") else 1.4, label=label)
        ax.annotate(f"{s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(6, offsets[label]),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_yticks([80, 100, 150, 200, 300, 500])
    ax.tick_params(which="minor", length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    _style(ax, "Net asset value, base 100 (log scale)",
           f"{res.config.name} · simulation {res.nav.index[0]:%d/%m/%Y} – {res.nav.index[-1]:%d/%m/%Y} · global leverage x{res.config.global_leverage}")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_drawdown(res: BacktestResult, path: Path) -> None:
    fig, ax = _fig(9, 3.2)
    peak = res.nav.cummax().clip(lower=res.config.initial_aum)
    dd = res.nav / peak - 1
    ax.fill_between(dd.index, dd, 0, color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(dd.index, dd, color=BLUE, linewidth=1.2)
    i = dd.idxmin()
    ax.annotate(f"{dd.min():.1%} on {i:%d/%m/%Y}", (i, dd.min()), xytext=(-140, -4), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax, "Drawdown from previous peak")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_var(var: VaRResult, path: Path) -> None:
    fig, ax = _fig(9, 3.6)
    v = var.var
    ax.plot(v.index, v, color=BLUE, linewidth=0.7, label="1-day VaR 99 % (each session)")
    yearly = v.groupby(v.index.year).transform("mean")
    ax.plot(yearly.index, yearly, color=ORANGE, linewidth=2, label="Yearly average")
    i = v.idxmax()
    ax.annotate(f"{v.max():.1%} on {i:%d/%m/%Y}", (i, v.max()), xytext=(-150, -2), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axhline(0.20 / np.sqrt(20), color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.0, 0.20 / np.sqrt(20) + 0.002, "UCITS limit: 20 % over 20 days ≈ 4.5 % per day", fontsize=8,
            color=INK2, ha="right", va="bottom", transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2, bbox_to_anchor=(0, 0.92))
    _style(ax, "Historical VaR 99 %, 1 day, in % of NAV", "Zero on flat sessions (volatility filter, circuit breaker, contract change)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_annual(table: pd.DataFrame, path: Path) -> None:
    fig, ax = _fig(9, 3.6)
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, table["return"], w, color=BLUE, label="Return")
    ax.bar(x + w / 2 + 0.01, table["vol"], w, color=ORANGE, label="Realised volatility")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [str(y) + ("*" if i == len(table) - 1 else "") for i, y in enumerate(table.index)])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2, ncols=2)
    _style(ax, "Return and volatility by calendar year", "* year to date")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_report(res: BacktestResult, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = summary(res)
    var = historical_var(res)
    table = annual_table(res, var)
    cur_pct, cur_eur = current_var(res)
    kup = kupiec_pof(int(var.invested.sum()), int(var.exceptions.sum()))

    plot_nav(res, out / "nav.png")
    plot_drawdown(res, out / "drawdown.png")
    plot_var(var, out / "var.png")
    plot_annual(table, out / "annual.png")
    table.to_csv(out / "annual_table.csv", float_format="%.6f")
    res.portfolio.to_csv(out / "portfolio.csv", float_format="%.6f")

    payload = {**{k: (v.isoformat() if hasattr(v, "isoformat") else float(v)) for k, v in s.items()},
               "current_var99_pct": cur_pct, "current_var99_eur": cur_eur, "kupiec": {k: float(v) for k, v in kup.items()}}
    (out / "summary.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# {res.config.name} — backtest report",
        "",
        f"Simulation {s['start']:%d/%m/%Y} → {s['end']:%d/%m/%Y}, initial NAV {res.config.initial_aum:,.0f} €, "
        f"global leverage x{res.config.global_leverage}, net of costs.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cumulative return | {s['cumulative_return']:+.1%} |",
        f"| Annualised return | {s['annualised_return']:.1%} |",
        f"| Annualised volatility | {s['annualised_vol']:.1%} |",
        f"| Sharpe ratio (over ECB deposit rate) | {s['sharpe']:.2f} |",
        f"| Maximum drawdown | {s['max_drawdown']:.1%} ({s['max_drawdown_date']:%d/%m/%Y}) |",
        f"| Average / max gross exposure | {s['avg_gross_exposure']:.1f}x / {s['max_gross_exposure']:.1f}x NAV |",
        f"| P&L Bund / Bobl | {s['pnl_bund']/1e6:.2f} M€ / {s['pnl_bobl']/1e6:.2f} M€ |",
        f"| 1-day VaR 99 % of the last position | {cur_pct:.1%} ({cur_eur/1e3:,.0f} k€) |",
        f"| VaR exceptions (Kupiec) | {kup['exceptions']:.0f} / {kup['n_obs']:.0f} = {kup['rate']:.2%} vs 1 % — p-value {kup['p_value']:.3f} |",
        "",
        "## Calendar years",
        "",
        "| Year | Return | Vol | Max DD | VaR 99 % avg | VaR 99 % max | Days UCITS VaR > 20 % | Avg exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for y, r in table.iterrows():
        lines.append(
            f"| {y} | {r['return']:+.1%} | {r['vol']:.1%} | {r['max_drawdown']:.1%} | {r['var99_avg']:.1%} | "
            f"{r['var99_max']:.1%} | {r['ucits_breach_days']:.0f} | {r['avg_exposure']:.1f}x |"
        )
    lines += ["", "![NAV](nav.png)", "![Drawdown](drawdown.png)", "![VaR](var.png)", "![Annual](annual.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
