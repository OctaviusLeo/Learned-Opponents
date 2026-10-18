"""Robustness plots: Success_Rate against each perturbed parameter (task 12).

One PNG per grid parameter, ``results/robustness_<parameter>.png``, with both
variants on the same axes, 95% Wilson intervals as error bars, the randomized
variant's training range shaded, and values outside it drawn as hollow
markers. Colors are the first two slots of the validated categorical palette;
marker shape and line style also separate the variants, so identity never
rests on color alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from learned_opponents.eval.summary_sections import LABELS, grid_rows  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
STYLE = {"baseline": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
         "randomized": {"color": "#eb6834", "marker": "s", "linestyle": "--"}}
INK, MUTED, GRID, SURFACE, BAND = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb", "#ecebe6"


def plot_parameter(data: Mapping[str, Any], param: str, out: Path) -> Path:
    rows = [r for r in grid_rows(data) if r["parameter"] == param]
    lo, hi = data["training_ranges"][param]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(lo, hi, color=BAND, zorder=0, label="training range (randomized)")
    ends = {v: max((r for r in rows if r["variant"] == v), key=lambda r: r["value"])["success"].rate
            for v in ("baseline", "randomized") if any(r["variant"] == v for r in rows)}
    close = len(ends) == 2 and abs(ends["baseline"] - ends["randomized"]) < 0.05
    nudge = {"baseline": -7 if close and ends["baseline"] <= ends["randomized"] else 7 if close else 0}
    nudge["randomized"] = -nudge["baseline"]
    for variant in ("baseline", "randomized"):
        vr = sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["value"])
        if not vr:
            continue
        st = STYLE[variant]
        x = [r["value"] for r in vr]
        y = [r["success"].rate for r in vr]
        err = [[r["success"].rate - r["success"].lo for r in vr], [r["success"].hi - r["success"].rate for r in vr]]
        ax.errorbar(x, y, yerr=err, color=st["color"], linestyle=st["linestyle"], linewidth=2, capsize=3,
                    elinewidth=1.2, zorder=3, label=variant)
        for r in vr:
            ax.plot(r["value"], r["success"].rate, marker=st["marker"], markersize=8, zorder=4,
                    color=st["color"], markerfacecolor=SURFACE if r["outside"] else st["color"],
                    markeredgecolor=st["color"], markeredgewidth=2)
        ax.annotate(variant, (x[-1], y[-1]), xytext=(7, nudge[variant]), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel(LABELS.get(param, param), color=INK)
    ax.set_ylabel("Success_Rate (pooled, 95% CI)", color=INK)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(f"Success_Rate vs {LABELS.get(param, param).split(' (')[0].lower()} "
                 "(hollow = outside training range)", fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.005, f"python -m learned_opponents.eval.robustness | commit {data['commit'][:12]} | "
             f"eval seed {data['eval_seed']}", fontsize=6, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def save_robustness_plots(data: Mapping[str, Any], results_dir: Path = RESULTS) -> list[Path]:
    return [plot_parameter(data, p, results_dir / f"robustness_{p}.png") for p in data["grid"]]
