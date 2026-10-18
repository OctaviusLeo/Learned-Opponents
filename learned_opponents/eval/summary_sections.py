"""Robustness-grid and transfer sections of results/summary.md (numpy/stdlib only).

Each entry of ``SECTIONS`` takes the results/eval directory and returns
Markdown, or an empty string when its raw result file does not exist yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from learned_opponents.eval import metrics as M

LABELS = {"mass_scale": "Mass scale", "friction_scale": "Wheel friction scale",
          "latency_steps": "Latency (action + observation, control steps)"}


def _load(eval_dir: Path, name: str) -> dict[str, Any] | None:
    p = eval_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cell_stats(cell: Mapping[str, Any], dt: float) -> dict[str, Any]:
    """Pooled (across seeds) metrics for one grid cell, plus each seed's own success rate."""
    ds = list(cell["seeds"].values())
    steps = [s for d in ds for s in d["success_steps"]]
    return {"success": M.pooled([d["success"] for d in ds], [d["n"] for d in ds]),
            "collision": M.pooled([d["collision"] for d in ds], [d["n"] for d in ds]),
            "ttg": M.mean_t_ci([s * dt for s in steps]),
            "per_seed": {s: d["success"] / d["n"] for s, d in cell["seeds"].items()}}


def grid_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    dt = float(data["control_dt_s"])
    return [{"variant": c["variant"], "parameter": c["parameter"], "value": c["value"],
             "outside": c["outside"], **cell_stats(c, dt)} for c in data["cells"]]


def _fmt_value(param: str, v: float) -> str:
    return f"{int(v)}" if param == "latency_steps" else f"{v:g}"


def grid_section(eval_dir: Path) -> str:
    data = _load(eval_dir, "grid.json")
    if not data:
        return ""
    rows = grid_rows(data)
    seeds = sorted({s for c in data["cells"] for s in c["seeds"]})
    lines = ["## Robustness grid", "",
             f"- Command: `{data['command']}`", f"- Commit: `{data['commit']}`",
             f"- Evaluation seed {data['eval_seed']}; {data['episodes_per_cell']} episodes per training seed per "
             f"cell (the same layouts in every cell); training seeds {', '.join(seeds)} per variant; "
             "deterministic actions; one parameter varied at a time, the rest nominal; a latency value sets "
             "both the action and the observation delay. Sweep wall time "
             f"{data['wall_s']:.0f} s.",
             "- Aggregation: episodes are pooled across the training seeds (Wilson interval on the pooled count; "
             "t interval on the pooled successful times); the last column lists each seed's own Success_Rate.",
             "- `out` marks values outside the randomized variant's training range "
             + "; ".join(f"{p} {lo:g}-{hi:g}" for p, (lo, hi) in data["training_ranges"].items()) + ".",
             ""]
    for param in data["grid"]:
        lines += [f"### {LABELS.get(param, param)}", "",
                  "| Variant | Value | Range | Success_Rate [95% CI] | Collision_Rate [95% CI] | "
                  "Mean Time_To_Goal [95% CI] | Per-seed Success_Rate |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            if r["parameter"] != param:
                continue
            per = ", ".join(f"{s}: {v:.3f}" for s, v in sorted(r["per_seed"].items()))
            lines.append(f"| {r['variant']} | {_fmt_value(param, r['value'])} | "
                         f"{'out' if r['outside'] else 'in'} | {M.fmt_rate(r['success'])} | "
                         f"{M.fmt_rate(r['collision'])} | {M.fmt_mean(r['ttg'])} | {per} |")
        lines += ["", f"Plot: `results/robustness_{param}.png`", ""]
    return "\n".join(lines)


def glance_section(eval_dir: Path) -> str:
    """Baseline vs randomized pooled Success_Rate per grid cell, side by side."""
    data = _load(eval_dir, "grid.json")
    if not data:
        return ""
    rows = {(r["variant"], r["parameter"], r["value"]): r for r in grid_rows(data)}
    lines = ["## Robustness at a glance", "",
             f"- Command: `{data['command']}`; commit `{data['commit']}`; evaluation seed {data['eval_seed']}; "
             f"{data['episodes_per_cell']} episodes per training seed per cell, pooled over 3 seeds per variant "
             "(full table with collision rates, times, and per-seed rates under \"Robustness grid\").", "",
             "| Parameter | Value | Range | Baseline Success_Rate [95% CI] | Randomized Success_Rate [95% CI] | "
             "Randomized - baseline |", "| --- | --- | --- | --- | --- | --- |"]
    for param, values in data["grid"].items():
        for v in values:
            b, r = rows.get(("baseline", param, v)), rows.get(("randomized", param, v))
            if not (b and r):
                continue
            lines.append(f"| {LABELS.get(param, param).split(' (')[0]} | {_fmt_value(param, v)} | "
                         f"{'out' if b['outside'] else 'in'} | {M.fmt_rate(b['success'])} | "
                         f"{M.fmt_rate(r['success'])} | {r['success'].rate - b['success'].rate:+.3f} |")
    return "\n".join(lines) + "\n"


def transfer_section(eval_dir: Path) -> str:
    data = _load(eval_dir, "transfer.json")
    if not data:
        return ""
    from learned_opponents.eval.transfer_stats import transfer_rows

    lines = ["## Transfer check (MuJoCo vs Unreal, same layouts)", "",
             f"- Command: `{data['command']}`", f"- Commit: `{data['commit']}`",
             f"- {data['n_layouts']} layouts from `results/transfer/episodes.json` (layout seed "
             f"{data['layout_seed']}) x {len(data['seeds'])} training seeds per variant, nominal parameters, "
             "deterministic actions; the game runs headless and steps each episode at control_dt_s.",
             "- Transfer_Gap = MuJoCo Success_Rate - in-game Success_Rate on the same episodes, reported as "
             "measured (nothing tuned to shrink it).", "",
             "| Variant | Episodes | MuJoCo Success_Rate [95% CI] | In-game Success_Rate [95% CI] | "
             "Transfer_Gap | In-game Collision_Rate [95% CI] |", "| --- | --- | --- | --- | --- | --- |"]
    for r in transfer_rows(data):
        lines.append(f"| {r['variant']} | {r['n']} | {M.fmt_rate(r['mujoco'])} | {M.fmt_rate(r['game'])} | "
                     f"{r['gap']:+.3f} | {M.fmt_rate(r['game_collision'])} |")
    return "\n".join(lines) + "\n"


SECTIONS = [glance_section, grid_section, transfer_section]
