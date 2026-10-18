"""Robustness-grid configuration: cells and the out-of-training-range check (numpy/stdlib only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from learned_opponents.randomization.domain_rand import PARAMS, ParamRange

# Grid parameter -> the domain parameters it sets.
GRID_PARAMS = {
    "mass_scale": ("mass_scale",),
    "friction_scale": ("friction_scale",),
    "latency_steps": ("action_latency_steps", "observation_latency_steps"),
}


@dataclass(frozen=True)
class Cell:
    parameter: str
    value: float
    params: dict[str, float]       # all six domain parameters for this cell
    outside: bool                  # value lies outside the training randomization range


def parse_grid(cfg: Mapping[str, Any], max_latency: int = 6) -> dict[str, list[float]]:
    missing = [p for p in GRID_PARAMS if p not in cfg]
    if missing:
        raise ValueError(f"robustness_grid: missing parameters {missing}")
    extra = set(cfg) - set(GRID_PARAMS) - {"episodes_per_cell"}
    if extra:
        raise ValueError(f"robustness_grid: unknown parameters {sorted(extra)}")
    grid = {}
    for p in GRID_PARAMS:
        values = [float(v) for v in cfg[p]]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"robustness_grid: '{p}' needs distinct values")
        if any(v < 0 for v in values) or (p.endswith("_scale") and any(v <= 0 for v in values)):
            raise ValueError(f"robustness_grid: '{p}' values out of range")
        if p == "latency_steps" and any(not v.is_integer() or v > max_latency for v in values):
            raise ValueError(f"robustness_grid: latency_steps must be whole steps <= {max_latency}")
        grid[p] = sorted(values)
    n = cfg.get("episodes_per_cell")
    if not (isinstance(n, int) and n > 0):
        raise ValueError("robustness_grid: 'episodes_per_cell' must be a positive integer")
    return grid


def training_range(ranges: Mapping[str, ParamRange], grid_param: str) -> tuple[float, float]:
    """The training randomization range a grid parameter perturbs (hull over the parameters it sets)."""
    rs = [ranges[p] for p in GRID_PARAMS[grid_param]]
    return min(r.low for r in rs), max(r.high for r in rs)


def outside_values(grid: Mapping[str, list[float]], ranges: Mapping[str, ParamRange]) -> dict[str, list[float]]:
    out = {}
    for p, values in grid.items():
        lo, hi = training_range(ranges, p)
        out[p] = [v for v in values if v < lo or v > hi]
    return out


def cells(grid: Mapping[str, list[float]], ranges: Mapping[str, ParamRange]) -> list[Cell]:
    nominal = {k: float(ranges[k].nominal) for k in PARAMS}
    out = []
    for p, values in grid.items():
        lo, hi = training_range(ranges, p)
        for v in values:
            params = dict(nominal)
            for dp in GRID_PARAMS[p]:
                params[dp] = float(v)
            out.append(Cell(p, float(v), params, v < lo or v > hi))
    return out
