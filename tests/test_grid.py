"""Robustness grid config (Property 11), summary rendering, and plots on synthetic results. GPU-free."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.config import load_yaml
from learned_opponents.eval import grid_config as G
from learned_opponents.eval import summary, summary_sections
from learned_opponents.randomization.domain_rand import parse_ranges

RANGES = parse_ranges(load_yaml("domain_rand.yaml"))


@st.composite
def range_and_grid(draw):
    """A training-range config and a grid that the validator accepts."""
    cfg = {}
    for name in ("mass_scale", "friction_scale", "motor_strength_scale"):
        lo = draw(st.floats(0.1, 2.0))
        hi = draw(st.floats(lo, lo + 2.0))
        cfg[name] = {"nominal": lo, "range": [lo, hi]}
    cfg["sensor_noise_sigma_m"] = {"nominal": 0.0, "range": [0.0, 0.05]}
    for name in ("action_latency_steps", "observation_latency_steps"):
        hi = draw(st.integers(0, 4))
        cfg[name] = {"nominal": 0, "range": [0, hi]}
    ranges = parse_ranges(cfg)
    grid = {"episodes_per_cell": 8}
    for p in G.GRID_PARAMS:
        lo, hi = G.training_range(ranges, p)
        inside = draw(st.lists(st.floats(lo, hi) if p != "latency_steps" else st.integers(int(lo), int(hi)),
                               min_size=0, max_size=4))
        out_v = (draw(st.floats(hi + 0.01, hi + 5.0)) if p != "latency_steps"
                 else draw(st.integers(int(hi) + 1, 6)))
        grid[p] = sorted({float(v) for v in inside} | {float(out_v)})
    return ranges, grid


# Feature: learned-opponents, Property 11: Grid includes out-of-training-range values
# (for every perturbed parameter, at least one grid value lies outside its training range, and the
# cells built from the grid flag exactly those values).
@settings(max_examples=200, deadline=None)
@given(rg=range_and_grid())
def test_property_11_grid_has_out_of_range_values(rg):
    ranges, cfg = rg
    grid = G.parse_grid(cfg)
    outside = G.outside_values(grid, ranges)
    for p in G.GRID_PARAMS:
        assert outside[p], p
    cells = G.cells(grid, ranges)
    assert len(cells) == sum(len(v) for v in grid.values())
    for c in cells:
        assert c.outside == (c.value in outside[c.parameter])
        for dp in G.GRID_PARAMS[c.parameter]:
            assert c.params[dp] == c.value
        others = set(c.params) - set(G.GRID_PARAMS[c.parameter])
        assert all(c.params[k] == ranges[k].nominal for k in others)


def test_committed_grid_matches_design_and_has_out_of_range_values():
    grid = G.parse_grid(load_yaml("robustness_grid.yaml"))
    out = G.outside_values(grid, RANGES)
    assert out == {"mass_scale": [0.5, 1.5, 2.0], "friction_scale": [0.3, 1.5], "latency_steps": [3.0, 4.0, 6.0]}
    assert len(G.cells(grid, RANGES)) == 17


@pytest.mark.parametrize("bad,match", [
    ({"latency_steps": [0, 1.5]}, "whole steps"),
    ({"latency_steps": [0, 9]}, "whole steps"),
    ({"mass_scale": [1.0, 1.0]}, "distinct"),
    ({"episodes_per_cell": 0}, "episodes_per_cell"),
])
def test_invalid_grids_are_rejected(bad, match):
    cfg = {**load_yaml("robustness_grid.yaml"), **bad}
    with pytest.raises(ValueError, match=match):
        G.parse_grid(cfg)


def synthetic_grid() -> dict:
    """Known counts: baseline succeeds 900/1000 at nominal and loses 100 per out-of-range step."""
    grid = G.parse_grid(load_yaml("robustness_grid.yaml"))
    cells = []
    for variant, drop in (("baseline", 100), ("randomized", 40)):
        for c in G.cells(grid, RANGES):
            k = 900 - (drop if c.outside else 0)
            seeds = {str(s): {"n": 1000, "success": k, "collision": 1000 - k, "timeout": 0,
                              "success_steps": [100] * k, "train_commit": "t" * 40} for s in (0, 1, 2)}
            cells.append({"variant": variant, "parameter": c.parameter, "value": c.value, "outside": c.outside,
                          "params": c.params, "seeds": seeds})
    return {"kind": "grid", "command": "python -m learned_opponents.eval.robustness", "commit": "c" * 40,
            "eval_seed": 1, "episodes_per_cell": 1000, "control_dt_s": 0.05, "grid": grid,
            "training_ranges": {p: list(G.training_range(RANGES, p)) for p in grid},
            "outside_values": G.outside_values(grid, RANGES), "cells": cells, "wall_s": 1.0}


def test_grid_rows_pool_known_counts():
    rows = summary_sections.grid_rows(synthetic_grid())
    r = next(r for r in rows if r["variant"] == "baseline" and r["parameter"] == "mass_scale" and r["value"] == 2.0)
    assert r["success"].k == 2400 and r["success"].n == 3000 and r["outside"]
    assert r["ttg"].mean == pytest.approx(5.0)
    assert r["per_seed"] == {"0": 0.8, "1": 0.8, "2": 0.8}


def test_summary_and_plots_from_synthetic_results(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    data = synthetic_grid()
    (eval_dir / "grid.json").write_text(json.dumps(data))
    nominal = {"kind": "nominal", "variant": "baseline", "command": "python -m x --nominal", "commit": "c" * 40,
               "eval_seed": 1, "episodes_per_seed": 10, "nominal_success_target": 0.8, "control_dt_s": 0.05,
               "seeds": {str(s): {"n": 10, "success": 8, "collision": 1, "timeout": 1, "success_steps": [100] * 8,
                                  "train_commit": "t" * 40} for s in (0, 1, 2)}}
    (eval_dir / "nominal_baseline.json").write_text(json.dumps(nominal))
    out = summary.write_summary(tmp_path / "summary.md", eval_dir)
    text = out.read_text()
    assert "## Nominal evaluation: baseline" in text and "**met**" in text
    assert "0.800 [0.627, 0.905]" in text          # pooled 24/30 Wilson interval
    assert "## Robustness grid" in text and "c" * 40 in text
    assert text.count("| baseline |") == 17 and text.count("| randomized |") == 17   # one row per grid cell
    from learned_opponents.plotting.plots import save_robustness_plots

    paths = save_robustness_plots(data, tmp_path)
    assert [p.name for p in paths] == [f"robustness_{p}.png" for p in data["grid"]]
    for p in paths:
        assert p.stat().st_size > 10_000 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.isfinite(paths[0].stat().st_size)


def test_glance_section_compares_variants(tmp_path):
    (tmp_path / "grid.json").write_text(json.dumps(synthetic_grid()))
    text = summary_sections.glance_section(tmp_path)
    assert text.startswith("## Robustness at a glance")
    # Out-of-range cells: baseline 800/1000, randomized 860/1000 per seed -> +0.060.
    assert "| Mass scale | 2 | out | 0.800" in text and "+0.060 |" in text
    assert text.count("\n| ") == 2 + 17          # header, rule, one row per cell
