"""Robustness-grid sweep on the GPU (task 11.1).

For each variant, training seed, and grid cell, the seed's deterministic
policy plays ``episodes_per_cell`` episodes on the shared evaluation layouts
with the cell's domain parameters. Raw counts go to
``results/eval/grid.json``; ``robustness.main`` then regenerates the plots and
``results/summary.md``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from typing import Sequence

from learned_opponents import config as cfglib
from learned_opponents.eval.grid_config import cells, outside_values, parse_grid, training_range
from learned_opponents.eval.robustness import EVAL_DIR, check_runs, provenance, run_dir, write_json


def evaluate_grid(variants: Sequence[str], allow_dirty: bool) -> None:
    import torch

    from learned_opponents.eval.rollout import eval_layouts, make_eval_env, run_episodes
    from learned_opponents.plotting.plots import save_robustness_plots
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.train.policy import load_actor

    commit = provenance(allow_dirty)
    task = cfglib.load_task()
    train = cfglib.load_yaml("train.yaml")
    ranges = parse_ranges(cfglib.load_yaml("domain_rand.yaml"))
    gcfg = cfglib.load_yaml("robustness_grid.yaml")
    grid = parse_grid(gcfg)
    n = int(gcfg["episodes_per_cell"])
    layouts = eval_layouts(task, task["eval_seed"], n)
    env = make_eval_env(task, n)
    all_cells = cells(grid, ranges)
    t0 = time.perf_counter()
    results = []
    for variant in variants:
        seeds = list(train["seeds"][variant])
        check_runs(variant, seeds)
        per = {c: {} for c in range(len(all_cells))}
        for seed in seeds:
            actor, rcfg = load_actor(run_dir(variant, seed), device="cuda:0")
            for ci, cell in enumerate(all_cells):
                eps, _ = run_episodes(env, actor, layouts, cell.params, noise_seed=task["eval_seed"])
                per[ci][str(seed)] = {**eps.as_dict(), "train_commit": rcfg["commit"]}
            print(f"{variant} seed {seed}: {len(all_cells)} cells done ({time.perf_counter() - t0:.0f} s)", flush=True)
            del actor
            torch.cuda.empty_cache()
        for ci, cell in enumerate(all_cells):
            results.append({"variant": variant, **asdict(cell), "seeds": per[ci]})
    wall = time.perf_counter() - t0
    data = {"kind": "grid", "command": "python -m learned_opponents.eval.robustness " + " ".join(sys.argv[1:]),
            "commit": commit, "eval_seed": task["eval_seed"], "episodes_per_cell": n,
            "control_dt_s": task["control_dt_s"], "grid": grid,
            "training_ranges": {p: list(training_range(ranges, p)) for p in grid},
            "outside_values": outside_values(grid, ranges), "cells": results, "wall_s": round(wall, 1)}
    write_json(EVAL_DIR / "grid.json", data)
    print(f"grid sweep wall time {wall:.1f} s")
    save_robustness_plots(data)
