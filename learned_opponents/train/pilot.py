"""Pilot evaluation: pick the training budget and measure training reproducibility (task 8.3).

    python -m learned_opponents.train.pilot curve   # after scripts/pilot_curve.sh trained runs/pilot/curve
    python -m learned_opponents.train.pilot repro   # after scripts/pilot_repro.sh trained repro_a and repro_b

The pilot is the baseline variant with seed 100 (``seeds.pilot``), never
reported as a result. It is evaluated on its own layouts (evaluation stream of
seed 100), not on the reported evaluation seed, so tuning never sees the
evaluation episodes.

Budget rule (design "Choosing the budget"): among the curve checkpoints that
the pilot reached within ``max_minutes_per_run``, take the best nominal
Success_Rate; the budget is the smallest checkpoint whose rate is within
``TOLERANCE`` of that best and at least ``nominal_success_target``. If none
reaches the target, the best checkpoint is used and the shortfall is reported.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from learned_opponents import config as cfglib
from learned_opponents.provenance import repo_commit

PILOT = cfglib.REPO_ROOT / "runs" / "pilot"
EVAL_DIR = cfglib.REPO_ROOT / "results" / "eval"
PILOT_EPISODES = 4096
TOLERANCE = 0.01


def choose_budget(points: Sequence[dict[str, Any]], target: float, max_minutes: float) -> dict[str, Any]:
    """Apply the budget rule to ``points`` (dicts with env_steps, elapsed_min, success_rate)."""
    in_time = [p for p in points if p["elapsed_min"] <= max_minutes]
    if not in_time:
        raise ValueError("no checkpoint within the time cap")
    best = max(p["success_rate"] for p in in_time)
    ok = [p for p in in_time if p["success_rate"] >= best - TOLERANCE and p["success_rate"] >= target]
    if ok:
        pick = min(ok, key=lambda p: p["env_steps"])
        return {"chosen_env_steps": pick["env_steps"], "chosen_success_rate": pick["success_rate"],
                "best_success_rate": best, "meets_target": True}
    pick = max(in_time, key=lambda p: (p["success_rate"], -p["env_steps"]))
    return {"chosen_env_steps": pick["env_steps"], "chosen_success_rate": pick["success_rate"],
            "best_success_rate": best, "meets_target": False}


def _evaluate(run_dir: Path, checkpoints: Sequence[Path], task, n: int) -> list[dict[str, Any]]:
    import torch

    from learned_opponents.eval.rollout import eval_layouts, make_eval_env, nominal_params, run_episodes
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.train.policy import load_actor

    seed = int(cfglib.load_yaml("train.yaml")["seeds"]["pilot"][0])
    layouts = eval_layouts(task, seed, n)
    env = make_eval_env(task, n)
    params = nominal_params(parse_ranges(cfglib.load_yaml("domain_rand.yaml")))
    out = []
    for ck in checkpoints:
        actor, _ = load_actor(run_dir, device="cuda:0", checkpoint=ck)
        eps, _ = run_episodes(env, actor, layouts, params, noise_seed=seed)
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        out.append({"checkpoint": str(ck.relative_to(cfglib.REPO_ROOT)), "iteration": int(meta["iter"]),
                    "success": eps.count(1), "collision": eps.count(2), "timeout": eps.count(3), "n": eps.n,
                    "success_rate": eps.count(1) / eps.n})
        print(f"{ck.name}: success {eps.count(1)}/{eps.n}", flush=True)
    return out


def _elapsed_by_iteration(run_dir: Path) -> dict[int, tuple[int, float]]:
    with open(run_dir / "train_log.csv") as f:
        return {int(r["iteration"]): (int(r["env_steps"]), float(r["elapsed_s"]) / 60) for r in csv.DictReader(f)}


def curve() -> int:
    task = cfglib.load_task()
    train = cfglib.load_yaml("train.yaml")
    run_dir = PILOT / "curve"
    cks = sorted((run_dir / "checkpoints").glob("iter*.pt")) + [run_dir / "checkpoint.pt"]
    points = _evaluate(run_dir, cks, task, PILOT_EPISODES)
    times = _elapsed_by_iteration(run_dir)
    for p in points:
        p["env_steps"], p["elapsed_min"] = times[p["iteration"]]
    choice = choose_budget(points, float(task["nominal_success_target"]), float(train["max_minutes_per_run"]))
    rcfg = cfglib.load_yaml(run_dir / "config.yaml")
    result = {"kind": "pilot_curve", "command": "python -m learned_opponents.train.pilot curve",
              "training_command": rcfg["command"], "commit": repo_commit(), "train_commit": rcfg["commit"],
              "seed": rcfg["seed"], "eval_layout_seed": rcfg["seed"], "episodes": PILOT_EPISODES,
              "rule": __doc__.split("Budget rule")[1].split("\n\n")[0].strip(), "tolerance": TOLERANCE,
              "nominal_success_target": task["nominal_success_target"],
              "max_minutes_per_run": train["max_minutes_per_run"], "points": points, **choice}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "pilot_curve.json", "w", newline="\n") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(choice))
    return 0


def repro() -> int:
    task = cfglib.load_task()
    runs = {}
    for name in ("repro_a", "repro_b"):
        d = PILOT / name
        (p,) = _evaluate(d, [d / "checkpoint.pt"], task, PILOT_EPISODES)
        s = json.loads((d / "summary.json").read_text())
        rcfg = cfglib.load_yaml(d / "config.yaml")
        runs[name] = {**p, "wall_min": s["wall_min"], "train_commit": rcfg["commit"],
                      "training_command": rcfg["command"]}
    a, b = runs["repro_a"]["success_rate"], runs["repro_b"]["success_rate"]
    result = {"kind": "pilot_reproducibility", "command": "python -m learned_opponents.train.pilot repro",
              "commit": repo_commit(), "seed": int(cfglib.load_yaml("train.yaml")["seeds"]["pilot"][0]),
              "eval_seed": "pilot evaluation stream of seed 100", "episodes": PILOT_EPISODES,
              "runs": runs, "tolerance": abs(a - b)}
    with open(EVAL_DIR / "pilot_reproducibility.json", "w", newline="\n") as f:
        json.dump(result, f, indent=1)
    print(f"tolerance {abs(a - b):.4f}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("curve", "repro"):
        raise SystemExit("usage: python -m learned_opponents.train.pilot {curve|repro}")
    raise SystemExit(curve() if cmd == "curve" else repro())
