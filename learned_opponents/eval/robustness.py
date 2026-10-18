"""Nominal evaluation and the robustness-grid sweep (tasks 9.1, 11.1, 12).

    python -m learned_opponents.eval.robustness --nominal [--variants baseline] [--repeat-check]
    python -m learned_opponents.eval.robustness            # full grid, both variants, + plots

Every evaluation uses deterministic actions, the layouts drawn from
``eval_seed`` (task.yaml), and one episode per layout. Raw counts go to
``results/eval/*.json`` with command, seeds, and commit; ``results/summary.md``
and the plots are regenerated from them (never hand-edited).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from learned_opponents import config as cfglib
from learned_opponents.provenance import dirty_paths, repo_commit

RESULTS = cfglib.REPO_ROOT / "results"
EVAL_DIR = RESULTS / "eval"
RUNS = cfglib.REPO_ROOT / "runs"


def run_dir(variant: str, seed: int) -> Path:
    return RUNS / variant / f"seed{seed}"


def check_runs(variant: str, seeds: Sequence[int]) -> None:
    missing = [s for s in seeds if not (run_dir(variant, s) / "summary.json").is_file()]
    if missing:
        raise SystemExit(f"missing complete runs for {variant} seeds {missing} (runs/{variant}/seed<seed>/)")


def provenance(allow_dirty: bool) -> str:
    dirty = dirty_paths()
    if dirty and not allow_dirty:
        raise SystemExit("refusing to report from a dirty tree; commit first:\n  " + "\n  ".join(dirty))
    return repo_commit()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")


def evaluate_nominal(variants: Sequence[str], repeats: int, allow_dirty: bool) -> list[Path]:
    """Evaluate each seed at nominal parameters; ``repeats`` extra identical evaluations
    measure the evaluation reproducibility tolerance (MuJoCo Warp's float atomics make
    repeated GPU rollouts differ in the last bits, which can flip borderline episodes)."""
    import torch

    from learned_opponents.eval.rollout import eval_layouts, make_eval_env, nominal_params, run_episodes
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.train.policy import load_actor

    commit = provenance(allow_dirty)
    task = cfglib.load_task()
    train = cfglib.load_yaml("train.yaml")
    ranges = parse_ranges(cfglib.load_yaml("domain_rand.yaml"))
    n = int(task["eval_episodes_per_seed"])
    params = nominal_params(ranges)
    layouts = eval_layouts(task, task["eval_seed"], n)
    env = make_eval_env(task, n)
    command = " ".join(["python -m learned_opponents.eval.robustness", *sys.argv[1:]])
    written = []
    for variant in variants:
        seeds = list(train["seeds"][variant])
        check_runs(variant, seeds)
        t0 = time.perf_counter()
        per_seed, repro = {}, {}
        for seed in seeds:
            actor, rcfg = load_actor(run_dir(variant, seed), device="cuda:0")
            eps, _ = run_episodes(env, actor, layouts, params, noise_seed=task["eval_seed"])
            per_seed[str(seed)] = {**eps.as_dict(), "train_commit": rcfg["commit"]}
            if repeats:
                succ = [eps.count(1)]
                identical = 0
                for _ in range(repeats):
                    again, _ = run_episodes(env, actor, layouts, params, noise_seed=task["eval_seed"])
                    succ.append(again.count(1))
                    identical += int((again.reason == eps.reason).all() and (again.steps == eps.steps).all())
                repro[str(seed)] = {"success_counts": succ, "identical_repeats": identical,
                                    "max_abs_rate_diff": (max(succ) - min(succ)) / n}
            del actor
            torch.cuda.empty_cache()
        out = EVAL_DIR / f"nominal_{variant}.json"
        write_json(out, {"kind": "nominal", "variant": variant, "command": command, "commit": commit,
                         "eval_seed": task["eval_seed"], "episodes_per_seed": n, "seeds": per_seed,
                         "params": params, "nominal_success_target": task["nominal_success_target"],
                         "control_dt_s": task["control_dt_s"],
                         "repeats": repro or None, "wall_s": round(time.perf_counter() - t0, 1)})
        written.append(out)
        print(f"wrote {out.relative_to(cfglib.REPO_ROOT)}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learned_opponents.eval.robustness")
    ap.add_argument("--nominal", action="store_true", help="nominal-parameter evaluation only")
    ap.add_argument("--variants", nargs="+", choices=["baseline", "randomized"])
    ap.add_argument("--repeats", type=int, default=0,
                    help="extra identical evaluations per checkpoint to measure evaluation reproducibility")
    ap.add_argument("--allow-dirty", action="store_true", help="development only; never for reported numbers")
    args = ap.parse_args(argv)
    from learned_opponents.eval import summary

    if args.nominal:
        evaluate_nominal(args.variants or ["baseline"], args.repeats, args.allow_dirty)
    else:
        from learned_opponents.eval.grid import evaluate_grid

        evaluate_grid(args.variants or ["baseline", "randomized"], args.allow_dirty)
    summary.write_summary()
    print(f"wrote {summary.SUMMARY.relative_to(cfglib.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
