"""Collect the reported training runs' budgets and wall times into results/eval/training_runs.json.

    python -m learned_opponents.eval.training_runs

Reads ``runs/<variant>/seed<seed>/summary.json`` (written by ``python -m learned_opponents.train``;
the run directories are not committed) for the seeds in train.yaml, so the "Training runs"
section of results/summary.md can show each run's environment steps and wall time. Standard
library only; no GPU.
"""

from __future__ import annotations

import json
import sys
from typing import Sequence

from learned_opponents import config as cfglib
from learned_opponents.eval.robustness import EVAL_DIR, check_runs, provenance, run_dir, write_json

FIELDS = ("commit", "iterations", "env_steps", "wall_s", "stopped_early")


def collect(allow_dirty: bool = False) -> dict:
    train = cfglib.load_yaml("train.yaml")
    runs = []
    for variant in ("baseline", "randomized"):
        seeds = list(train["seeds"][variant])
        check_runs(variant, seeds)
        for seed in seeds:
            s = json.loads((run_dir(variant, seed) / "summary.json").read_text(encoding="utf-8"))
            runs.append({"variant": variant, "seed": seed, **{k: s[k] for k in FIELDS}})
    return {"kind": "training_runs", "command": "python -m learned_opponents.eval.training_runs",
            "commit": provenance(allow_dirty), "runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = EVAL_DIR / "training_runs.json"
    write_json(path, collect(allow_dirty="--allow-dirty" in argv))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
