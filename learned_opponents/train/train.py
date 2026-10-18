"""PPO training entry point (task 8.1).

    python -m learned_opponents.train --variant {baseline|randomized} --seed <int>

Trains the navigation policy with rsl-rl PPO on the MuJoCo Warp environment
for ``training_budget_steps`` environment steps (rounded up to whole PPO
iterations) and writes the Run_Artifacts to ``runs/<variant>/seed<seed>/``:
``config.yaml`` (fully resolved task, train, and domain-randomization config,
variant, seed, commit), ``seed``, ``commit``, ``checkpoint.pt``,
``train_log.csv`` (one row per iteration), and ``summary.json``.

``--set section.key=value`` overrides a config value (pilot tuning only; the
override is recorded in config.yaml). The config helpers import no GPU
framework, so CI can test them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from learned_opponents import config as cfglib
from learned_opponents.provenance import repo_commit

VARIANTS = ("baseline", "randomized")
SECTIONS = ("task", "train", "domain_rand")
FRAMEWORK_PACKAGES = ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "tensordict")


def parse_overrides(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``["train.ppo.learning_rate=3e-4"]`` -> ``{"train": {"ppo": {"learning_rate": 3e-4}}}``."""
    out: dict[str, dict[str, Any]] = {s: {} for s in SECTIONS}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        if parts[0] not in SECTIONS or len(parts) < 2:
            raise ValueError(f"override key must start with one of {SECTIONS}: {key!r}")
        node = out[parts[0]]
        for p in parts[1:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _scalar(raw)
    return out


def _scalar(raw: str) -> Any:
    """YAML value, but numbers like ``3e-4`` (a string in YAML 1.1) become floats."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def resolve(variant: str, seed: int, overrides: Sequence[str] = (), commit: str | None = None) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    ov = parse_overrides(overrides)
    task = cfglib.validate_task(cfglib.deep_merge(cfglib.load_yaml("task.yaml"), ov["task"]))
    train = cfglib.deep_merge(cfglib.load_yaml("train.yaml"), ov["train"])
    dr = cfglib.deep_merge(cfglib.load_yaml("domain_rand.yaml"), ov["domain_rand"])
    validate_train(train)
    return {"variant": variant, "seed": int(seed), "commit": commit or repo_commit(),
            "overrides": list(overrides), "task": task, "train": train, "domain_rand": dr}


def validate_train(train: Mapping[str, Any]) -> None:
    for key in ("training_budget_steps", "num_parallel_envs", "max_minutes_per_run", "bank_size"):
        v = train.get(key)
        if not (isinstance(v, int) and v > 0):
            raise cfglib.ConfigError(f"train.yaml: '{key}' must be a positive integer, got {v!r}")
    if train["num_parallel_envs"] < 4096:
        raise cfglib.ConfigError("train.yaml: 'num_parallel_envs' must be >= 4096 (the smoke-test count)")
    ppo = train["ppo"]
    for key in ("num_steps_per_env", "num_learning_epochs", "num_mini_batches"):
        if not (isinstance(ppo.get(key), int) and ppo[key] > 0):
            raise cfglib.ConfigError(f"train.yaml: 'ppo.{key}' must be a positive integer")
    for key in ("learning_rate", "clip_param", "gamma", "lam", "max_grad_norm", "value_loss_coef"):
        if not float(ppo.get(key, -1)) > 0:
            raise cfglib.ConfigError(f"train.yaml: 'ppo.{key}' must be > 0")
    if ppo["schedule"] not in ("adaptive", "fixed"):
        raise cfglib.ConfigError("train.yaml: 'ppo.schedule' must be adaptive or fixed")
    seeds = train["seeds"]
    for v in VARIANTS:
        if len(seeds[v]) < 3:
            raise cfglib.ConfigError(f"train.yaml: 'seeds.{v}' needs at least 3 seeds")


def iterations_for(train: Mapping[str, Any]) -> int:
    per_iter = int(train["num_parallel_envs"]) * int(train["ppo"]["num_steps_per_env"])
    return max(1, math.ceil(int(train["training_budget_steps"]) / per_iter))


def rsl_cfg(train: Mapping[str, Any]) -> dict[str, Any]:
    """The configuration dict rsl-rl's ``PPO.construct_algorithm`` consumes."""
    from learned_opponents.train.policy import OBS_GROUP, actor_kwargs, critic_kwargs

    p = train["ppo"]
    alg = {k: p[k] for k in ("num_learning_epochs", "num_mini_batches", "learning_rate", "schedule",
                             "desired_kl", "clip_param", "entropy_coef", "value_loss_coef",
                             "max_grad_norm", "gamma", "lam", "use_clipped_value_loss")}
    return {"num_steps_per_env": int(p["num_steps_per_env"]),
            "obs_groups": {"actor": [OBS_GROUP], "critic": [OBS_GROUP]},
            "algorithm": {"class_name": "PPO", **alg, "rnd_cfg": None, "symmetry_cfg": None},
            "actor": {"class_name": "MLPModel", **actor_kwargs(train)},
            "critic": {"class_name": "MLPModel", **critic_kwargs(train)},
            "multi_gpu": None}


def framework_versions() -> dict[str, str]:
    out = {}
    for pkg in FRAMEWORK_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


LOG_FIELDS = ["iteration", "env_steps", "elapsed_s", "env_steps_per_s", "episodes", "success_frac",
              "collision_frac", "timeout_frac", "mean_episode_steps", "mean_success_steps",
              "mean_reward_per_step", "action_std", "learning_rate", "value_loss", "surrogate_loss",
              "entropy"]


def run(resolved: dict[str, Any], out_dir: Path, max_minutes: float | None = None,
        checkpoint_every_steps: int | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from rsl_rl.algorithms import PPO

    from learned_opponents.env.nav_env import NavEnv
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.train.rsl_env import RslNavEnv

    seed, train, task = resolved["seed"], resolved["train"], resolved["task"]
    device = train.get("device", "cuda:0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    iters = iterations_for(train)
    resolved = {**resolved, "iterations": iters,
                "steps_per_iteration": int(train["num_parallel_envs"]) * int(train["ppo"]["num_steps_per_env"]),
                "framework": "mjlab / MuJoCo Warp, rsl-rl PPO", "framework_versions": framework_versions(),
                "jax_default_matmul_precision": "not applicable (MJX not chosen in M0)",
                "command": "python -m learned_opponents.train " + " ".join(sys.argv[1:])}
    resolved["planned_env_steps"] = iters * resolved["steps_per_iteration"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    (out_dir / "seed").write_text(f"{seed}\n")
    (out_dir / "commit").write_text(f"{resolved['commit']}\n")

    env = NavEnv(task, int(train["num_parallel_envs"]), seed, resolved["variant"],
                 parse_ranges(resolved["domain_rand"]), device=device, bank_size=int(train["bank_size"]))
    venv = RslNavEnv(env)
    alg = PPO.construct_algorithm(venv.get_observations(), venv, rsl_cfg(train), device)
    alg.train_mode()
    obs = venv.get_observations()
    steps_per_it = resolved["steps_per_iteration"]
    t_start = time.perf_counter()
    stopped_early = False
    with open(out_dir / "train_log.csv", "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for it in range(iters):
            with torch.inference_mode():
                for _ in range(int(train["ppo"]["num_steps_per_env"])):
                    actions = alg.act(obs)
                    obs, rew, dones, extras = venv.step(actions)
                    alg.process_env_step(obs, rew, dones, extras)
                alg.compute_returns(obs)
            losses = alg.update()
            elapsed = time.perf_counter() - t_start
            row = {"iteration": it + 1, "env_steps": (it + 1) * steps_per_it, "elapsed_s": round(elapsed, 2),
                   "env_steps_per_s": round((it + 1) * steps_per_it / elapsed),
                   **venv.stats.read(), "action_std": float(alg.get_policy().output_std.mean()),
                   "learning_rate": alg.learning_rate,
                   "value_loss": losses.get("value", losses.get("value_function", float("nan"))),
                   "surrogate_loss": losses.get("surrogate", float("nan")),
                   "entropy": losses.get("entropy", float("nan"))}
            writer.writerow({k: row.get(k) for k in LOG_FIELDS})
            if checkpoint_every_steps and (it + 1) * steps_per_it // checkpoint_every_steps > it * steps_per_it // checkpoint_every_steps:
                (out_dir / "checkpoints").mkdir(exist_ok=True)
                torch.save({**alg.save(), "iter": it + 1, "env_steps": (it + 1) * steps_per_it},
                           out_dir / "checkpoints" / f"iter{it + 1:05d}.pt")
            if (it + 1) % 10 == 0 or it == 0 or it + 1 == iters:
                logf.flush()
                print(f"it {it + 1}/{iters} steps {row['env_steps']:,} {row['env_steps_per_s']:,}/s "
                      f"success {row['success_frac']:.3f} collision {row['collision_frac']:.3f} "
                      f"timeout {row['timeout_frac']:.3f} std {row['action_std']:.3f}", flush=True)
            if max_minutes is not None and elapsed > 60 * max_minutes:
                stopped_early = True
                print(f"stopping after {it + 1} iterations: --max-minutes {max_minutes} reached", flush=True)
                break
    done_its = it + 1
    torch.save({**alg.save(), "iter": done_its}, out_dir / "checkpoint.pt")
    wall = time.perf_counter() - t_start
    summary = {"variant": resolved["variant"], "seed": seed, "commit": resolved["commit"],
               "iterations": done_its, "env_steps": done_its * steps_per_it, "wall_s": round(wall, 1),
               "wall_min": round(wall / 60, 2), "stopped_early": stopped_early,
               "within_max_minutes": wall <= 60 * float(train["max_minutes_per_run"])}
    with open(out_dir / "summary.json", "w", newline="\n") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learned_opponents.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", type=Path, help="default runs/<variant>/seed<seed>")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--max-minutes", type=float, help="stop early after this wall time (pilot exploration)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    ap.add_argument("--checkpoint-every-steps", type=int, help="also save checkpoints/iterNNNNN.pt (pilot curve)")
    args = ap.parse_args(argv)
    resolved = resolve(args.variant, args.seed, args.overrides)
    out = args.out or (cfglib.REPO_ROOT / "runs" / args.variant / f"seed{args.seed}")
    if (out / "checkpoint.pt").exists() and not args.force:
        print(f"refusing: {out}/checkpoint.pt exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    os.environ.setdefault("LO_COMMIT", resolved["commit"])
    run(resolved, out, args.max_minutes, args.checkpoint_every_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
