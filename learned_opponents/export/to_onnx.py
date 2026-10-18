"""Export trained policies to ONNX and save parity fixtures (task 13.1).

    python -m learned_opponents.export.to_onnx --checkpoint runs/baseline/seed0 --out results/models/x.onnx
    python -m learned_opponents.export.to_onnx --all      # six models + fixtures under results/

The graph is the whole deterministic policy, so the game feeds it the raw
Obs_Action_Spec vector and uses its output directly: the learned observation
normalization ``(obs - mean) / (std + eps)``, the MLP, and the clip to
[-1, 1] the environment applies. I/O: input ``obs`` float32[1, n_rays + 5],
output ``action`` float32[1, 2], opset 17 (IR version 8).

Parity fixtures per model (``results/parity/<variant>_seed<seed>/``): 1,000
observations, half sampled from nominal evaluation rollouts and half uniform
within the declared observation ranges, with the actions the Python policy
produces for them (``observations.npy``, ``actions.npy``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from learned_opponents import config as cfglib
from learned_opponents.provenance import repo_commit

MODELS = cfglib.REPO_ROOT / "results" / "models"
PARITY = cfglib.REPO_ROOT / "results" / "parity"
OPSET = 17
N_ROLLOUT, N_UNIFORM = 500, 500


def model_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def export(run_dir: Path, out: Path, commit: str) -> dict:
    import onnx
    import torch
    from torch import nn

    from learned_opponents.train.policy import load_actor

    actor, cfg = load_actor(run_dir, device="cpu")
    obs_dim = int(cfg["task"]["n_rays"]) + 5

    class DeterministicPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalizer = actor.obs_normalizer
            self.mlp = actor.mlp

        def forward(self, obs):
            return torch.clamp(self.mlp(self.normalizer(obs)), -1.0, 1.0)

    module = DeterministicPolicy().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, (torch.zeros(1, obs_dim),), str(out), input_names=["obs"], output_names=["action"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    model = onnx.load(str(out))
    meta = {"variant": cfg["variant"], "seed": str(cfg["seed"]), "train_commit": cfg["commit"],
            "export_commit": commit, "obs_layout": "rays[n_rays]/max_range, goal_dir_x, goal_dir_y, "
            "goal_distance, vel_linear, vel_angular (learned_opponents/spec/spec.md)",
            "action_layout": "wheel_left, wheel_right in [-1, 1] (x max_wheel_speed rad/s)"}
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    path = out.resolve()
    shown = path.relative_to(cfglib.REPO_ROOT) if path.is_relative_to(cfglib.REPO_ROOT) else path
    return {"path": str(shown), "obs_dim": obs_dim, "opset": OPSET,
            "ir_version": model.ir_version, **meta}


def make_fixtures(variant: str, seed: int, run_dir: Path, out_dir: Path) -> dict:
    """1,000 parity observations and the Python policy's actions for them."""
    import torch

    from learned_opponents.eval.rollout import eval_layouts, make_eval_env, nominal_params, run_episodes
    from learned_opponents.randomization.domain_rand import make_rng, parse_ranges
    from learned_opponents.spec.obs_action_spec import observation_bounds
    from learned_opponents.train.policy import deterministic_action, load_actor

    task = cfglib.load_task()
    rng = make_rng(seed, f"parity_{variant}")
    gpu_actor, _ = load_actor(run_dir, device="cuda:0")
    n_eps = 256
    env = make_eval_env(task, n_eps)
    params = nominal_params(parse_ranges(cfglib.load_yaml("domain_rand.yaml")))
    _, seen = run_episodes(env, gpu_actor, eval_layouts(task, task["eval_seed"], n_eps), params,
                           noise_seed=task["eval_seed"], record_obs=True)
    rollout = seen[rng.choice(len(seen), N_ROLLOUT, replace=False)]
    low, high = observation_bounds(task)
    uniform = rng.uniform(low, high, size=(N_UNIFORM, low.size))
    obs = np.concatenate([rollout, uniform]).astype(np.float32)
    cpu_actor, _ = load_actor(run_dir, device="cpu")
    actions = deterministic_action(cpu_actor, torch.from_numpy(obs)).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "observations.npy", obs)
    np.save(out_dir / "actions.npy", actions)
    return {"observations": int(obs.shape[0]), "from_rollouts": N_ROLLOUT, "uniform": N_UNIFORM}


def export_all(allow_dirty: bool) -> int:
    from learned_opponents.eval.robustness import check_runs, provenance, run_dir

    commit = provenance(allow_dirty)
    train = cfglib.load_yaml("train.yaml")
    manifest = {"command": "python -m learned_opponents.export.to_onnx " + " ".join(sys.argv[1:]),
                "commit": commit, "models": {}}
    for variant in ("baseline", "randomized"):
        seeds = train["seeds"][variant]
        check_runs(variant, seeds)
        for seed in seeds:
            name = model_name(variant, seed)
            info = export(run_dir(variant, seed), MODELS / f"{name}.onnx", commit)
            info["fixtures"] = make_fixtures(variant, seed, run_dir(variant, seed), PARITY / name)
            manifest["models"][name] = info
            print(f"exported {info['path']} and fixtures for {name}", flush=True)
    with open(MODELS / "manifest.json", "w", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learned_opponents.export.to_onnx")
    ap.add_argument("--checkpoint", type=Path, help="run directory holding checkpoint.pt and config.yaml")
    ap.add_argument("--out", type=Path, help="output .onnx path")
    ap.add_argument("--all", action="store_true", help="export all six reported runs with parity fixtures")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    if args.all:
        return export_all(args.allow_dirty)
    if not (args.checkpoint and args.out):
        ap.error("give --checkpoint and --out, or --all")
    print(json.dumps(export(args.checkpoint, args.out, repo_commit()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
