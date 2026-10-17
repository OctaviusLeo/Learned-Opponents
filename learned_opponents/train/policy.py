"""Build and load the rsl-rl actor; the deterministic action used everywhere.

The deterministic action is the Gaussian mean of the rsl-rl ``MLPModel``
(learned observation normalization inside the model), clipped to [-1, 1] as
the environment clips it. Evaluation, parity fixtures, and the ONNX export all
go through ``deterministic_action`` so they agree with training.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

OBS_GROUP = "policy"


def actor_kwargs(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    p = train_cfg["policy"]
    return {"hidden_dims": list(p["hidden_sizes"]), "activation": p["activation"],
            "obs_normalization": bool(p["obs_normalization"]),
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": float(p["init_std"]),
                                 "std_type": "scalar"}}


def critic_kwargs(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    p = train_cfg["policy"]
    return {"hidden_dims": list(p["hidden_sizes"]), "activation": p["activation"],
            "obs_normalization": bool(p["obs_normalization"])}


def build_actor(obs_dim: int, train_cfg: Mapping[str, Any], device: str = "cpu") -> MLPModel:
    dummy = TensorDict({OBS_GROUP: torch.zeros(1, obs_dim)}, batch_size=[1])
    kw = copy.deepcopy(actor_kwargs(train_cfg))
    return MLPModel(dummy, {"actor": [OBS_GROUP]}, "actor", 2, **kw).to(device)


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_actor(run_dir: str | Path, device: str = "cpu") -> tuple[MLPModel, dict[str, Any]]:
    """Actor from ``<run_dir>/checkpoint.pt`` with the run's resolved config, in eval mode."""
    cfg = load_run_config(run_dir)
    obs_dim = int(cfg["task"]["n_rays"]) + 5
    actor = build_actor(obs_dim, cfg["train"], device)
    state = torch.load(Path(run_dir) / "checkpoint.pt", map_location=device, weights_only=False)
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()
    return actor, cfg


@torch.no_grad()
def deterministic_action(actor: MLPModel, obs: torch.Tensor) -> torch.Tensor:
    td = TensorDict({OBS_GROUP: obs}, batch_size=[obs.shape[0]])
    return torch.clip(actor(td), -1.0, 1.0)
