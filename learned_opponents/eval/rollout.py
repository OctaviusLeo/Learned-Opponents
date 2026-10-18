"""Deterministic evaluation episodes on the GPU environment.

Each world plays exactly one episode on its own layout with fixed domain
parameters; the first termination (reason and step count) is recorded and
later steps of that world are ignored. Layouts come from the evaluation seed,
never a training seed, so a fixed checkpoint and evaluation seed reproduce
the same episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from learned_opponents.env import obstacles as ob
from learned_opponents.env import rules
from learned_opponents.env.nav_env import NavEnv
from learned_opponents.randomization import domain_rand as dr
from learned_opponents.train.policy import deterministic_action

EVAL_LAYOUT_STREAM = "eval_layouts"


@dataclass
class Episodes:
    reason: np.ndarray        # (n,) rules.SUCCESS / COLLISION / TIMEOUT
    steps: np.ndarray         # (n,) control steps until the episode ended
    control_dt_s: float

    @property
    def n(self) -> int:
        return int(self.reason.size)

    def count(self, reason: int) -> int:
        return int((self.reason == reason).sum())

    def time_to_goal_s(self) -> np.ndarray:
        return self.steps[self.reason == rules.SUCCESS] * self.control_dt_s

    def as_dict(self) -> dict:
        return {"n": self.n, "success": self.count(rules.SUCCESS), "collision": self.count(rules.COLLISION),
                "timeout": self.count(rules.TIMEOUT),
                "success_steps": self.steps[self.reason == rules.SUCCESS].astype(int).tolist()}


def nominal_params(ranges: Mapping[str, dr.ParamRange]) -> dict[str, float]:
    return {k: float(r.nominal) for k, r in ranges.items()}


def eval_layouts(task: Mapping, eval_seed: int, n: int) -> ob.Layouts:
    return ob.sample_layouts(dr.make_rng(int(eval_seed), EVAL_LAYOUT_STREAM), n, task)


def make_eval_env(task: Mapping, n: int, device: str = "cuda:0") -> NavEnv:
    return NavEnv(task, num_envs=n, seed=int(task["eval_seed"]), device=device, bank_size=n, auto_reset=False)


@torch.no_grad()
def run_episodes(env: NavEnv, actor, layouts: ob.Layouts, params: Mapping[str, float], noise_seed: int,
                 record_obs: bool = False) -> tuple[Episodes, np.ndarray | None]:
    """Play one episode per world. Optionally return every (pre-termination) observation."""
    if len(layouts) != env.num_envs or env.auto_reset:
        raise ValueError("needs a non-auto-reset env with one world per layout")
    env.use_layouts(layouts)
    env.set_param_override(params)
    env.noise_gen.manual_seed(int(noise_seed))
    obs = env.reset()
    n = env.num_envs
    reason = torch.zeros(n, dtype=torch.long, device=env.device)
    steps = torch.zeros(n, dtype=torch.long, device=env.device)
    seen = []
    for t in range(env.episode_limit):
        running = reason == rules.NONE
        if record_obs:
            seen.append(obs[running].cpu())
        act = deterministic_action(actor, obs)
        obs, _, done, info = env.step(act)
        first = running & done
        reason = torch.where(first, info.reason, reason)
        steps = torch.where(first, t + 1, steps)
        if t % 25 == 24 and not bool((reason == rules.NONE).any()):
            break
    assert not bool((reason == rules.NONE).any()), "every episode ends by the step limit"
    eps = Episodes(reason.cpu().numpy(), steps.cpu().numpy(), float(env.task["control_dt_s"]))
    return eps, (torch.cat(seen).numpy() if record_obs else None)
