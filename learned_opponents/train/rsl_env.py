"""rsl-rl ``VecEnv`` adapter for ``NavEnv`` plus per-iteration episode statistics."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from learned_opponents.env import rules
from learned_opponents.env.nav_env import NavEnv

OBS_GROUP = "policy"


class EpisodeStats:
    """Counts of finished episodes by reason, kept on the GPU until read."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.counts = torch.zeros(4, dtype=torch.long, device=self.device)
        self.length_sum = torch.zeros((), dtype=torch.long, device=self.device)
        self.success_length_sum = torch.zeros((), dtype=torch.long, device=self.device)
        self.reward_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.steps = 0

    def add(self, reward: torch.Tensor, done: torch.Tensor, reason: torch.Tensor, ep_steps: torch.Tensor) -> None:
        self.counts += torch.bincount(reason[done], minlength=4)[:4]
        self.length_sum += ep_steps.sum()
        self.success_length_sum += ep_steps[reason == rules.SUCCESS].sum()
        self.reward_sum += reward.double().sum()
        self.steps += reward.numel()

    def read(self) -> dict[str, float]:
        c = self.counts.tolist()
        n = c[rules.SUCCESS] + c[rules.COLLISION] + c[rules.TIMEOUT]
        out = {
            "episodes": n,
            "success_frac": c[rules.SUCCESS] / n if n else float("nan"),
            "collision_frac": c[rules.COLLISION] / n if n else float("nan"),
            "timeout_frac": c[rules.TIMEOUT] / n if n else float("nan"),
            "mean_episode_steps": self.length_sum.item() / n if n else float("nan"),
            "mean_success_steps": (self.success_length_sum.item() / c[rules.SUCCESS]
                                   if c[rules.SUCCESS] else float("nan")),
            "mean_reward_per_step": self.reward_sum.item() / self.steps if self.steps else float("nan"),
        }
        self.reset()
        return out


class RslNavEnv(VecEnv):
    def __init__(self, env: NavEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = 2
        self.max_episode_length = env.episode_limit
        self.device = torch.device(env.device)
        self.cfg = {"task": env.task, "variant": env.variant}
        self.stats = EpisodeStats(self.device)
        env.reset()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.step_count

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.step_count = value

    def _td(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({OBS_GROUP: obs}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._td(self.env.obs)

    def step(self, actions: torch.Tensor):
        obs, reward, done, info = self.env.step(actions)
        self.stats.add(reward, done, info.reason, info.episode_steps)
        extras = {"time_outs": info.time_outs, "log": {}}
        return self._td(obs), reward, done.long(), extras
