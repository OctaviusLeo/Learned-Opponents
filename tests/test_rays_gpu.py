"""Range_Rays wiring and GPU parallel stepping (task 6.2). Run locally in WSL."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mjlab")
import torch  # noqa: E402

from learned_opponents.config import load_yaml  # noqa: E402
from learned_opponents.env.nav_env import NavEnv, known_distance_layout  # noqa: E402

pytestmark = pytest.mark.gpu


def test_parallel_step_returns_range_rays_block(task):
    n = int(load_yaml("train.yaml")["num_parallel_envs"])
    assert n >= 4096                                   # the smoke-test parallel count
    env = NavEnv(task, num_envs=n, seed=1, bank_size=n)
    env.reset()
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(10):
        obs, rew, done, _ = env.step(torch.rand((n, 2), generator=gen, device="cuda") * 2 - 1)
        rays = env.range_rays()
        assert rays.is_cuda and tuple(rays.shape) == (n, task["n_rays"])
        assert bool(((rays >= 0) & (rays <= task["max_range_m"])).all())
        assert tuple(obs.shape) == (n, task["n_rays"] + 5) and tuple(rew.shape) == (n,)


@pytest.mark.parametrize("yaw,expected", [
    (0.0, {12: 1.7, 18: 1.3, 6: 0.85, 0: 4.0}),
    (math.pi / 2, {12: 1.3, 6: 1.7, 0: 0.85}),     # facing +y: cylinder ahead, big box right
])
def test_known_distances_on_expected_ray_index(task, yaw, expected):
    env = NavEnv(task, num_envs=1, seed=0, bank_size=1, auto_reset=False)
    env.use_layouts(known_distance_layout(env.k, yaw))
    env.reset()
    r = env.range_rays()[0].cpu().numpy()
    for i, d in expected.items():
        assert r[i] == pytest.approx(d, abs=1e-3), f"ray {i}"
    # The observation carries the same readings divided by max_range_m.
    np.testing.assert_allclose(env.obs[0, : task["n_rays"]].cpu().numpy(), r / task["max_range_m"], atol=1e-6)


def test_rays_never_hit_the_robot(task):
    """Empty arena, robot in the middle: every ray reaches a wall (> 3 m), none reads the chassis."""
    k = int(task["n_obstacles"][1])
    lay = known_distance_layout(k)
    lay.kind[:] = 0
    env = NavEnv(task, num_envs=1, seed=0, bank_size=1, auto_reset=False)
    env.use_layouts(lay)
    env.reset()
    raw = env.raw_rays()[0].cpu().numpy()
    assert np.all((raw < 0) | (raw > 3.9))
