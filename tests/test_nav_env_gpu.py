"""GPU tests of the MuJoCo Warp environment (tasks 5.2, 6.2). Run locally in WSL."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("mjlab")
import torch  # noqa: E402

from learned_opponents.env import obstacles as ob  # noqa: E402
from learned_opponents.env import rules  # noqa: E402
from learned_opponents.env.nav_env import NavEnv  # noqa: E402
from learned_opponents.randomization.domain_rand import make_rng  # noqa: E402
from learned_opponents.spec import obs_action_spec as spec  # noqa: E402

pytestmark = pytest.mark.gpu


# Feature: learned-opponents, Property 4: Reset returns a valid initial observation
# (conforms to the Obs_Action_Spec; exactly n_rays readings, each within [0, max_range_m]).
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_property_4_reset_returns_valid_observation(gpu_env, task, seed):
    lay = ob.sample_layouts(make_rng(seed, "layouts"), gpu_env.num_envs, task)
    gpu_env.use_layouts(lay)
    obs = gpu_env.reset().cpu().numpy()
    cfg = gpu_env.cfg
    assert obs.shape == (gpu_env.num_envs, task["n_rays"] + 5) and obs.dtype == np.float32
    low, high = spec.observation_bounds(cfg)
    assert np.all(obs >= low) and np.all(obs <= high)
    raw = gpu_env.raw_rays().cpu().numpy()
    assert raw.shape == (gpu_env.num_envs, task["n_rays"])
    mapped = np.where((raw < 0) | (raw > cfg.max_range_m), cfg.max_range_m, raw)
    assert np.all((mapped >= 0) & (mapped <= cfg.max_range_m))
    # The robot starts at rest at the layout's pose with its goal features matching the layout.
    xy, yaw, fwd, rate = (t.cpu().numpy() for t in gpu_env.robot_state())
    np.testing.assert_allclose(xy, lay.start[:, :2], atol=1e-5)
    np.testing.assert_allclose(np.cos(yaw - lay.start[:, 2]), 1.0, atol=1e-6)
    np.testing.assert_allclose(fwd, 0.0, atol=1e-6)
    dist = np.hypot(*(lay.goal - lay.start[:, :2]).T)
    np.testing.assert_allclose(obs[:, -3], dist / cfg.arena_diagonal_m, atol=1e-5)


def _single_episode(env, kind, center, size, start, goal):
    lay = ob.Layouts(np.asarray(kind, np.int8)[None], np.asarray(center, float)[None],
                     np.asarray(size, float)[None], np.asarray(start, float)[None], np.asarray(goal, float)[None])
    reps = env.num_envs
    env.use_layouts(lay.subset(np.zeros(reps, int)))
    env.reset()


def test_scripted_episodes_end_with_the_right_reason(gpu_env, task):
    k = gpu_env.k
    empty = (np.zeros(k), np.zeros((k, 2)), np.zeros(k))
    # Drive straight at a box 2 m ahead: collision when the footprint reaches it.
    kind = np.zeros(k); kind[0] = ob.BOX
    center = np.zeros((k, 2)); center[0] = (1.0, 0.0)
    size = np.zeros(k); size[0] = 0.6
    _single_episode(gpu_env, kind, center, size, (-1.0, 0.0, 0.0), (3.0, 0.0))
    reasons = _run(gpu_env, lambda: torch.full((gpu_env.num_envs, 2), 0.6, device="cuda"))
    assert set(reasons.values()) == {rules.COLLISION}
    # Spin in place in an empty arena: timeout exactly at the limit.
    _single_episode(gpu_env, *empty, (0.0, 0.0, 0.0), (3.0, 3.0))
    act = torch.tensor([[-0.5, 0.5]], device="cuda").repeat(gpu_env.num_envs, 1)
    steps = {}
    reasons = _run(gpu_env, lambda: act, steps)
    assert set(reasons.values()) == {rules.TIMEOUT} and set(steps.values()) == {task["episode_limit"]}
    # Drive straight at a goal 2 m ahead: success.
    _single_episode(gpu_env, *empty, (-1.0, 0.0, 0.0), (1.0, 0.0))
    reasons = _run(gpu_env, lambda: torch.full((gpu_env.num_envs, 2), 0.8, device="cuda"))
    assert set(reasons.values()) == {rules.SUCCESS}


def _run(env, policy, steps=None):
    first = {}
    for t in range(env.episode_limit):
        _, _, done, info = env.step(policy())
        for i in torch.nonzero(done).flatten().tolist():
            if i not in first:
                first[i] = int(info.reason[i])
                if steps is not None:
                    steps[i] = t + 1
        if len(first) == env.num_envs:
            break
    assert len(first) == env.num_envs
    return first


def test_forward_speed_and_turn_rate_follow_the_wheels(gpu_env):
    k = gpu_env.k
    _single_episode(gpu_env, np.zeros(k), np.zeros((k, 2)), np.zeros(k), (-3.0, 0.0, 0.0), (3.0, 3.0))
    for _ in range(20):
        obs, *_ = gpu_env.step(torch.full((gpu_env.num_envs, 2), 0.5, device="cuda"))
    vl = obs[:, -2].mean().item()
    assert vl == pytest.approx(0.5, abs=0.05)       # 0.5 * v_max, normalized
    for _ in range(20):
        obs, *_ = gpu_env.step(torch.tensor([[-0.3, 0.3]], device="cuda").repeat(gpu_env.num_envs, 1))
    va = obs[:, -1].mean().item()
    assert va == pytest.approx(0.3, abs=0.05)       # counter-clockwise is positive


def test_randomized_env_applies_domain_parameters(task, dr_ranges):
    env = NavEnv(task, num_envs=256, seed=5, variant="randomized", dr_ranges=dr_ranges, bank_size=1024)
    env.reset()
    mass = env.sim.model.body_mass[:, env.robot_bodies[0]].cpu().numpy()
    nominal = task["chassis_mass_kg"] + 2 * 0.01          # chassis plus the two casters
    assert env.nom_mass[0].item() == pytest.approx(nominal)
    np.testing.assert_allclose(mass, nominal * env.params["mass_scale"].cpu().numpy(), rtol=1e-6)
    fr = env.sim.model.geom_friction[:, env.wheel_geoms[0], 0].cpu().numpy()
    np.testing.assert_allclose(fr, task["wheel_friction"] * env.params["friction_scale"].cpu().numpy(), rtol=1e-6)
    gain = env.sim.model.actuator_gainprm[:, 0, 0].cpu().numpy()
    np.testing.assert_allclose(gain, task["motor_kv"] * env.params["motor_strength_scale"].cpu().numpy(), rtol=1e-6)
    assert mass.std() > 0 and fr.std() > 0
    lat = env.params["action_latency_steps"].cpu().numpy()
    assert set(np.unique(lat)) <= {0, 1, 2} and len(np.unique(lat)) > 1


def test_action_latency_delays_the_applied_command(task):
    env = NavEnv(task, num_envs=2, seed=0, bank_size=2, auto_reset=False)
    env.set_param_override({"mass_scale": 1, "friction_scale": 1, "motor_strength_scale": 1,
                            "sensor_noise_sigma_m": 0, "action_latency_steps": 2,
                            "observation_latency_steps": 0})
    env.reset()
    ctrl = []
    for t in range(4):
        env.step(torch.full((2, 2), 0.1 * (t + 1), device="cuda"))
        ctrl.append(env.sim.data.ctrl[0, 0].item() / task["max_wheel_speed"])
    np.testing.assert_allclose(ctrl, [0.0, 0.0, 0.1, 0.2], atol=1e-6)


def test_observation_latency_returns_older_observations(task):
    env = NavEnv(task, num_envs=2, seed=0, bank_size=2, auto_reset=False)
    env.set_param_override({"mass_scale": 1, "friction_scale": 1, "motor_strength_scale": 1,
                            "sensor_noise_sigma_m": 0, "action_latency_steps": 0,
                            "observation_latency_steps": 1})
    first = env.reset().clone()
    o1, *_ = env.step(torch.ones((2, 2), device="cuda"))
    torch.testing.assert_close(o1, first)                 # one step old = the reset observation
    o2, *_ = env.step(torch.ones((2, 2), device="cuda"))
    assert not torch.allclose(o2, first)
    assert math.isfinite(o2.sum().item())
