"""Tests for the engine-independent Obs_Action_Spec (tasks 4.1, 4.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.config import load_task
from learned_opponents.spec import obs_action_spec as spec

DEFAULT = spec.SpecConfig.from_mapping(load_task())

configs = st.builds(
    spec.SpecConfig,
    n_rays=st.integers(2, 64),
    ray_fov_deg=st.one_of(st.just(360.0), st.floats(10.0, 359.0)),
    max_range_m=st.floats(0.5, 20.0),
    arena_size_m=st.floats(1.0, 50.0),
    wheel_radius_m=st.floats(0.01, 0.5),
    track_width_m=st.floats(0.05, 2.0),
    max_wheel_speed=st.floats(0.5, 100.0),
)
any_float = st.floats(allow_nan=True, allow_infinity=True, width=64)
coord = st.floats(-30.0, 30.0)


def _expected_ray(r: float, max_range: float) -> float:
    if not math.isfinite(r) or r < 0 or r > max_range:
        r = max_range
    return min(max(r / max_range, 0.0), 1.0)


# Feature: learned-opponents, Property 1: Observation layout conforms to the Obs_Action_Spec
# (dimension n_rays + 5, every field in its declared range, fixed slice ordering).
@settings(max_examples=200, deadline=None)
@given(cfg=configs, data=st.data(), rx=coord, ry=coord, gx=coord, gy=coord,
       yaw=st.floats(-10.0, 10.0), v=st.floats(-1e3, 1e3), w=st.floats(-1e3, 1e3))
def test_property_1_observation_layout(cfg, data, rx, ry, gx, gy, yaw, v, w):
    rays = data.draw(st.lists(any_float, min_size=cfg.n_rays, max_size=cfg.n_rays))
    pose = {"robot_x": rx, "robot_y": ry, "robot_yaw": yaw, "goal_x": gx, "goal_y": gy}
    obs = spec.build_observation(rays, pose, {"linear": v, "angular": w}, cfg)

    assert obs.shape == (cfg.n_rays + 5,) and obs.dtype == np.float32
    low, high = spec.observation_bounds(cfg)
    assert np.all(obs >= low) and np.all(obs <= high)

    # Ray block first, in index order, with the miss mapping.
    expected = np.array([_expected_ray(r, cfg.max_range_m) for r in rays], dtype=np.float32)
    np.testing.assert_allclose(obs[: cfg.n_rays], expected, atol=1e-6)

    # Then goal_dir_x, goal_dir_y, goal_distance, vel_linear, vel_angular.
    n = cfg.n_rays
    dx, dy = gx - rx, gy - ry
    lx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
    dist = math.hypot(lx, ly)
    if dist > 1e-6:
        np.testing.assert_allclose(obs[n:n + 2], [lx / dist, ly / dist], atol=1e-5)
        assert abs(float(np.hypot(obs[n], obs[n + 1])) - 1.0) < 1e-5
    np.testing.assert_allclose(obs[n + 2], min(dist / cfg.arena_diagonal_m, 1.0), atol=1e-5)
    np.testing.assert_allclose(obs[n + 3], np.clip(v / cfg.v_max, -1, 1), atol=1e-6)
    np.testing.assert_allclose(obs[n + 4], np.clip(w / cfg.omega_max, -1, 1), atol=1e-6)


# Feature: learned-opponents, Property 2: Action layout conforms to the Obs_Action_Spec
# (dimension 2, both wheel-velocity components in [-1, 1]).
@settings(max_examples=200, deadline=None)
@given(cfg=configs, a=st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=2))
def test_property_2_action_layout(cfg, a):
    act = spec.build_action(a)
    assert act.shape == (spec.ACTION_DIM,) and act.dtype == np.float32
    assert np.all(act >= -1.0) and np.all(act <= 1.0)
    np.testing.assert_allclose(act, np.clip(np.float32(a), -1, 1))
    wheel = spec.wheel_speeds_rad_s(np, act, cfg)
    assert np.all(np.abs(wheel) <= cfg.max_wheel_speed * (1 + 1e-6))
    v, om = spec.body_velocity(np, act, cfg)
    assert abs(v) <= cfg.v_max * (1 + 1e-6) and abs(om) <= cfg.omega_max * (1 + 1e-6)


def test_default_ray_layout_matches_design():
    deg = np.degrees(spec.ray_angles_rad(DEFAULT))
    assert DEFAULT.n_rays == 24 and len(deg) == 24
    assert deg[0] == pytest.approx(-180.0)            # straight back
    assert deg[6] == pytest.approx(-90.0)             # right
    assert deg[12] == pytest.approx(0.0)              # ahead
    assert deg[18] == pytest.approx(90.0)             # left
    assert np.allclose(np.diff(deg), 15.0)            # counter-clockwise, uniform
    dirs = spec.ray_directions(DEFAULT)
    np.testing.assert_allclose(dirs[12], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dirs[18], [0.0, 1.0], atol=1e-12)


def test_narrow_fan_spans_edges():
    cfg = spec.SpecConfig(5, 90.0, 4.0, 8.0, 0.05, 0.3, 20.0)
    np.testing.assert_allclose(np.degrees(spec.ray_angles_rad(cfg)), [-45, -22.5, 0, 22.5, 45])


def test_normalizers_from_default_config():
    assert DEFAULT.v_max == pytest.approx(1.0)
    assert DEFAULT.omega_max == pytest.approx(2 * 0.05 * 20 / 0.30)
    assert DEFAULT.arena_diagonal_m == pytest.approx(math.sqrt(2) * 8.0)
    assert DEFAULT.obs_dim == 29


def test_miss_and_self_hit_map_to_max_range():
    rays = np.full(DEFAULT.n_rays, 2.0)
    rays[0], rays[1], rays[2] = -1.0, 7.5, np.inf
    hits = np.zeros(DEFAULT.n_rays, dtype=bool)
    hits[3] = True
    pose = {"robot_x": 0, "robot_y": 0, "robot_yaw": 0, "goal_x": 1, "goal_y": 0}
    obs = spec.build_observation(rays, pose, {"linear": 0, "angular": 0}, DEFAULT, self_hit=hits)
    np.testing.assert_allclose(obs[:5], [1, 1, 1, 1, 0.5])


def test_goal_in_robot_frame_examples():
    pose = {"robot_x": 1.0, "robot_y": 1.0, "robot_yaw": math.pi / 2, "goal_x": 1.0, "goal_y": 3.0}
    obs = spec.build_observation(np.full(24, 4.0), pose, {"linear": 0.5, "angular": -1.0}, DEFAULT)
    np.testing.assert_allclose(obs[24:27], [1.0, 0.0, 2.0 / DEFAULT.arena_diagonal_m], atol=1e-6)
    pose["goal_x"], pose["goal_y"] = 0.0, 1.0      # behind-left turns into robot-frame +y
    obs = spec.build_observation(np.full(24, 4.0), pose, {"linear": 0, "angular": 0}, DEFAULT)
    np.testing.assert_allclose(obs[24:26], [0.0, 1.0], atol=1e-6)
    pose["goal_x"], pose["goal_y"] = 1.0, 1.0      # on the robot: zero direction
    obs = spec.build_observation(np.full(24, 4.0), pose, {"linear": 0, "angular": 0}, DEFAULT)
    np.testing.assert_allclose(obs[24:27], [0.0, 0.0, 0.0])


def test_vectorized_builder_matches_reference():
    rng = np.random.default_rng(3)
    n = 16
    ranges = rng.uniform(-1, 6, (n, 24))
    rxy, gxy = rng.uniform(-4, 4, (n, 2)), rng.uniform(-4, 4, (n, 2))
    yaw, v, w = rng.uniform(-4, 4, n), rng.uniform(-2, 2, n), rng.uniform(-9, 9, n)
    batch = spec.build_observation_xp(np, ranges, rxy, yaw, gxy, v, w, DEFAULT)
    for i in range(n):
        pose = {"robot_x": rxy[i, 0], "robot_y": rxy[i, 1], "robot_yaw": yaw[i],
                "goal_x": gxy[i, 0], "goal_y": gxy[i, 1]}
        one = spec.build_observation(ranges[i], pose, {"linear": v[i], "angular": w[i]}, DEFAULT)
        np.testing.assert_allclose(batch[i], one, atol=1e-6)


def test_unreal_conversion():
    assert spec.unreal_to_spec_xy(250.0, -100.0) == pytest.approx((2.5, 1.0))
    assert spec.unreal_to_spec_yaw(90.0) == pytest.approx(-math.pi / 2)   # Unreal +90 = facing right
    assert spec.unreal_to_spec_yaw_rate(-180.0) == pytest.approx(math.pi)
    assert spec.unreal_to_spec_speed(50.0) == pytest.approx(0.5)
    assert spec.spec_ray_angle_to_unreal_yaw_offset_deg(math.pi / 2) == pytest.approx(-90.0)


def test_build_observation_rejects_wrong_ray_count():
    pose = {"robot_x": 0, "robot_y": 0, "robot_yaw": 0, "goal_x": 1, "goal_y": 0}
    with pytest.raises(ValueError):
        spec.build_observation(np.zeros(23), pose, {"linear": 0, "angular": 0}, DEFAULT)


def test_build_action_rejects_bad_input():
    with pytest.raises(ValueError):
        spec.build_action([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        spec.build_action([float("nan"), 0.0])


def test_spec_config_validation():
    with pytest.raises(ValueError):
        spec.SpecConfig(1, 360.0, 4.0, 8.0, 0.05, 0.3, 20.0)
    with pytest.raises(ValueError):
        spec.SpecConfig(24, 400.0, 4.0, 8.0, 0.05, 0.3, 20.0)
    with pytest.raises(KeyError):
        spec.SpecConfig.from_mapping({"n_rays": 24})


def test_spec_module_imports_only_numpy_and_stdlib():
    import ast
    import sys

    src = open(spec.__file__, encoding="utf-8").read()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names) | {"numpy", "__future__"}
