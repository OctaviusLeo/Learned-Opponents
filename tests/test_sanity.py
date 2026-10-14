"""Trivial sanity test so the suite is non-empty while property tests are
deferred. Confirms the spec imports and that a sample observation has the
expected length ``n_rays + 5``.
"""

import numpy as np

from learned_opponents.spec import obs_action_spec as spec


def test_spec_imports():
    assert hasattr(spec, "build_observation")
    assert hasattr(spec, "build_action")


def test_observation_length_is_n_rays_plus_5():
    n_rays = 16
    config = spec.SpecConfig(
        n_rays=n_rays,
        max_range_m=5.0,
        max_wheel_speed=10.0,
        arena_diagonal_m=float(np.hypot(10.0, 10.0)),
    )
    raycasts = np.full(n_rays, 2.5, dtype=np.float32)
    goal_pose = {
        "goal_x": 1.0,
        "goal_y": 2.0,
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_theta": 0.0,
    }
    robot_vel = {"linear": 1.0, "angular": 0.0}

    obs = spec.build_observation(raycasts, goal_pose, robot_vel, config)

    assert obs.shape == (n_rays + 5,)
    assert obs.dtype == np.float32
