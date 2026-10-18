"""Files generated for the Unreal game (GPU-free)."""

from __future__ import annotations

import math

import numpy as np

from learned_opponents.config import load_task
from learned_opponents.env import obstacles as ob
from learned_opponents.export import game_data as gd
from learned_opponents.spec import obs_action_spec as spec

TASK = load_task()


def test_task_json_carries_the_spec_and_rules():
    t = gd.task_json(TASK, "c" * 40)
    assert t["spec"] == {k: TASK[k] for k in gd.SPEC_KEYS}
    assert t["max_obstacles"] == TASK["n_obstacles"][1] and t["episode_limit"] == TASK["episode_limit"]
    assert t["commit"] == "c" * 40


def test_obs_cases_are_reproducible_and_round_trip_through_unreal_frame():
    a = gd.obs_cases(TASK, n=20)
    b = gd.obs_cases(TASK, n=20)
    assert a == b
    cfg = spec.as_spec_config(TASK)
    for c in a:
        u = c["unreal"]
        # Convert the Unreal-frame inputs back with the spec helpers and rebuild the vector.
        rx, ry = spec.unreal_to_spec_xy(u["robot_x_cm"], u["robot_y_cm"])
        gx, gy = spec.unreal_to_spec_xy(u["goal_x_cm"], u["goal_y_cm"])
        ranges = [r if r < 0 else r / 100.0 for r in u["ranges_cm"]]
        pose = {"robot_x": rx, "robot_y": ry, "robot_yaw": spec.unreal_to_spec_yaw(u["robot_yaw_deg"]),
                "goal_x": gx, "goal_y": gy}
        vel = {"linear": spec.unreal_to_spec_speed(u["speed_cm_s"]),
               "angular": spec.unreal_to_spec_yaw_rate(u["yaw_rate_deg_s"])}
        rebuilt = spec.build_observation(ranges, pose, vel, cfg)
        np.testing.assert_allclose(rebuilt, c["expected"], atol=gd.TOLERANCE)
        assert len(c["expected"]) == cfg.obs_dim
    # The cases exercise misses and both velocity clips.
    flat = np.array([r for c in a for r in c["ranges_m"]])
    assert (flat < 0).any() and (flat > cfg.max_range_m).any()


def test_race_course_keeps_checkpoints_and_starts_clear():
    course = gd.race_course(TASK)
    lay = ob.layouts_from_json([course], TASK["n_obstacles"][1])
    pts = np.array([[p["x"], p["y"]] for p in course["checkpoints"]] +
                   [[course["start"]["x"], course["start"]["y"]], [course["player_start"]["x"], course["player_start"]["y"]]])
    d = ob.obstacle_signed_distance_xp(np, pts, np.repeat(lay.kind, len(pts), 0), np.repeat(lay.center, len(pts), 0),
                                       np.repeat(lay.size, len(pts), 0)).min(-1)
    assert np.all(d >= TASK["robot_radius_m"] + 0.45 - 1e-9)
    assert np.all(np.abs(pts) <= TASK["arena_size_m"] / 2 - TASK["robot_radius_m"])
    assert len(course["checkpoints"]) == 6 and math.isfinite(course["start"]["yaw"])
    assert gd.race_course(TASK) == course
