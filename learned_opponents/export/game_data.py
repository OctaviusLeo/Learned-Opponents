"""Files the Unreal game reads, generated from the Python sources of truth (tasks 15.2, 16.2).

    python -m learned_opponents.export.game_data

- ``results/game/task.json``: the task.yaml values the game needs (spec, robot, timing, rules).
- ``results/fixtures/obs_cases.json``: Property 3 cases. Each holds raw spec-frame inputs, the same
  inputs converted to Unreal's frame (cm, y mirrored, clockwise yaw in degrees), and the vector
  ``build_observation`` returns; the C++ builder must match within ``TOLERANCE``.
- ``results/game/race_course.json``: the race level's obstacles, checkpoints, and start poses.

Imports only numpy and the standard library (plus this package), so CI can run it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from learned_opponents import config as cfglib
from learned_opponents.env import obstacles as ob
from learned_opponents.provenance import repo_commit
from learned_opponents.randomization.domain_rand import make_rng
from learned_opponents.spec import obs_action_spec as spec

RESULTS = cfglib.REPO_ROOT / "results"
FIXTURE_SEED = 20261018
RACE_SEED = 2026
N_CASES = 128
TOLERANCE = 1e-5
SPEC_KEYS = ("n_rays", "ray_fov_deg", "max_range_m", "arena_size_m", "wheel_radius_m", "track_width_m",
             "max_wheel_speed")


def spec_dict(task: Mapping[str, Any]) -> dict[str, Any]:
    return {k: task[k] for k in SPEC_KEYS}


def task_json(task: Mapping[str, Any], commit: str) -> dict[str, Any]:
    keys = ("robot_radius_m", "ray_height_m", "control_dt_s", "physics_dt_s", "arrival_tolerance_m",
            "wall_height_m", "obstacle_height_m", "episode_limit")
    return {"spec": spec_dict(task), **{k: task[k] for k in keys}, "max_obstacles": int(task["n_obstacles"][1]),
            "commit": commit, "generated_by": "python -m learned_opponents.export.game_data"}


def obs_cases(task: Mapping[str, Any], n: int = N_CASES, seed: int = FIXTURE_SEED) -> list[dict[str, Any]]:
    cfg = spec.as_spec_config(task)
    rng = make_rng(seed, "obs_fixtures")
    half = cfg.arena_size_m / 2.0
    cases = []
    for _ in range(n):
        ranges = rng.uniform(0.05, cfg.max_range_m * 1.3, cfg.n_rays)
        ranges[rng.random(cfg.n_rays) < 0.2] = -1.0          # misses
        rx, ry, gx, gy = rng.uniform(-half, half, 4)
        yaw = float(rng.uniform(-math.pi, math.pi))
        v = float(rng.uniform(-1.3, 1.3) * cfg.v_max)
        w = float(rng.uniform(-1.3, 1.3) * cfg.omega_max)
        pose = {"robot_x": rx, "robot_y": ry, "robot_yaw": yaw, "goal_x": gx, "goal_y": gy}
        expected = spec.build_observation(ranges, pose, {"linear": v, "angular": w}, cfg)
        cases.append({
            "ranges_m": ranges.tolist(), "robot_x": rx, "robot_y": ry, "robot_yaw": yaw, "goal_x": gx, "goal_y": gy,
            "vel_linear": v, "vel_angular": w,
            "unreal": {"ranges_cm": [r if r < 0 else 100.0 * r for r in ranges.tolist()],
                       "robot_x_cm": 100.0 * rx, "robot_y_cm": -100.0 * ry, "robot_yaw_deg": -math.degrees(yaw),
                       "goal_x_cm": 100.0 * gx, "goal_y_cm": -100.0 * gy, "speed_cm_s": 100.0 * v,
                       "yaw_rate_deg_s": -math.degrees(w)},
            "expected": [float(x) for x in expected],
        })
    return cases


def race_course(task: Mapping[str, Any], seed: int = RACE_SEED, n_checkpoints: int = 6, n_obstacles: int = 7) -> dict:
    """Checkpoints on a ring, obstacles between them, starts behind checkpoint 0."""
    rng = make_rng(seed, "race_course")
    ring = 2.7
    angles = [2 * math.pi * i / n_checkpoints - math.pi / 2 for i in range(n_checkpoints)]
    checkpoints = np.array([[ring * math.cos(a), ring * math.sin(a)] for a in angles])
    a0 = angles[0] - 0.5
    heading = math.atan2(checkpoints[0, 1] - ring * math.sin(a0), checkpoints[0, 0] - ring * math.cos(a0))
    tangent = np.array([-math.sin(a0), math.cos(a0)])
    base = np.array([ring * math.cos(a0), ring * math.sin(a0)])
    starts = [base + 0.35 * tangent, base - 0.35 * tangent]
    keep_clear = np.vstack([checkpoints, np.array(starts)])
    clear = float(task["robot_radius_m"]) + 0.45
    half = float(task["arena_size_m"]) / 2
    lo_s, hi_s = (float(v) for v in task["obstacle_size_m"])
    kinds = [ob.KIND_CODES[k] for k in task["obstacle_kinds"]]
    obstacles = []
    while len(obstacles) < n_obstacles:
        s = float(rng.uniform(lo_s, hi_s))
        c = rng.uniform(-half + s / 2, half - s / 2, 2)
        k = int(kinds[rng.integers(0, len(kinds))])
        d = ob.obstacle_signed_distance_xp(np, keep_clear, np.full((len(keep_clear), 1), k),
                                           np.tile(c, (len(keep_clear), 1, 1)), np.full((len(keep_clear), 1), s))
        if float(d.min()) >= clear:
            obstacles.append({"kind": ob.KIND_NAMES[k], "x": float(c[0]), "y": float(c[1]), "size": s})
    return {"seed": seed, "obstacles": obstacles,
            "checkpoints": [{"x": float(x), "y": float(y)} for x, y in checkpoints],
            "start": {"x": float(starts[0][0]), "y": float(starts[0][1]), "yaw": heading},
            "player_start": {"x": float(starts[1][0]), "y": float(starts[1][1]), "yaw": heading},
            "goal": {"x": float(checkpoints[0, 0]), "y": float(checkpoints[0, 1])}}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")


def main() -> int:
    task = cfglib.load_task()
    commit = repo_commit()
    write_json(RESULTS / "game" / "task.json", task_json(task, commit))
    write_json(RESULTS / "fixtures" / "obs_cases.json",
               {"generated_by": "python -m learned_opponents.export.game_data", "commit": commit,
                "seed": FIXTURE_SEED, "tolerance": TOLERANCE, "spec": spec_dict(task), "cases": obs_cases(task)})
    write_json(RESULTS / "game" / "race_course.json", race_course(task))
    print("wrote results/game/task.json, results/fixtures/obs_cases.json, results/game/race_course.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
