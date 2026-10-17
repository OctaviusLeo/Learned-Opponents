"""Load and validate the YAML configs under ``config/``.

Invalid configuration is rejected before any run starts, with an error that
names the missing or out-of-range parameter (design "Error Handling").
Imports only the standard library, numpy, and PyYAML, so CI can use it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(ValueError):
    """A configuration value is missing or out of range."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a mapping at the top level")
    return data


def _require(cfg: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in cfg:
        raise ConfigError(f"{where}: missing parameter '{key}'")
    return cfg[key]


def _positive(cfg: Mapping[str, Any], key: str, where: str) -> float:
    v = float(_require(cfg, key, where))
    if not (math.isfinite(v) and v > 0):
        raise ConfigError(f"{where}: '{key}' must be a finite number > 0, got {v}")
    return v


def _range(cfg: Mapping[str, Any], key: str, where: str, lo_min: float = -math.inf,
           integer: bool = False) -> tuple[float, float]:
    v = _require(cfg, key, where)
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ConfigError(f"{where}: '{key}' must be a [min, max] pair, got {v!r}")
    lo, hi = (int(v[0]), int(v[1])) if integer else (float(v[0]), float(v[1]))
    if integer and (lo != v[0] or hi != v[1]):
        raise ConfigError(f"{where}: '{key}' must hold integers, got {v!r}")
    if not (lo <= hi and lo >= lo_min):
        raise ConfigError(f"{where}: '{key}' must satisfy {lo_min} <= min <= max, got {v!r}")
    return lo, hi


TASK_POSITIVE = ["arena_size_m", "wall_height_m", "obstacle_height_m", "robot_radius_m",
                 "wheel_radius_m", "track_width_m", "max_wheel_speed", "chassis_mass_kg",
                 "wheel_mass_kg", "wheel_friction", "motor_kv", "motor_torque_limit_nm",
                 "ray_fov_deg", "ray_height_m", "max_range_m", "control_dt_s", "physics_dt_s",
                 "arrival_tolerance_m", "min_start_goal_distance_m"]


def validate_task(cfg: Mapping[str, Any], where: str = "task.yaml") -> dict[str, Any]:
    for key in TASK_POSITIVE:
        _positive(cfg, key, where)
    if float(cfg["start_goal_clearance_m"]) < 0:
        raise ConfigError(f"{where}: 'start_goal_clearance_m' must be >= 0")
    n_rays = _require(cfg, "n_rays", where)
    if not (isinstance(n_rays, int) and n_rays >= 2):
        raise ConfigError(f"{where}: 'n_rays' must be an integer >= 2, got {n_rays!r}")
    if float(cfg["ray_fov_deg"]) > 360:
        raise ConfigError(f"{where}: 'ray_fov_deg' must be <= 360")
    _range(cfg, "n_obstacles", where, lo_min=0, integer=True)
    _range(cfg, "obstacle_size_m", where, lo_min=1e-6)
    kinds = _require(cfg, "obstacle_kinds", where)
    if not kinds or any(k not in ("box", "cylinder") for k in kinds):
        raise ConfigError(f"{where}: 'obstacle_kinds' must be a non-empty subset of [box, cylinder]")
    steps = float(cfg["control_dt_s"]) / float(cfg["physics_dt_s"])
    if abs(steps - round(steps)) > 1e-9:
        raise ConfigError(f"{where}: 'control_dt_s' must be a whole multiple of 'physics_dt_s'")
    limit = _require(cfg, "episode_limit", where)
    if not (isinstance(limit, int) and limit >= 1):
        raise ConfigError(f"{where}: 'episode_limit' must be an integer >= 1")
    reward = _require(cfg, "reward", where)
    for key in ("w_progress", "w_success", "w_collision", "w_time"):
        w = float(_require(reward, key, f"{where}: reward"))
        if not math.isfinite(w):
            raise ConfigError(f"{where}: reward '{key}' must be finite")
    t = float(_require(cfg, "nominal_success_target", where))
    if not 0.0 <= t <= 1.0:
        raise ConfigError(f"{where}: 'nominal_success_target' must be in [0, 1], got {t}")
    n = _require(cfg, "eval_episodes_per_seed", where)
    if not (isinstance(n, int) and n >= 1):
        raise ConfigError(f"{where}: 'eval_episodes_per_seed' must be an integer >= 1")
    _require(cfg, "eval_seed", where)
    half = float(cfg["arena_size_m"]) / 2
    if float(cfg["robot_radius_m"]) + float(cfg["start_goal_clearance_m"]) >= half:
        raise ConfigError(f"{where}: robot does not fit in the arena")
    return dict(cfg)


def load_task(path: str | Path = "task.yaml") -> dict[str, Any]:
    return validate_task(load_yaml(path), str(path))


def substeps(task: Mapping[str, Any]) -> int:
    return int(round(float(task["control_dt_s"]) / float(task["physics_dt_s"])))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out
