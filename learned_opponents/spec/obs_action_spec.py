"""Engine-independent Obs_Action_Spec (single source of truth).

Observation, dimension ``n_rays + 5``, in this fixed order::

    [ray_0 .. ray_{n_rays-1}, goal_dir_x, goal_dir_y, goal_distance, vel_linear, vel_angular]

Action, dimension 2::

    [wheel_left, wheel_right]   normalized wheel angular velocity, each in [-1, 1]

Frames: right-handed robot frame, x forward, y left, z up; angles and yaw rate
are positive counter-clockwise seen from above (MuJoCo's convention). The game
converts Unreal's left-handed centimetre frame with the ``unreal_*`` helpers
below before calling the same rules (see ``spec.md``).

The construction functions are written against a tiny array API (``xp`` is
numpy or ``jax.numpy``) so the training environment and this reference share
one implementation. This module imports only numpy and the standard library,
so CI can test it without a GPU framework. The task is goal navigation with
obstacle avoidance; there are no legged-locomotion fields (Requirement 2.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

N_TAIL = 5
ACTION_DIM = 2
# Offsets of the non-ray fields after the ray block.
GOAL_DIR_X, GOAL_DIR_Y, GOAL_DISTANCE, VEL_LINEAR, VEL_ANGULAR = range(N_TAIL)
WHEEL_LEFT, WHEEL_RIGHT = range(ACTION_DIM)
FIELD_NAMES = ("goal_dir_x", "goal_dir_y", "goal_distance", "vel_linear", "vel_angular")


@dataclass(frozen=True)
class SpecConfig:
    """The task.yaml values the observation and action rules depend on."""

    n_rays: int
    ray_fov_deg: float
    max_range_m: float
    arena_size_m: float
    wheel_radius_m: float
    track_width_m: float
    max_wheel_speed: float

    def __post_init__(self) -> None:
        if self.n_rays < 2:
            raise ValueError(f"n_rays must be >= 2, got {self.n_rays}")
        if not 0.0 < self.ray_fov_deg <= 360.0:
            raise ValueError(f"ray_fov_deg must be in (0, 360], got {self.ray_fov_deg}")
        for name in ("max_range_m", "arena_size_m", "wheel_radius_m", "track_width_m", "max_wheel_speed"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "SpecConfig":
        """Pick the spec fields out of a parsed task.yaml mapping."""
        missing = [f.name for f in fields(cls) if f.name not in cfg]
        if missing:
            raise KeyError(f"task config is missing {missing}")
        values = {f.name: float(cfg[f.name]) for f in fields(cls)}
        values["n_rays"] = int(cfg["n_rays"])
        return cls(**values)

    @property
    def obs_dim(self) -> int:
        return self.n_rays + N_TAIL

    @property
    def v_max(self) -> float:
        """Forward speed with both wheels at max_wheel_speed (m/s)."""
        return self.wheel_radius_m * self.max_wheel_speed

    @property
    def omega_max(self) -> float:
        """Yaw rate with the wheels at +/- max_wheel_speed (rad/s)."""
        return 2.0 * self.wheel_radius_m * self.max_wheel_speed / self.track_width_m

    @property
    def arena_diagonal_m(self) -> float:
        return math.sqrt(2.0) * self.arena_size_m


def as_spec_config(cfg: SpecConfig | Mapping[str, Any]) -> SpecConfig:
    return cfg if isinstance(cfg, SpecConfig) else SpecConfig.from_mapping(cfg)


# --- Ray layout ---------------------------------------------------------------

def ray_angles_rad(cfg: SpecConfig | Mapping[str, Any]) -> np.ndarray:
    """Ray angles in the robot frame, index order.

    360 deg: ``theta_i = -180 + i * 360 / n``, so index 0 points straight back,
    indices increase counter-clockwise, and index ``n / 2`` points ahead.
    Narrower fans: ``theta_i = -fov/2 + i * fov / (n - 1)``, index 0 at the
    rightmost edge.
    """
    c = as_spec_config(cfg)
    step = c.ray_fov_deg / (c.n_rays if c.ray_fov_deg >= 360.0 else c.n_rays - 1)
    return np.radians(-c.ray_fov_deg / 2.0 + np.arange(c.n_rays) * step)


def ray_directions(cfg: SpecConfig | Mapping[str, Any]) -> np.ndarray:
    """Unit ray directions ``(n_rays, 2)`` in the robot frame (x forward, y left)."""
    th = ray_angles_rad(cfg)
    return np.stack([np.cos(th), np.sin(th)], axis=1)


# --- Construction rules (array-API, shared with the JAX environment) ----------

def normalize_rays(xp, ranges, max_range_m: float, self_hit=None):
    """Map misses and self-hits to max range, clamp, and scale to [0, 1].

    A reading is replaced by ``max_range_m`` when it is negative (MuJoCo's
    miss value is -1), above ``max_range_m``, not finite, or flagged as a hit on
    the robot itself.
    """
    r = xp.asarray(ranges)
    bad = (r < 0) | (r > max_range_m) | ~xp.isfinite(r)
    if self_hit is not None:
        bad = bad | xp.asarray(self_hit, dtype=bool)
    r = xp.where(bad, max_range_m, r)
    return xp.clip(r / max_range_m, 0.0, 1.0)


def goal_features(xp, robot_xy, robot_yaw, goal_xy, arena_diagonal_m: float):
    """Goal direction (robot frame unit vector) and normalized distance.

    Returns ``(goal_dir_x, goal_dir_y, goal_distance)``; the direction is
    ``(0, 0)`` when the goal coincides with the robot centre.
    """
    robot_xy = xp.asarray(robot_xy)
    goal_xy = xp.asarray(goal_xy)
    dx = goal_xy[..., 0] - robot_xy[..., 0]
    dy = goal_xy[..., 1] - robot_xy[..., 1]
    c, s = xp.cos(robot_yaw), xp.sin(robot_yaw)
    lx = c * dx + s * dy        # rotate the world offset by -yaw
    ly = -s * dx + c * dy
    dist = xp.sqrt(lx * lx + ly * ly)
    safe = xp.where(dist > 1e-9, dist, 1.0)
    gx = xp.where(dist > 1e-9, lx / safe, 0.0)
    gy = xp.where(dist > 1e-9, ly / safe, 0.0)
    return (xp.clip(gx, -1.0, 1.0), xp.clip(gy, -1.0, 1.0),
            xp.clip(dist / arena_diagonal_m, 0.0, 1.0))


def build_observation_xp(xp, ranges, robot_xy, robot_yaw, goal_xy, vel_linear, vel_angular,
                         cfg: SpecConfig, self_hit=None):
    """Vectorizable observation builder; leading batch dimensions are allowed.

    ``ranges`` has shape ``(..., n_rays)`` in metres, ``robot_xy``/``goal_xy``
    ``(..., 2)`` in metres in the world frame, ``robot_yaw`` in radians, and the
    velocities are the robot-frame forward speed (m/s) and yaw rate (rad/s).
    """
    rays = normalize_rays(xp, ranges, cfg.max_range_m, self_hit)
    gx, gy, gd = goal_features(xp, robot_xy, robot_yaw, goal_xy, cfg.arena_diagonal_m)
    vl = xp.clip(xp.asarray(vel_linear) / cfg.v_max, -1.0, 1.0)
    va = xp.clip(xp.asarray(vel_angular) / cfg.omega_max, -1.0, 1.0)
    tail = xp.stack([gx, gy, gd, vl, va], axis=-1)
    return xp.concatenate([rays, tail], axis=-1)


def build_observation(raycasts: Sequence[float] | np.ndarray, goal_pose: Mapping[str, float],
                      robot_vel: Mapping[str, float], config: SpecConfig | Mapping[str, Any],
                      self_hit: Sequence[bool] | None = None) -> np.ndarray:
    """Reference observation builder for one robot (float32, ``n_rays + 5``).

    Args:
        raycasts: the only engine-specific input: ``n_rays`` raw ray distances
            in metres in index order; negative means no hit.
        goal_pose: ``robot_x``, ``robot_y``, ``robot_yaw`` (radians) and
            ``goal_x``, ``goal_y`` in the world frame, metres.
        robot_vel: ``linear`` (robot-frame forward speed, m/s) and ``angular``
            (yaw rate, rad/s, counter-clockwise positive).
        config: SpecConfig or a parsed task.yaml mapping.
        self_hit: optional per-ray flags for hits on the robot's own geometry.
    """
    cfg = as_spec_config(config)
    r = np.asarray(raycasts, dtype=np.float64)
    if r.shape != (cfg.n_rays,):
        raise ValueError(f"raycasts must have shape ({cfg.n_rays},), got {r.shape}")
    obs = build_observation_xp(
        np, r,
        np.array([goal_pose["robot_x"], goal_pose["robot_y"]], dtype=np.float64),
        float(goal_pose["robot_yaw"]),
        np.array([goal_pose["goal_x"], goal_pose["goal_y"]], dtype=np.float64),
        float(robot_vel["linear"]), float(robot_vel["angular"]), cfg, self_hit)
    return obs.astype(np.float32)


def observation_bounds(cfg: SpecConfig | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Declared per-field ``(low, high)`` ranges of the observation."""
    c = as_spec_config(cfg)
    low = np.concatenate([np.zeros(c.n_rays), [-1.0, -1.0, 0.0, -1.0, -1.0]]).astype(np.float32)
    high = np.ones(c.obs_dim, dtype=np.float32)
    return low, high


# --- Action -------------------------------------------------------------------

def build_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate a policy output ``[wheel_left, wheel_right]`` and clip to [-1, 1]."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action must have {ACTION_DIM} elements, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"action must be finite, got {a.tolist()}")
    return np.clip(a, -1.0, 1.0)


def wheel_speeds_rad_s(xp, action, cfg: SpecConfig):
    """Clipped normalized action -> left/right wheel angular velocity (rad/s)."""
    return xp.clip(xp.asarray(action), -1.0, 1.0) * cfg.max_wheel_speed


def body_velocity(xp, action, cfg: SpecConfig):
    """Ideal differential-drive kinematics: action -> (forward m/s, yaw rate rad/s)."""
    w = wheel_speeds_rad_s(xp, action, cfg)
    wl, wr = w[..., WHEEL_LEFT], w[..., WHEEL_RIGHT]
    v = cfg.wheel_radius_m * (wl + wr) / 2.0
    omega = cfg.wheel_radius_m * (wr - wl) / cfg.track_width_m
    return v, omega


# --- Unreal conversion (left-handed, x forward, y right, z up, centimetres) ---

def unreal_to_spec_xy(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Unreal world position (cm) -> spec world position (m): y is mirrored."""
    return x_cm / 100.0, -y_cm / 100.0


def unreal_to_spec_yaw(yaw_deg: float) -> float:
    """Unreal yaw (degrees, clockwise from above) -> spec yaw (radians, CCW)."""
    return -math.radians(yaw_deg)


def unreal_to_spec_yaw_rate(yaw_rate_deg_s: float) -> float:
    return -math.radians(yaw_rate_deg_s)


def unreal_to_spec_speed(speed_cm_s: float) -> float:
    return speed_cm_s / 100.0


def spec_ray_angle_to_unreal_yaw_offset_deg(theta_rad: float) -> float:
    """Yaw offset (degrees, Unreal sign) of a ray at spec angle ``theta_rad``."""
    return -math.degrees(theta_rad)


def _demo() -> None:
    cfg = SpecConfig(n_rays=24, ray_fov_deg=360.0, max_range_m=4.0, arena_size_m=8.0,
                     wheel_radius_m=0.05, track_width_m=0.30, max_wheel_speed=20.0)
    rng = np.random.default_rng(0)
    rays = rng.uniform(-1.0, 6.0, cfg.n_rays)
    pose = {"robot_x": -1.0, "robot_y": 0.5, "robot_yaw": 0.7, "goal_x": 2.5, "goal_y": -3.0}
    vel = {"linear": 0.6, "angular": -9.0}
    obs = build_observation(rays, pose, vel, cfg)
    low, high = observation_bounds(cfg)
    print(f"obs_dim={obs.shape[0]} (expected n_rays + 5 = {cfg.obs_dim}), dtype={obs.dtype}")
    print(f"all fields in declared ranges: {bool(np.all((obs >= low) & (obs <= high)))}")
    print("tail:", dict(zip(FIELD_NAMES, np.round(obs[cfg.n_rays:], 4).tolist())))
    print("action:", build_action([1.7, -0.25]).tolist())


if __name__ == "__main__":
    _demo()
