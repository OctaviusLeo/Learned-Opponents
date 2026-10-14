"""Engine-independent Obs_Action_Spec.

Defines the FIXED observation and action layout shared by the training
environment and the game's in-engine observation builder. The only
engine-specific input is ``raycasts``; everything else is computed here from
geometry so the Python side and the C++ game side can produce byte-identical
layouts.

Observation vector ordering (dimension ``N_rays + 5``)::

    [ray_0 ... ray_{N-1}, goal_dir_x, goal_dir_y, goal_distance,
     vel_linear, vel_angular]

Action vector (dimension 2)::

    [wheel_left, wheel_right]        # normalized wheel velocity, each in [-1, 1]

This spec EXCLUDES all legged-locomotion fields (Requirement 2.5): no joint
angles, no foot contacts, no gait phase, no body orientation beyond what the
goal-direction vector already encodes.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5.

Run as a script for a demo::

    python -m learned_opponents.spec.obs_action_spec
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# --- Fixed slice layout ------------------------------------------------------
# These constants let the game mirror the layout exactly. The non-ray tail is
# always the last 5 entries, in this order.
N_NON_RAY_FIELDS = 5
ACTION_DIM = 2

# Indices WITHIN the non-ray tail (offsets after the N_rays ray block).
TAIL_GOAL_DIR_X = 0
TAIL_GOAL_DIR_Y = 1
TAIL_GOAL_DISTANCE = 2
TAIL_VEL_LINEAR = 3
TAIL_VEL_ANGULAR = 4

# Action indices.
ACT_WHEEL_LEFT = 0
ACT_WHEEL_RIGHT = 1


@dataclass(frozen=True)
class SpecConfig:
    """Configured_Values needed to build an observation.

    Attributes:
        n_rays: number of rangefinder rays (N_rays).
        max_range_m: ray max range in meters; rays normalized by this.
        max_wheel_speed: rad/s; velocities normalized by this.
        arena_diagonal_m: arena diagonal in meters; goal distance normalized by this.
    """

    n_rays: int
    max_range_m: float
    max_wheel_speed: float
    arena_diagonal_m: float

    @property
    def obs_dim(self) -> int:
        return self.n_rays + N_NON_RAY_FIELDS

    @classmethod
    def from_mapping(cls, config: Mapping) -> "SpecConfig":
        """Build a SpecConfig from a dict-like config (e.g. parsed task.yaml).

        ``arena_diagonal_m`` is used directly if present, otherwise derived
        from ``arena.width_m`` / ``arena.height_m``.
        """
        n_rays = int(config["n_rays"])
        max_range_m = float(config["max_range_m"])
        max_wheel_speed = float(config["max_wheel_speed"])

        if "arena_diagonal_m" in config:
            arena_diagonal_m = float(config["arena_diagonal_m"])
        else:
            arena = config["arena"]
            w = float(arena["width_m"])
            h = float(arena["height_m"])
            arena_diagonal_m = float(np.hypot(w, h))

        return cls(
            n_rays=n_rays,
            max_range_m=max_range_m,
            max_wheel_speed=max_wheel_speed,
            arena_diagonal_m=arena_diagonal_m,
        )


# Index-map helpers so the game can mirror the slice layout. ------------------
def ray_slice(config: SpecConfig) -> slice:
    """Slice of the observation vector holding the ray block."""
    return slice(0, config.n_rays)


def goal_dir_x_index(config: SpecConfig) -> int:
    return config.n_rays + TAIL_GOAL_DIR_X


def goal_dir_y_index(config: SpecConfig) -> int:
    return config.n_rays + TAIL_GOAL_DIR_Y


def goal_distance_index(config: SpecConfig) -> int:
    return config.n_rays + TAIL_GOAL_DISTANCE


def vel_linear_index(config: SpecConfig) -> int:
    return config.n_rays + TAIL_VEL_LINEAR


def vel_angular_index(config: SpecConfig) -> int:
    return config.n_rays + TAIL_VEL_ANGULAR


def build_observation(
    raycasts: Sequence[float] | np.ndarray,
    goal_pose: Mapping,
    robot_vel: Mapping,
    config: SpecConfig | Mapping,
) -> np.ndarray:
    """Construct the fixed-layout observation vector.

    Args:
        raycasts: the ONLY engine-specific input. ``N_rays`` ray distances in
            meters (unnormalized). The engine provides these; everything else
            is computed here.
        goal_pose: mapping with the goal position and the robot pose, in the
            world frame::

                {
                    "goal_x": float, "goal_y": float,
                    "robot_x": float, "robot_y": float,
                    "robot_theta": float,   # robot heading in radians
                }

        robot_vel: mapping ``{"linear": float, "angular": float}`` — the
            robot's linear (m/s) and angular (rad/s) velocity. Both are
            normalized by ``max_wheel_speed`` and clamped to [-1, 1].
        config: a SpecConfig, or a dict-like config (parsed task.yaml).

    Returns:
        A ``float32`` numpy array of length ``N_rays + 5`` with every field in
        its declared range.
    """
    if not isinstance(config, SpecConfig):
        config = SpecConfig.from_mapping(config)

    rays = np.asarray(raycasts, dtype=np.float32)
    if rays.shape != (config.n_rays,):
        raise ValueError(
            f"raycasts must have shape ({config.n_rays},), got {rays.shape}"
        )

    # Rays: normalize by max_range_m and clamp to [0, 1].
    rays_norm = np.clip(rays / config.max_range_m, 0.0, 1.0)

    # Goal direction/distance in the ROBOT's local frame.
    dx = float(goal_pose["goal_x"]) - float(goal_pose["robot_x"])
    dy = float(goal_pose["goal_y"]) - float(goal_pose["robot_y"])
    theta = float(goal_pose["robot_theta"])

    # Rotate world-frame (dx, dy) into the robot frame (rotate by -theta).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    distance_m = float(np.hypot(local_x, local_y))
    if distance_m > 1e-9:
        goal_dir_x = local_x / distance_m
        goal_dir_y = local_y / distance_m
    else:
        # At the goal: direction is undefined; use zero vector.
        goal_dir_x = 0.0
        goal_dir_y = 0.0

    # Unit vector components are already in [-1, 1]; clamp for float safety.
    goal_dir_x = float(np.clip(goal_dir_x, -1.0, 1.0))
    goal_dir_y = float(np.clip(goal_dir_y, -1.0, 1.0))

    # Goal distance normalized by arena diagonal, clipped to [0, 1].
    goal_distance = float(np.clip(distance_m / config.arena_diagonal_m, 0.0, 1.0))

    # Velocities normalized by max wheel speed, clamped to [-1, 1].
    vel_linear = float(
        np.clip(float(robot_vel["linear"]) / config.max_wheel_speed, -1.0, 1.0)
    )
    vel_angular = float(
        np.clip(float(robot_vel["angular"]) / config.max_wheel_speed, -1.0, 1.0)
    )

    tail = np.array(
        [goal_dir_x, goal_dir_y, goal_distance, vel_linear, vel_angular],
        dtype=np.float32,
    )

    obs = np.concatenate([rays_norm, tail]).astype(np.float32)
    assert obs.shape == (config.obs_dim,)
    return obs


def build_action(wheel_left: float, wheel_right: float) -> np.ndarray:
    """Validate and clamp a 2-dim normalized wheel-velocity action.

    Args:
        wheel_left: normalized left wheel velocity (mapped to rad/s by
            ``max_wheel_speed`` downstream).
        wheel_right: normalized right wheel velocity.

    Returns:
        A ``float32`` array ``[wheel_left, wheel_right]`` with each component
        clamped to [-1, 1] (differential drive).
    """
    action = np.array([wheel_left, wheel_right], dtype=np.float32)
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    assert action.shape == (ACTION_DIM,)
    return action


def _demo() -> None:
    """Task 4.1 demo: build a correct-length, in-range observation + action."""
    config = SpecConfig(
        n_rays=16,
        max_range_m=5.0,
        max_wheel_speed=10.0,
        arena_diagonal_m=float(np.hypot(10.0, 10.0)),
    )

    rng = np.random.default_rng(0)
    raycasts = rng.uniform(0.0, 7.0, size=config.n_rays)  # some beyond max_range
    goal_pose = {
        "goal_x": 3.0,
        "goal_y": 4.0,
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_theta": 0.5,
    }
    robot_vel = {"linear": 6.0, "angular": -12.0}  # angular beyond max -> clamps

    obs = build_observation(raycasts, goal_pose, robot_vel, config)
    action = build_action(1.5, -0.3)  # left beyond range -> clamps to 1.0

    print(f"obs_dim expected = n_rays + 5 = {config.obs_dim}")
    print(f"obs length       = {obs.shape[0]}")
    print(f"obs dtype        = {obs.dtype}")
    print(f"rays in [0,1]        : {bool(np.all((obs[ray_slice(config)] >= 0) & (obs[ray_slice(config)] <= 1)))}")
    print(f"goal_dir_x in [-1,1] : {-1.0 <= obs[goal_dir_x_index(config)] <= 1.0}")
    print(f"goal_dir_y in [-1,1] : {-1.0 <= obs[goal_dir_y_index(config)] <= 1.0}")
    print(f"goal_distance [0,1]  : {0.0 <= obs[goal_distance_index(config)] <= 1.0}")
    print(f"vel_linear in [-1,1] : {-1.0 <= obs[vel_linear_index(config)] <= 1.0}")
    print(f"vel_angular in [-1,1]: {-1.0 <= obs[vel_angular_index(config)] <= 1.0}")
    print(f"action           = {action.tolist()} (clamped, in [-1,1])")
    print(f"full obs vector  = {obs.tolist()}")


if __name__ == "__main__":
    _demo()
