# Obs_Action_Spec (engine-independent)

This document is the human-readable contract for the observation and action
layout. It matches `obs_action_spec.py` **field-for-field**. The game's C++
observation builder mirrors this document so that the vector built in-engine is
identical to the one built in Python.

The **only** engine-specific input is `raycasts`. Every other field is computed
from geometry (goal pose, robot pose, robot velocity), so the Python side and
the game side compute them the same way.

## Configured_Values

| Name | Symbol | Meaning |
|---|---|---|
| `n_rays` | `N_rays` | number of rangefinder rays |
| `max_range_m` | — | ray max range (m); rays normalized by this |
| `max_wheel_speed` | — | max wheel velocity (rad/s); velocities normalized by this |
| `arena_diagonal_m` | — | arena diagonal (m); goal distance normalized by this |

`arena_diagonal_m` is derived from arena size when not given directly:
`arena_diagonal_m = sqrt(width_m^2 + height_m^2)`.

## Observation vector

Dimension: **`N_rays + 5`**. dtype: `float32`.

Fixed ordering:

```
[ ray_0, ray_1, ..., ray_{N_rays-1}, goal_dir_x, goal_dir_y, goal_distance, vel_linear, vel_angular ]
```

| Slice | Field | Count | Units (raw) | Range (normalized) | Construction rule |
|---|---|---|---|---|---|
| `[0 : N_rays]` | ray distances | `N_rays` | meters | `[0, 1]` | `clip(ray / max_range_m, 0, 1)` per ray |
| `[N_rays + 0]` | `goal_dir_x` | 1 | unit vector | `[-1, 1]` | x of unit vector to goal, in robot local frame |
| `[N_rays + 1]` | `goal_dir_y` | 1 | unit vector | `[-1, 1]` | y of unit vector to goal, in robot local frame |
| `[N_rays + 2]` | `goal_distance` | 1 | meters | `[0, 1]` | `clip(distance_m / arena_diagonal_m, 0, 1)` |
| `[N_rays + 3]` | `vel_linear` | 1 | m/s | `[-1, 1]` | `clip(linear / max_wheel_speed, -1, 1)` |
| `[N_rays + 4]` | `vel_angular` | 1 | rad/s | `[-1, 1]` | `clip(angular / max_wheel_speed, -1, 1)` |

### Goal direction / distance (robot local frame)

Given goal position `(goal_x, goal_y)`, robot position `(robot_x, robot_y)`,
and robot heading `robot_theta` (radians), all in the world frame:

```
dx = goal_x - robot_x
dy = goal_y - robot_y

# rotate world delta into the robot frame (rotate by -theta)
local_x =  cos(theta) * dx + sin(theta) * dy
local_y = -sin(theta) * dx + cos(theta) * dy

distance_m = hypot(local_x, local_y)
goal_dir_x = local_x / distance_m      # 0 if distance_m ~ 0
goal_dir_y = local_y / distance_m      # 0 if distance_m ~ 0
```

`goal_dir_*` are unit-vector components (already in `[-1, 1]`). When the robot
is at the goal (`distance_m ~ 0`), the direction is undefined and both
components are set to `0`.

## Action vector

Dimension: **2**. dtype: `float32`. Differential drive.

```
[ wheel_left, wheel_right ]
```

| Index | Field | Range | Meaning |
|---|---|---|---|
| `0` | `wheel_left` | `[-1, 1]` | normalized left-wheel velocity |
| `1` | `wheel_right` | `[-1, 1]` | normalized right-wheel velocity |

Each component is a normalized wheel velocity, mapped to rad/s by
`max_wheel_speed` downstream. `build_action` clamps both components to `[-1, 1]`.

## Excluded fields (Requirement 2.5)

This is a wheeled differential-drive robot. The spec **excludes all
legged-locomotion fields**: no joint angles, no joint velocities, no foot/toe
contact flags, no gait phase or clock signal, and no full body-orientation
quaternion. Heading information is carried solely by `goal_dir_x`/`goal_dir_y`
in the robot frame.
