# Obs_Action_Spec

The engine-independent contract between training (MuJoCo Warp) and the game
(Unreal Engine 5.8). The executable reference is
`learned_opponents/spec/obs_action_spec.py`; the game's C++ builder
(`game/LearnedOpponents/Source/LearnedOpponents/Obs/`) reimplements these
rules and is checked against Python-generated cases
(`results/fixtures/obs_cases.json`, tolerance 1e-5). All values come from
`config/task.yaml`; the defaults are shown in brackets.

The task is goal navigation with obstacle avoidance by a differential-drive
robot. There are no legged-locomotion fields (no joint angles, foot contacts,
gait phase, or body orientation beyond what the goal direction carries).

## Frames

- **Spec frame (world and robot):** right-handed, x forward, y left, z up,
  metres, radians. Angles and yaw rates are positive counter-clockwise seen
  from above. This is MuJoCo's convention.
- **Robot frame:** origin at the robot centre, x along the robot's heading.

## Observation, dimension `n_rays + 5` [29], float32

| Index | Field | Construction | Range |
| --- | --- | --- | --- |
| `0 .. n_rays-1` | `ray_i` | range along ray `i`, misses mapped (below), divided by `max_range_m` [4.0] | [0, 1] |
| `n_rays + 0` | `goal_dir_x` | x of the unit vector from robot to goal, robot frame | [-1, 1] |
| `n_rays + 1` | `goal_dir_y` | y of the same unit vector | [-1, 1] |
| `n_rays + 2` | `goal_distance` | robot-to-goal distance / arena diagonal (`sqrt(2) * arena_size_m` [11.314 m]), clipped to 1 | [0, 1] |
| `n_rays + 3` | `vel_linear` | robot-frame forward speed / `v_max`, clipped | [-1, 1] |
| `n_rays + 4` | `vel_angular` | yaw rate / `omega_max`, clipped | [-1, 1] |

### Rays

- `n_rays` [24] horizontal rays start at the robot centre at height
  `ray_height_m` [0.15 m] (above the 0.10 m wheel tops).
- Angle of ray `i` in the robot frame:
  - `ray_fov_deg` = 360: `theta_i = -180 deg + i * 360 / n_rays`. Index 0 points
    straight back, indices increase counter-clockwise, index `n_rays / 4`
    points right (-90 deg), `n_rays / 2` ahead (0 deg), `3 n_rays / 4` left (+90 deg).
  - narrower fans: `theta_i = -fov/2 + i * fov / (n_rays - 1)`; index 0 is the
    rightmost edge.
- With the defaults the angles are -180, -165, ..., +165 degrees (15 degree steps).
- **Miss and self-hit mapping:** a reading that is negative (MuJoCo returns
  -1 for no hit), above `max_range_m`, not finite, or a hit on the robot's own
  geometry becomes `max_range_m` before dividing. Both engines exclude the
  robot from its own rays (MuJoCo: the rangefinder skips the site's body and
  the rays are above the wheels; Unreal: the trace ignores the robot actor).
- Sensor noise (randomized training only) is added to hit distances in metres
  before this mapping.

### Goal features

`d = goal_xy - robot_xy` in the world frame is rotated into the robot frame
by `-yaw`: `lx = cos(yaw) dx + sin(yaw) dy`, `ly = -sin(yaw) dx + cos(yaw) dy`.
`goal_dir = (lx, ly) / |(lx, ly)|`, or `(0, 0)` when the distance is below
1e-9 m.

### Velocity normalizers

- `v_max = wheel_radius_m * max_wheel_speed` [0.05 * 20 = 1.0 m/s]
- `omega_max = 2 * wheel_radius_m * max_wheel_speed / track_width_m` [6.667 rad/s]

## Action, dimension 2, float32

| Index | Field | Meaning | Range |
| --- | --- | --- | --- |
| 0 | `wheel_left` | left wheel angular velocity / `max_wheel_speed` [20 rad/s] | [-1, 1] |
| 1 | `wheel_right` | right wheel angular velocity / `max_wheel_speed` | [-1, 1] |

Actions are clipped to [-1, 1] before use. Positive wheel speed drives the
robot forward. Ideal differential-drive kinematics (the game's kinematic
robot): `v = r (w_l + w_r) / 2`, `omega = r (w_r - w_l) / track_width_m`.

## Timing

The policy runs once per control step of `control_dt_s` [0.05 s, 20 Hz] in
both engines; the action is held for the whole step. The observation is built
from the state at the end of the previous control step.

## Unreal conversion (left-handed, x forward, y right, z up, centimetres)

| Quantity | Unreal | Spec |
| --- | --- | --- |
| position | `(x_cm, y_cm)` | `(x_cm / 100, -y_cm / 100)` |
| yaw | `yaw_deg`, clockwise seen from above | `-radians(yaw_deg)` |
| yaw rate | `deg/s`, clockwise | `-radians(rate)` |
| forward speed | `cm/s` along the actor's forward vector | `/ 100` |
| ray `i` direction | actor yaw + `(-degrees(theta_i))` | angle `theta_i` |
| distances | cm | `/ 100` |

The game converts each quantity first and then applies exactly the rules
above, so the only intended difference between engines is the raycast source.
