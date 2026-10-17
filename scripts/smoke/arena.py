"""Minimal smoke-test arena shared by both framework checks (task 2.1).

Builds an MJCF string with primitive geoms only: a floor plane, four walls,
a few box and upright-capsule obstacles, and a differential-drive robot carrying
``n_rays`` horizontal rangefinders plus one extra rangefinder that points
straight up and can never hit anything (it proves the miss value).

Contacts: only the wheels and casters touch the floor. Obstacles and walls
have no contact bits, because the navigation task detects collisions from
geometry; rays ignore contact bits, so they still hit obstacles and walls.

The rays sit at 0.15 m, above the wheel tops (0.10 m), so the only robot
geometry at ray height is the chassis, which the rangefinder excludes because
it belongs to the same body as the ray sites.
"""

from __future__ import annotations

import math

N_RAYS = 24
RAY_FOV_DEG = 360.0
ARENA_SIZE_M = 8.0
WHEEL_RADIUS_M = 0.05
TRACK_WIDTH_M = 0.30
RAY_HEIGHT_M = 0.15
MAX_WHEEL_SPEED = 20.0


def ray_angles_rad(n_rays: int = N_RAYS, fov_deg: float = RAY_FOV_DEG) -> list[float]:
    """Ray angles from the design: index 0 points back (360 deg), counter-clockwise."""
    if fov_deg >= 360.0:
        step = fov_deg / n_rays
    else:
        step = fov_deg / (n_rays - 1)
    return [math.radians(-fov_deg / 2.0 + i * step) for i in range(n_rays)]


def build_xml(n_rays: int = N_RAYS) -> str:
    half = ARENA_SIZE_M / 2.0
    wall_t, wall_h = 0.05, 0.25
    # Round obstacles are upright capsules: MJX-JAX 3.16.0 has no cylinder ray
    # routine (ray.py _RAY_FUNC), and at ray height a capsule's straight
    # section has exactly a cylinder's cross-section.
    obstacles = [
        ("box", (1.5, 1.0), 0.4),
        ("box", (-2.0, -1.5), 0.3),
        ("capsule", (-1.5, 2.0), 0.35),
        ("capsule", (2.0, -2.0), 0.25),
    ]
    geoms = []
    for name, (x, y, sx, sy) in {
        "wall_n": (0.0, half + wall_t, half + 2 * wall_t, wall_t),
        "wall_s": (0.0, -half - wall_t, half + 2 * wall_t, wall_t),
        "wall_e": (half + wall_t, 0.0, wall_t, half),
        "wall_w": (-half - wall_t, 0.0, wall_t, half),
    }.items():
        geoms.append(
            f'<geom name="{name}" type="box" pos="{x} {y} {wall_h}" '
            f'size="{sx} {sy} {wall_h}" contype="0" conaffinity="0"/>'
        )
    for i, (kind, (x, y), r) in enumerate(obstacles):
        size = f"{r} {r} {wall_h}" if kind == "box" else f"{r} {wall_h}"
        geoms.append(
            f'<geom name="obs{i}" type="{kind}" pos="{x} {y} {wall_h}" size="{size}" '
            f'contype="0" conaffinity="0"/>'
        )

    site_z = RAY_HEIGHT_M - WHEEL_RADIUS_M  # body origin sits at wheel-axle height
    sites, sensors = [], []
    for i, th in enumerate(ray_angles_rad(n_rays)):
        sites.append(
            f'<site name="ray{i}" pos="0 0 {site_z}" zaxis="{math.cos(th):.9f} {math.sin(th):.9f} 0" size="0.005"/>'
        )
        sensors.append(f'<rangefinder name="ray{i}" site="ray{i}"/>')
    sites.append(f'<site name="ray_up" pos="0 0 {site_z}" zaxis="0 0 1" size="0.005"/>')
    sensors.append('<rangefinder name="ray_up" site="ray_up"/>')

    yw = TRACK_WIDTH_M / 2.0
    return f"""
<mujoco model="smoke_arena">
  <option timestep="0.01" integrator="implicitfast"/>
  <default>
    <geom contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
  </default>
  <asset>
    <!-- MJX's ray filter indexes mat_rgba, so the model needs at least one material. -->
    <material name="floor" rgba="0.5 0.5 0.5 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="{half + 1} {half + 1} 0.1" material="floor"/>
    {chr(10).join(geoms)}
    <body name="robot" pos="0 0 {WHEEL_RADIUS_M}">
      <freejoint name="root"/>
      <geom name="chassis" type="cylinder" size="0.15 0.03" pos="0 0 0.05" mass="2.0" contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.02" pos="0.1 0 -0.03" mass="0.01" condim="1"/>
      <geom name="caster_b" type="sphere" size="0.02" pos="-0.1 0 -0.03" mass="0.01" condim="1"/>
      {chr(10).join(sites)}
      <body name="wheel_l" pos="0 {yw} 0">
        <joint name="wheel_l" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="wheel_l" type="sphere" size="{WHEEL_RADIUS_M}" mass="0.1"/>
      </body>
      <body name="wheel_r" pos="0 {-yw} 0">
        <joint name="wheel_r" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="wheel_r" type="sphere" size="{WHEEL_RADIUS_M}" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="wheel_l" body2="wheel_r"/>
  </contact>
  <actuator>
    <velocity name="wheel_l" joint="wheel_l" kv="0.2" ctrlrange="{-MAX_WHEEL_SPEED} {MAX_WHEEL_SPEED}"/>
    <velocity name="wheel_r" joint="wheel_r" kv="0.2" ctrlrange="{-MAX_WHEEL_SPEED} {MAX_WHEEL_SPEED}"/>
  </actuator>
  <sensor>
    {chr(10).join(sensors)}
  </sensor>
</mujoco>
"""
