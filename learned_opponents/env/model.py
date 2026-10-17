"""Fill the MJCF template ``model.xml`` from the task config (standard library only).

Geometry names that the environment looks up are defined here, so the
environment and the tests agree on them.
"""

from __future__ import annotations

import math
from importlib import resources
from string import Template
from typing import Any, Mapping

from learned_opponents.spec.obs_action_spec import ray_angles_rad

PARK_Z = -10.0       # parked obstacle bodies sit far below the floor, out of every ray's path
WALL_THICKNESS = 0.1


def box_body(j: int) -> str:
    return f"obs_box_{j}"


def cyl_body(j: int) -> str:
    return f"obs_cyl_{j}"


def ray_site(i: int) -> str:
    return f"ray_{i}"


def build_mjcf(task: Mapping[str, Any]) -> str:
    half = float(task["arena_size_m"]) / 2.0
    k = int(task["n_obstacles"][1])
    s_max = float(task["obstacle_size_m"][1])
    h_obs = float(task["obstacle_height_m"]) / 2.0
    h_wall = float(task["wall_height_m"]) / 2.0
    t = WALL_THICKNESS / 2.0

    walls = []
    for name, x, y, sx, sy in (("wall_n", 0.0, half + t, half + 2 * t, t),
                               ("wall_s", 0.0, -half - t, half + 2 * t, t),
                               ("wall_e", half + t, 0.0, t, half),
                               ("wall_w", -half - t, 0.0, t, half)):
        walls.append(f'    <geom name="{name}" type="box" pos="{x} {y} {h_wall}" '
                     f'size="{sx} {sy} {h_wall}" material="wall"/>')

    obstacles = []
    for j in range(k):
        # Compiled at the largest size so bounding volumes stay conservative after resizing.
        obstacles.append(f'    <body name="{box_body(j)}" mocap="true" pos="0 0 {PARK_Z}">'
                         f'<geom name="{box_body(j)}" type="box" size="{s_max / 2} {s_max / 2} {h_obs}" '
                         f'material="obstacle"/></body>')
        obstacles.append(f'    <body name="{cyl_body(j)}" mocap="true" pos="0 0 {PARK_Z}">'
                         f'<geom name="{cyl_body(j)}" type="cylinder" size="{s_max / 2} {h_obs}" '
                         f'material="obstacle"/></body>')

    site_z = float(task["ray_height_m"]) - float(task["wheel_radius_m"])
    sites, sensors = [], []
    for i, th in enumerate(ray_angles_rad(task)):
        sites.append(f'      <site name="{ray_site(i)}" pos="0 0 {site_z}" '
                     f'zaxis="{math.cos(th):.12f} {math.sin(th):.12f} 0" size="0.005"/>')
        sensors.append(f'    <rangefinder name="{ray_site(i)}" site="{ray_site(i)}"/>')

    template = Template(resources.files("learned_opponents.env").joinpath("model.xml").read_text())
    values = {key: task[key] for key in ("physics_dt_s", "wheel_radius_m", "robot_radius_m",
                                         "chassis_mass_kg", "wheel_mass_kg", "wheel_friction",
                                         "motor_kv", "motor_torque_limit_nm", "max_wheel_speed")}
    return template.substitute(
        values,
        floor_half=half + 1.0,
        half_track=float(task["track_width_m"]) / 2.0,
        walls="\n".join(walls),
        obstacles="\n".join(obstacles),
        ray_sites="\n".join(sites),
        rangefinders="\n".join(sensors),
    )
