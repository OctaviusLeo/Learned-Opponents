"""Obstacle layouts and footprint geometry (pure functions, numpy only).

A layout is one episode's arena content: up to ``max_obstacles`` slots, each
inactive, an axis-aligned square box, or an upright cylinder, plus the robot
start pose ``(x, y, yaw)`` and the goal ``(x, y)``. Layouts are sampled in
batches from an explicit generator, so a seed reproduces them exactly. The
same sampler writes the shared transfer-check layouts, and the geometry
functions (written against an ``xp`` array module) are used by the GPU
environment for collision checks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

INACTIVE, BOX, CYLINDER = 0, 1, 2
KIND_NAMES = {BOX: "box", CYLINDER: "cylinder"}
KIND_CODES = {v: k for k, v in KIND_NAMES.items()}


@dataclass
class Layouts:
    """A batch of ``n`` layouts with ``k`` obstacle slots each."""

    kind: np.ndarray     # (n, k) int8: INACTIVE, BOX, CYLINDER
    center: np.ndarray   # (n, k, 2) obstacle centres, metres
    size: np.ndarray     # (n, k) box side or cylinder diameter, metres (0 when inactive)
    start: np.ndarray    # (n, 3) robot x, y, yaw
    goal: np.ndarray     # (n, 2)

    def __len__(self) -> int:
        return int(self.kind.shape[0])

    def subset(self, idx) -> "Layouts":
        return Layouts(self.kind[idx], self.center[idx], self.size[idx], self.start[idx], self.goal[idx])


def obstacle_signed_distance_xp(xp, xy, kind, center, size):
    """Signed distance from points to each obstacle's footprint (negative inside).

    ``xy`` (..., 2); ``kind``/``size`` (..., k); ``center`` (..., k, 2). Inactive
    slots return a large positive distance.
    """
    p = xy[..., None, :] - center
    half = size * 0.5
    # Axis-aligned square: exact 2D signed distance.
    qx = xp.abs(p[..., 0]) - half
    qy = xp.abs(p[..., 1]) - half
    ox, oy = xp.maximum(qx, 0.0), xp.maximum(qy, 0.0)
    d_box = xp.sqrt(ox * ox + oy * oy) + xp.minimum(xp.maximum(qx, qy), 0.0)
    d_cyl = xp.sqrt(p[..., 0] * p[..., 0] + p[..., 1] * p[..., 1]) - half
    d = xp.where(kind == BOX, d_box, d_cyl)
    return xp.where(kind == INACTIVE, 1e6, d)


def wall_clearance_xp(xp, xy, arena_size_m: float):
    """Distance from points to the nearest arena wall (inside the square interior)."""
    half = arena_size_m / 2.0
    return xp.min(xp.stack([half - xy[..., 0], half + xy[..., 0], half - xy[..., 1], half + xy[..., 1]],
                           axis=-1), axis=-1)


def collides_xp(xp, xy, radius: float, kind, center, size, arena_size_m: float):
    """True where a robot footprint circle of ``radius`` at ``xy`` touches an obstacle or wall."""
    d_obs = xp.min(obstacle_signed_distance_xp(xp, xy, kind, center, size), axis=-1)
    return (d_obs < radius) | (wall_clearance_xp(xp, xy, arena_size_m) < radius)


def sample_layouts(rng: np.random.Generator, n: int, task: Mapping[str, Any],
                   max_tries: int = 200) -> Layouts:
    """Sample ``n`` valid layouts.

    Obstacle count is uniform in ``n_obstacles`` (inclusive), each obstacle's
    kind uniform over ``obstacle_kinds``, its size uniform in
    ``obstacle_size_m``, and its centre uniform where it fits inside the arena
    (obstacles may overlap each other). Start and goal are uniform in the
    arena with ``robot_radius_m + start_goal_clearance_m`` of clearance from
    walls and obstacles and at least ``min_start_goal_distance_m`` apart; the
    start yaw is uniform in [-pi, pi). Rows that fail are resampled
    (start/goal first, the whole layout every 20 tries).
    """
    lo_n, hi_n = (int(v) for v in task["n_obstacles"])
    lo_s, hi_s = (float(v) for v in task["obstacle_size_m"])
    kinds = np.array([KIND_CODES[k] for k in task["obstacle_kinds"]], dtype=np.int8)
    half = float(task["arena_size_m"]) / 2.0
    clear = float(task["robot_radius_m"]) + float(task["start_goal_clearance_m"])
    min_sep = float(task["min_start_goal_distance_m"])
    k = hi_n

    kind = np.zeros((n, k), np.int8)
    center = np.zeros((n, k, 2))
    size = np.zeros((n, k))
    start = np.zeros((n, 3))
    goal = np.zeros((n, 2))

    def draw_obstacles(rows: np.ndarray) -> None:
        m = rows.size
        count = rng.integers(lo_n, hi_n + 1, size=m)
        active = np.arange(k)[None, :] < count[:, None]
        kk = kinds[rng.integers(0, kinds.size, size=(m, k))]
        ss = rng.uniform(lo_s, hi_s, size=(m, k))
        lim = half - ss / 2.0
        cc = rng.uniform(-1.0, 1.0, size=(m, k, 2)) * lim[..., None]
        kind[rows] = np.where(active, kk, INACTIVE)
        size[rows] = np.where(active, ss, 0.0)
        center[rows] = np.where(active[..., None], cc, 0.0)

    def draw_endpoints(rows: np.ndarray) -> None:
        m = rows.size
        start[rows, :2] = rng.uniform(-half + clear, half - clear, size=(m, 2))
        start[rows, 2] = rng.uniform(-np.pi, np.pi, size=m)
        goal[rows] = rng.uniform(-half + clear, half - clear, size=(m, 2))

    def valid(rows: np.ndarray) -> np.ndarray:
        ds = obstacle_signed_distance_xp(np, start[rows, :2], kind[rows], center[rows], size[rows]).min(-1)
        dg = obstacle_signed_distance_xp(np, goal[rows], kind[rows], center[rows], size[rows]).min(-1)
        sep = np.hypot(*(start[rows, :2] - goal[rows]).T)
        return (ds >= clear) & (dg >= clear) & (sep >= min_sep)

    rows = np.arange(n)
    draw_obstacles(rows)
    draw_endpoints(rows)
    for attempt in range(1, max_tries + 1):
        bad = rows[~valid(rows)]
        if bad.size == 0:
            return Layouts(kind, center, size, start, goal)
        if attempt % 20 == 0:
            draw_obstacles(bad)
        draw_endpoints(bad)
        rows = bad
    raise RuntimeError(f"could not sample {rows.size} valid layouts in {max_tries} tries")


def layouts_to_json(layouts: Layouts) -> list[dict[str, Any]]:
    """Plain-JSON form used by results/transfer/episodes.json and the game."""
    out = []
    for i in range(len(layouts)):
        obstacles = [{"kind": KIND_NAMES[int(c)], "x": float(layouts.center[i, j, 0]),
                      "y": float(layouts.center[i, j, 1]), "size": float(layouts.size[i, j])}
                     for j, c in enumerate(layouts.kind[i]) if c != INACTIVE]
        out.append({"index": i, "start": {"x": float(layouts.start[i, 0]), "y": float(layouts.start[i, 1]),
                                          "yaw": float(layouts.start[i, 2])},
                    "goal": {"x": float(layouts.goal[i, 0]), "y": float(layouts.goal[i, 1])},
                    "obstacles": obstacles})
    return out


def layouts_from_json(episodes: list[Mapping[str, Any]], max_obstacles: int) -> Layouts:
    n = len(episodes)
    kind = np.zeros((n, max_obstacles), np.int8)
    center = np.zeros((n, max_obstacles, 2))
    size = np.zeros((n, max_obstacles))
    start = np.zeros((n, 3))
    goal = np.zeros((n, 2))
    for i, ep in enumerate(episodes):
        obs = ep["obstacles"]
        if len(obs) > max_obstacles:
            raise ValueError(f"episode {i} has {len(obs)} obstacles > {max_obstacles} slots")
        for j, o in enumerate(obs):
            kind[i, j] = KIND_CODES[o["kind"]]
            center[i, j] = (o["x"], o["y"])
            size[i, j] = o["size"]
        start[i] = (ep["start"]["x"], ep["start"]["y"], ep["start"]["yaw"])
        goal[i] = (ep["goal"]["x"], ep["goal"]["y"])
    return Layouts(kind, center, size, start, goal)


def save_layouts_json(path: str, layouts: Layouts, meta: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({**meta, "episodes": layouts_to_json(layouts)}, f, indent=1)
        f.write("\n")
