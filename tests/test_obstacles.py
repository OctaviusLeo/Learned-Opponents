"""Layout sampling, footprint geometry, and the MJCF builder (GPU-free)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.config import load_task
from learned_opponents.env import obstacles as ob
from learned_opponents.env.model import build_mjcf
from learned_opponents.randomization.domain_rand import make_rng

TASK = load_task()
HALF = TASK["arena_size_m"] / 2
CLEAR = TASK["robot_radius_m"] + TASK["start_goal_clearance_m"]


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_sampled_layouts_satisfy_every_rule(seed):
    lay = ob.sample_layouts(make_rng(seed, "layouts"), 64, TASK)
    n_active = (lay.kind != ob.INACTIVE).sum(1)
    assert np.all((n_active >= TASK["n_obstacles"][0]) & (n_active <= TASK["n_obstacles"][1]))
    act = lay.kind != ob.INACTIVE
    s = lay.size[act]
    assert np.all((s >= TASK["obstacle_size_m"][0]) & (s <= TASK["obstacle_size_m"][1]))
    # Every obstacle lies fully inside the arena.
    assert np.all(np.abs(lay.center[act]) + s[:, None] / 2 <= HALF + 1e-9)
    for pts in (lay.start[:, :2], lay.goal):
        assert np.all(np.abs(pts) <= HALF - CLEAR + 1e-9)
        d = ob.obstacle_signed_distance_xp(np, pts, lay.kind, lay.center, lay.size).min(-1)
        assert np.all(d >= CLEAR - 1e-9)
    assert np.all(np.hypot(*(lay.start[:, :2] - lay.goal).T) >= TASK["min_start_goal_distance_m"])
    assert np.all((lay.start[:, 2] >= -math.pi) & (lay.start[:, 2] < math.pi))


def test_layouts_reproduce_under_seed():
    a = ob.sample_layouts(make_rng(7, "layouts"), 200, TASK)
    b = ob.sample_layouts(make_rng(7, "layouts"), 200, TASK)
    for f in ("kind", "center", "size", "start", "goal"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_signed_distance_examples():
    kind = np.array([[ob.BOX, ob.CYLINDER, ob.INACTIVE]])
    center = np.array([[[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]]])
    size = np.array([[1.0, 1.0, 0.0]])
    pts = np.array([[1.5, 0.0], [1.5, 1.5], [0.0, 0.0], [3.0, 1.5]])
    d = ob.obstacle_signed_distance_xp(np, pts, np.repeat(kind, 4, 0), np.repeat(center, 4, 0),
                                       np.repeat(size, 4, 0))
    np.testing.assert_allclose(d[:, 0], [1.0, math.hypot(1.0, 1.0), -0.5, math.hypot(2.5, 1.0)])
    np.testing.assert_allclose(d[:, 1], [1.0, math.hypot(1.5, 1.5) - 0.5, 2.5, 1.0])
    assert np.all(d[:, 2] > 1e5)


def test_collides_with_obstacles_and_walls():
    kind = np.array([[ob.BOX]] * 5)
    center = np.zeros((5, 1, 2))
    size = np.ones((5, 1))
    pts = np.array([[0.64, 0.0], [0.66, 0.0], [3.9, 0.0], [3.8, -3.8], [3.0, -3.86]])
    hit = ob.collides_xp(np, pts, 0.15, kind, center, size, 8.0)
    assert hit.tolist() == [True, False, True, False, True]   # 0.2 m and 0.14 m from a wall


def test_json_round_trip():
    lay = ob.sample_layouts(make_rng(3, "layouts"), 10, TASK)
    back = ob.layouts_from_json(ob.layouts_to_json(lay), TASK["n_obstacles"][1])
    act = lay.kind != ob.INACTIVE
    for i in range(10):
        n = act[i].sum()
        np.testing.assert_array_equal(np.sort(back.kind[i, :n]), np.sort(lay.kind[i][act[i]]))
    np.testing.assert_allclose(back.start, lay.start)
    np.testing.assert_allclose(back.goal, lay.goal)
    with pytest.raises(ValueError):
        ob.layouts_from_json(ob.layouts_to_json(lay), 2)


def test_mjcf_has_rays_slots_and_contacts():
    root = ET.fromstring(build_mjcf(TASK))
    assert len(root.findall("./sensor/rangefinder")) == TASK["n_rays"]
    mocap = [b for b in root.iter("body") if b.get("mocap") == "true"]
    assert len(mocap) == 2 * TASK["n_obstacles"][1]
    contacting = {g.get("name") for g in root.iter("geom") if g.get("contype") == "1"}
    assert contacting == {"floor", "caster_f", "caster_b", "wheel_l", "wheel_r"}
    sites = [s for s in root.iter("site") if s.get("name", "").startswith("ray_")]
    z = [float(s.get("pos").split()[2]) for s in sites]
    assert z == [pytest.approx(TASK["ray_height_m"] - TASK["wheel_radius_m"])] * TASK["n_rays"]
    # Ray 12 points straight ahead (+x), ray 18 to the left (+y).
    ahead = np.array([float(v) for v in sites[12].get("zaxis").split()])
    left = np.array([float(v) for v in sites[18].get("zaxis").split()])
    np.testing.assert_allclose(ahead, [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(left, [0, 1, 0], atol=1e-9)
