"""Tests for the domain-randomization sampler (tasks 7.1, 7.2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.config import load_yaml
from learned_opponents.randomization import domain_rand as dr

DEFAULT = dr.parse_ranges(load_yaml("domain_rand.yaml"))


@st.composite
def range_configs(draw):
    cfg = {}
    for name in dr.PARAMS:
        if name in dr.INTEGER_PARAMS:
            lo = draw(st.integers(0, 5))
            hi = draw(st.integers(lo, lo + 6))
            nom = draw(st.integers(lo, hi))
        else:
            lo = draw(st.floats(0.01 if name.endswith("_scale") else 0.0, 3.0))
            hi = draw(st.floats(lo, lo + 3.0))
            nom = draw(st.floats(lo, hi))
        cfg[name] = {"nominal": nom, "range": [lo, hi]}
    return dr.parse_ranges(cfg)


# Feature: learned-opponents, Property 7: Domain randomization samples within configured ranges
# (and the baseline variant leaves every parameter at nominal).
@settings(max_examples=200, deadline=None)
@given(ranges=range_configs(), seed=st.integers(0, 2**32 - 1), n=st.integers(1, 64))
def test_property_7_samples_within_ranges(ranges, seed, n):
    draws = dr.sample_params(dr.make_rng(seed, "domain_rand"), n, ranges, "randomized")
    base = dr.sample_params(dr.make_rng(seed, "domain_rand"), n, ranges, "baseline")
    assert set(draws) == set(dr.PARAMS) == set(base)
    for name, r in ranges.items():
        assert draws[name].shape == (n,)
        assert r.contains(draws[name]).all()
        if r.integer:
            assert draws[name].dtype == np.int64
        assert np.all(base[name] == r.nominal)


# Feature: learned-opponents, Property 8: Randomization is reproducible under a fixed seed.
@settings(max_examples=200, deadline=None)
@given(ranges=range_configs(), seed=st.integers(0, 2**32 - 1), n=st.integers(1, 64))
def test_property_8_reproducible_under_seed(ranges, seed, n):
    a = dr.sample_params(dr.make_rng(seed, "domain_rand"), n, ranges, "randomized")
    b = dr.sample_params(dr.make_rng(seed, "domain_rand"), n, ranges, "randomized")
    for name in dr.PARAMS:
        np.testing.assert_array_equal(a[name], b[name])


def test_default_ranges_match_design():
    assert (DEFAULT["mass_scale"].low, DEFAULT["mass_scale"].high) == (0.8, 1.2)
    assert (DEFAULT["friction_scale"].low, DEFAULT["friction_scale"].high) == (0.6, 1.2)
    assert (DEFAULT["motor_strength_scale"].low, DEFAULT["motor_strength_scale"].high) == (0.8, 1.2)
    assert (DEFAULT["sensor_noise_sigma_m"].low, DEFAULT["sensor_noise_sigma_m"].high) == (0.0, 0.05)
    for k in ("action_latency_steps", "observation_latency_steps"):
        assert (DEFAULT[k].low, DEFAULT[k].high, DEFAULT[k].nominal) == (0, 2, 0)


def test_integer_latencies_cover_whole_range():
    d = dr.sample_params(dr.make_rng(1, "domain_rand"), 3000, DEFAULT, "randomized")
    assert set(np.unique(d["action_latency_steps"])) == {0, 1, 2}


def test_different_streams_and_seeds_differ():
    a = dr.sample_params(dr.make_rng(1, "domain_rand"), 50, DEFAULT, "randomized")
    b = dr.sample_params(dr.make_rng(2, "domain_rand"), 50, DEFAULT, "randomized")
    c = dr.sample_params(dr.make_rng(1, "other"), 50, DEFAULT, "randomized")
    assert not np.array_equal(a["mass_scale"], b["mass_scale"])
    assert not np.array_equal(a["mass_scale"], c["mass_scale"])


@pytest.mark.parametrize("bad,match", [
    ({"mass_scale": {"nominal": 1.5, "range": [0.8, 1.2]}}, "nominal"),
    ({"action_latency_steps": {"nominal": 0, "range": [0, 1.5]}}, "whole numbers"),
    ({"friction_scale": {"nominal": 0.0, "range": [0.0, 1.0]}}, "> 0"),
    ({"bogus": {"nominal": 0, "range": [0, 1]}}, "unknown"),
])
def test_invalid_configs_are_rejected(bad, match):
    cfg = load_yaml("domain_rand.yaml")
    cfg.update(bad)
    with pytest.raises(ValueError, match=match):
        dr.parse_ranges(cfg)


def test_missing_parameter_is_rejected():
    cfg = load_yaml("domain_rand.yaml")
    del cfg["sensor_noise_sigma_m"]
    with pytest.raises(ValueError, match="sensor_noise_sigma_m"):
        dr.parse_ranges(cfg)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        dr.sample_params(dr.make_rng(0, "x"), 1, DEFAULT, "medium")
