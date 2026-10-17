"""Unit tests for config loading and validation."""

from __future__ import annotations

import pytest

from learned_opponents import config


def test_task_yaml_loads_and_validates():
    t = config.load_task()
    assert t["n_rays"] == 24 and config.substeps(t) == 5


@pytest.mark.parametrize("key,value,match", [
    ("n_rays", 1, "n_rays"),
    ("max_range_m", -1.0, "max_range_m"),
    ("n_obstacles", [8, 4], "n_obstacles"),
    ("obstacle_kinds", ["sphere"], "obstacle_kinds"),
    ("physics_dt_s", 0.03, "control_dt_s"),
    ("nominal_success_target", 1.5, "nominal_success_target"),
    ("episode_limit", 0, "episode_limit"),
])
def test_invalid_task_values_are_named(key, value, match):
    t = config.load_task()
    t[key] = value
    with pytest.raises(config.ConfigError, match=match):
        config.validate_task(t)


def test_missing_parameter_is_named():
    t = config.load_task()
    del t["arrival_tolerance_m"]
    with pytest.raises(config.ConfigError, match="arrival_tolerance_m"):
        config.validate_task(t)


def test_deep_merge_overrides_nested_values_only():
    base = {"a": 1, "reward": {"w_progress": 1.0, "w_success": 10.0}}
    out = config.deep_merge(base, {"reward": {"w_success": 5.0}})
    assert out == {"a": 1, "reward": {"w_progress": 1.0, "w_success": 5.0}}
    assert base["reward"]["w_success"] == 10.0
