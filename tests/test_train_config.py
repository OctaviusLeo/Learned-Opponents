"""Training configuration helpers (GPU-free)."""

from __future__ import annotations

import pytest

from learned_opponents import config as cfglib
from learned_opponents.train import train as tr


def test_override_parsing_nests_and_types_values():
    ov = tr.parse_overrides(["train.ppo.learning_rate=3e-4", "task.reward.w_time=0.01", "train.bank_size=1024"])
    assert ov["train"] == {"ppo": {"learning_rate": 3e-4}, "bank_size": 1024}
    assert ov["task"] == {"reward": {"w_time": 0.01}}
    with pytest.raises(ValueError):
        tr.parse_overrides(["ppo.learning_rate=1"])
    with pytest.raises(ValueError):
        tr.parse_overrides(["train.ppo.learning_rate"])


def test_resolve_merges_overrides_and_records_them():
    r = tr.resolve("baseline", 100, ["train.ppo.entropy_coef=0.01"], commit="abc")
    assert r["train"]["ppo"]["entropy_coef"] == 0.01
    assert r["train"]["ppo"]["gamma"] == cfglib.load_yaml("train.yaml")["ppo"]["gamma"]
    assert r["overrides"] == ["train.ppo.entropy_coef=0.01"] and r["commit"] == "abc"
    assert r["seed"] == 100 and r["variant"] == "baseline"
    with pytest.raises(ValueError):
        tr.resolve("medium", 0, commit="x")


def test_iterations_round_the_budget_up_to_whole_iterations():
    train = cfglib.load_yaml("train.yaml")
    per = train["num_parallel_envs"] * train["ppo"]["num_steps_per_env"]
    train["training_budget_steps"] = per * 10 + 1
    assert tr.iterations_for(train) == 11


def test_committed_train_config_is_valid_and_has_three_seeds_each():
    train = cfglib.load_yaml("train.yaml")
    tr.validate_train(train)
    assert len(train["seeds"]["baseline"]) >= 3 and len(train["seeds"]["randomized"]) >= 3
    assert train["seeds"]["pilot"] == [100]
    assert not set(train["seeds"]["pilot"]) & set(train["seeds"]["baseline"])


@pytest.mark.parametrize("key,value", [("num_parallel_envs", 1024), ("training_budget_steps", 0),
                                       ("bank_size", -1)])
def test_invalid_train_values_are_rejected(key, value):
    train = cfglib.load_yaml("train.yaml")
    train[key] = value
    with pytest.raises(cfglib.ConfigError, match=key):
        tr.validate_train(train)


def test_pilot_budget_rule():
    from learned_opponents.train.pilot import choose_budget

    pts = [{"env_steps": s, "elapsed_min": s / 1e7, "success_rate": r}
           for s, r in [(1e7, 0.70), (2e7, 0.85), (3e7, 0.905), (4e7, 0.91), (9e8, 0.99)]]
    c = choose_budget(pts, target=0.8, max_minutes=60)          # the 9e8 point is over the time cap
    assert c["chosen_env_steps"] == 3e7 and c["best_success_rate"] == 0.91 and c["meets_target"]
    low = choose_budget(pts[:2], target=0.9, max_minutes=60)     # target not reached: best is kept
    assert low["chosen_env_steps"] == 2e7 and not low["meets_target"]
    with pytest.raises(ValueError):
        choose_budget(pts[4:], target=0.8, max_minutes=60)
