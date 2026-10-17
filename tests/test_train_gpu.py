"""A launched training run writes complete Run_Artifacts (task 8.2). GPU, run locally."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

pytest.importorskip("mjlab")

pytestmark = pytest.mark.gpu


def test_one_command_run_writes_run_artifacts(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "learned_opponents.train", "--variant", "baseline", "--seed", "7",
           "--out", str(out), "--set", "train.training_budget_steps=196608"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-2000:]
    for name in ("config.yaml", "seed", "commit", "checkpoint.pt", "train_log.csv", "summary.json"):
        assert (out / name).is_file(), name
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["seed"] == 7 and (out / "seed").read_text().strip() == "7"
    assert cfg["variant"] == "baseline" and cfg["iterations"] == 2
    assert cfg["commit"] == (out / "commit").read_text().strip() and len(cfg["commit"]) == 40
    assert {"task", "train", "domain_rand"} <= set(cfg)
    assert cfg["framework_versions"]["mjlab"] == "1.6.0"
    # Requirement 0.10 / 4.7 only bind when MJX is the training framework; the M0
    # smoke test chose mjlab, and the run records that the setting does not apply.
    assert cfg["framework"].startswith("mjlab")
    assert cfg["jax_default_matmul_precision"].startswith("not applicable")
    # Refuses to overwrite an existing checkpoint without --force.
    again = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert again.returncode == 2


def test_checkpoint_loads_into_a_deterministic_policy(tmp_path):
    import torch

    from learned_opponents.train.policy import deterministic_action, load_actor

    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "learned_opponents.train", "--variant", "randomized", "--seed", "3",
                    "--out", str(out), "--set", "train.training_budget_steps=98304"], check=True,
                   capture_output=True, timeout=900)
    actor, cfg = load_actor(out)
    obs = torch.rand(64, cfg["task"]["n_rays"] + 5)
    a1, a2 = deterministic_action(actor, obs), deterministic_action(actor, obs)
    assert a1.shape == (64, 2) and torch.equal(a1, a2)
    assert bool(((a1 >= -1) & (a1 <= 1)).all())
