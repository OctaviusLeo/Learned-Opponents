"""ONNX export, parity fixtures, and the evaluation rollout on a tiny trained run (GPU, local)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mjlab")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "baseline_seed5"
    subprocess.run([sys.executable, "-m", "learned_opponents.train", "--variant", "baseline", "--seed", "5",
                    "--out", str(out), "--set", "train.training_budget_steps=196608"], check=True,
                   capture_output=True, timeout=900)
    return out


def test_export_and_fixtures_agree_with_the_python_policy(tiny_run, tmp_path):
    import onnx

    from learned_opponents.config import load_task
    from learned_opponents.export.parity import compare_actions, onnx_actions
    from learned_opponents.export.to_onnx import N_ROLLOUT, N_UNIFORM, export, make_fixtures
    from learned_opponents.spec.obs_action_spec import observation_bounds

    model = tmp_path / "m.onnx"
    info = export(tiny_run, model, "c" * 40)
    assert info["opset"] == 17 and info["ir_version"] <= 10 and info["obs_dim"] == 29
    graph = onnx.load(str(model)).graph
    assert [i.name for i in graph.input] == ["obs"] and [o.name for o in graph.output] == ["action"]
    fx = make_fixtures("baseline", 5, tiny_run, tmp_path / "fixtures")
    assert fx == {"observations": N_ROLLOUT + N_UNIFORM, "from_rollouts": N_ROLLOUT, "uniform": N_UNIFORM}
    obs = np.load(tmp_path / "fixtures" / "observations.npy")
    act = np.load(tmp_path / "fixtures" / "actions.npy")
    low, high = observation_bounds(load_task())
    assert obs.shape == (1000, 29) and act.shape == (1000, 2) and obs.dtype == np.float32
    assert np.all(obs >= low - 1e-6) and np.all(obs <= high + 1e-6)
    assert compare_actions(act, onnx_actions(model, obs)).passed


def test_rollout_records_one_outcome_per_layout(tiny_run, task):
    from learned_opponents.env import rules
    from learned_opponents.eval.rollout import eval_layouts, make_eval_env, nominal_params, run_episodes
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.config import load_yaml
    from learned_opponents.train.policy import load_actor

    n = 64
    env = make_eval_env(task, n)
    actor, _ = load_actor(tiny_run, device="cuda:0")
    params = nominal_params(parse_ranges(load_yaml("domain_rand.yaml")))
    eps, seen = run_episodes(env, actor, eval_layouts(task, 7, n), params, noise_seed=7, record_obs=True)
    assert eps.n == n and set(np.unique(eps.reason)) <= {rules.SUCCESS, rules.COLLISION, rules.TIMEOUT}
    assert eps.count(rules.SUCCESS) + eps.count(rules.COLLISION) + eps.count(rules.TIMEOUT) == n
    assert np.all((eps.steps >= 1) & (eps.steps <= task["episode_limit"]))
    np.testing.assert_allclose(eps.time_to_goal_s(), eps.steps[eps.reason == rules.SUCCESS] * task["control_dt_s"])
    assert seen.shape[1] == 29 and len(seen) >= n
    d = eps.as_dict()
    assert d["success"] == len(d["success_steps"])
