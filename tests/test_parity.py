"""Parity between the Python policy and the exported ONNX models (tasks 14.1, 14.2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from learned_opponents.export import parity

MODELS = parity.model_names()
needs_models = pytest.mark.skipif(not MODELS, reason="no exported models under results/models yet")


# Feature: learned-opponents, Property 13: Parity failure reports the worst disagreement
# (when at least one action exceeds the tolerance, the comparison fails and names the index and
# magnitude of the largest disagreement).
@settings(max_examples=300, deadline=None)
@given(data=st.data(), n=st.integers(1, 200))
def test_property_13_failure_reports_worst_disagreement(data, n):
    ref = data.draw(hnp.arrays(np.float32, (n, 2), elements=st.floats(-1, 1, width=32)))
    small = 2.0 ** -15                      # 3.05e-5, exactly representable in float32
    noise = data.draw(hnp.arrays(np.float32, (n, 2), elements=st.floats(-small, small, width=32)))
    cand = (ref + noise).astype(np.float32)
    within = parity.compare_actions(ref, cand)
    assert within.passed and within.max_abs_diff <= parity.TOLERANCE
    i = data.draw(st.integers(0, n - 1))
    k = data.draw(st.integers(0, 1))
    bump = data.draw(st.floats(2e-4, 1.0))
    cand[i, k] = ref[i, k] + np.float32(bump)
    res = parity.compare_actions(ref, cand)
    diffs = np.abs(ref.astype(np.float64) - cand.astype(np.float64)).max(axis=1)
    assert not res.passed
    assert res.worst_index == int(np.argmax(diffs)) and res.max_abs_diff == pytest.approx(float(diffs.max()))
    assert res.max_abs_diff > parity.TOLERANCE
    assert f"observation {res.worst_index}" in res.report("m") and res.report("m").startswith("FAIL")


def test_nan_actions_fail_parity():
    ref = np.zeros((3, 2), np.float32)
    cand = ref.copy()
    cand[1, 0] = np.nan
    res = parity.compare_actions(ref, cand)
    assert not res.passed and res.worst_index == 1


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        parity.compare_actions(np.zeros((3, 2)), np.zeros((2, 2)))


# Feature: learned-opponents, Property 12: ONNX parity holds within tolerance
# (for saved parity observations, ONNX Runtime reproduces the Python policy's saved actions).
@needs_models
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_property_12_onnx_matches_saved_python_actions(data):
    name = data.draw(st.sampled_from(MODELS))
    obs = np.load(parity.PARITY / name / "observations.npy")
    saved = np.load(parity.PARITY / name / "actions.npy")
    idx = data.draw(st.lists(st.integers(0, len(obs) - 1), min_size=1, max_size=16, unique=True))
    out = parity.onnx_actions(parity.MODELS / f"{name}.onnx", obs[idx])
    res = parity.compare_actions(saved[idx], out)
    assert res.passed, res.report(name)


@needs_models
@pytest.mark.gpu
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_property_12_python_policy_vs_onnx_on_generated_observations(data):
    """The live Python policy (torch) against ONNX Runtime on observations drawn inside the spec ranges."""
    pytest.importorskip("torch")
    from learned_opponents.config import load_task
    from learned_opponents.spec.obs_action_spec import observation_bounds

    name = data.draw(st.sampled_from(MODELS))
    low, high = observation_bounds(load_task())
    obs = data.draw(hnp.arrays(np.float32, (8, low.size), elements=st.floats(0, 1, width=32)))
    obs = (low + obs * (high - low)).astype(np.float32)
    py = parity.python_actions(*parity.split_name(name), obs)
    res = parity.compare_actions(py, parity.onnx_actions(parity.MODELS / f"{name}.onnx", obs))
    assert res.passed, res.report(name)
