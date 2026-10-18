"""Exported ONNX models conform to the Obs_Action_Spec I/O contract (task 13.2; runs in CI)."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from learned_opponents.config import load_task
from learned_opponents.export import parity

MODELS = parity.model_names()
OBS_DIM = load_task()["n_rays"] + 5


@pytest.mark.skipif(not MODELS, reason="no exported models under results/models yet")
@pytest.mark.parametrize("name", MODELS)
def test_model_io_matches_spec(name):
    path = parity.MODELS / f"{name}.onnx"
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    (inp,), (out,) = model.graph.input, model.graph.output
    assert inp.name == "obs" and out.name == "action"
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, OBS_DIM]
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == [1, 2]
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT == out.type.tensor_type.elem_type
    assert {o.domain or "ai.onnx" for o in model.opset_import} == {"ai.onnx"}
    meta = {p.key: p.value for p in model.metadata_props}
    variant, seed = parity.split_name(name)
    assert meta["variant"] == variant and meta["seed"] == str(seed) and len(meta["train_commit"]) == 40
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    a = sess.run(["action"], {"obs": np.zeros((1, OBS_DIM), np.float32)})[0]
    assert a.shape == (1, 2) and np.all(np.abs(a) <= 1.0)


def test_six_models_once_exported():
    if not MODELS:
        pytest.skip("no exported models yet")
    assert sorted(MODELS) == sorted(f"{v}_seed{s}" for v in ("baseline", "randomized") for s in (0, 1, 2))
