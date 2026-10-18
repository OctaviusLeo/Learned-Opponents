"""Parity test: Python policy vs exported ONNX model (task 14.1).

    python -m learned_opponents.export.parity                 # all six models, full check
    python -m learned_opponents.export.parity --fixtures-only # saved actions vs ONNX (no torch; CI)
    python -m learned_opponents.export.parity --mismatch-demo # a perturbed model must fail

For each model the saved parity observations go through the Python policy
(rsl-rl actor, deterministic mean, clipped; the framework's own inference)
and through ONNX Runtime on the CPU. The test passes only when every action
agrees within ``TOLERANCE`` (max absolute difference); on failure it names the
observation index with the largest disagreement and its magnitude.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOLERANCE = 1e-4
REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "results" / "models"
PARITY = REPO / "results" / "parity"


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    n: int
    worst_index: int
    max_abs_diff: float
    tolerance: float

    def report(self, name: str) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{verdict} {name}: {self.n} observations, max |action diff| {self.max_abs_diff:.3e} "
                f"at observation {self.worst_index} (tolerance {self.tolerance:.0e})")


def compare_actions(reference: np.ndarray, candidate: np.ndarray, tolerance: float = TOLERANCE) -> ParityResult:
    """Element-wise comparison of two ``(n, 2)`` action arrays."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 2:
        raise ValueError(f"action arrays must have the same (n, 2) shape, got {ref.shape} and {cand.shape}")
    diff = np.abs(ref - cand).max(axis=1)
    diff = np.where(np.isnan(diff), np.inf, diff)
    worst = int(np.argmax(diff))
    return ParityResult(bool(np.all(diff <= tolerance)), int(ref.shape[0]), worst, float(diff[worst]), tolerance)


def onnx_actions(model_path: Path, obs: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    batch = inp.shape[0]
    if batch == 1:
        return np.concatenate([sess.run(["action"], {inp.name: o[None, :]})[0] for o in obs.astype(np.float32)])
    return sess.run(["action"], {inp.name: obs.astype(np.float32)})[0]


def python_actions(variant: str, seed: int, obs: np.ndarray) -> np.ndarray:
    import torch

    from learned_opponents.train.policy import deterministic_action, load_actor

    actor, _ = load_actor(REPO / "runs" / variant / f"seed{seed}", device="cpu")
    return deterministic_action(actor, torch.from_numpy(obs.astype(np.float32))).numpy()


def model_names() -> list[str]:
    return sorted(p.stem for p in MODELS.glob("*_seed*.onnx"))


def split_name(name: str) -> tuple[str, int]:
    variant, seed = name.rsplit("_seed", 1)
    return variant, int(seed)


def check_model(name: str, fixtures_only: bool) -> list[ParityResult]:
    obs = np.load(PARITY / name / "observations.npy")
    saved = np.load(PARITY / name / "actions.npy")
    onnx_out = onnx_actions(MODELS / f"{name}.onnx", obs)
    results = [compare_actions(saved, onnx_out)]
    print(results[-1].report(f"{name} (saved Python actions vs ONNX Runtime)"))
    if not fixtures_only:
        live = python_actions(*split_name(name), obs)
        results.append(compare_actions(live, onnx_out))
        print(results[-1].report(f"{name} (Python policy vs ONNX Runtime)"))
    return results


def mismatch_demo() -> int:
    """Perturb one weight of a copy of the first model; parity must fail and name the worst observation."""
    import tempfile

    import onnx
    from onnx import numpy_helper

    name = model_names()[0]
    model = onnx.load(str(MODELS / f"{name}.onnx"))
    init = max(model.graph.initializer, key=lambda t: np.prod(t.dims))
    w = numpy_helper.to_array(init).copy()
    w.flat[0] += 0.5
    init.CopyFrom(numpy_helper.from_array(w, init.name))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mismatched.onnx"
        onnx.save(model, str(path))
        obs = np.load(PARITY / name / "observations.npy")
        res = compare_actions(np.load(PARITY / name / "actions.npy"), onnx_actions(path, obs))
    print(res.report(f"{name} with one perturbed weight (expected to FAIL)"))
    return 0 if not res.passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learned_opponents.export.parity")
    ap.add_argument("--fixtures-only", action="store_true", help="saved actions vs ONNX only (no torch)")
    ap.add_argument("--mismatch-demo", action="store_true")
    args = ap.parse_args(argv)
    if args.mismatch_demo:
        return mismatch_demo()
    names = model_names()
    if not names:
        print("no exported models under results/models", file=sys.stderr)
        return 1
    results = [r for n in names for r in check_model(n, args.fixtures_only)]
    ok = all(r.passed for r in results)
    print(f"parity: {'PASS' if ok else 'FAIL'} ({len(names)} models, tolerance {TOLERANCE:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
