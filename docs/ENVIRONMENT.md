# Environment & Versions

Every reported number in this repo comes from a repository command with a
fixed seed, recorded with its command, seed, and full commit hash. This file
records the exact tool/library/API versions and a **verifiable artifact** for
each (a package query, a docs reference, or a passing minimal test).

Rows marked `TBD` must be proven on the **owner's GPU machine**
(Ubuntu 24.04 + NVIDIA RTX 3080) during the M0 smoke test (task 2, `[ME]`).
Do not fill them with typed or estimated values — only with a value backed by
a verifiable artifact captured on that machine.

Dev box used for the pure-Python stack: **Windows, Python 3.11.5**.

| Tool / Library / API | Exact version | Verifiable artifact |
|---|---|---|
| OS (GPU machine) | TBD (Ubuntu 24.04) | `lsb_release -a` capture on target machine |
| NVIDIA driver | TBD | `nvidia-smi` capture (see block below) |
| CUDA | TBD | `nvcc --version` / `nvidia-smi` capture on target |
| Python (dev box) | 3.11.5 | `python --version` -> `Python 3.11.5` |
| MuJoCo | TBD | `python -c "import mujoco; print(mujoco.__version__)"` on target |
| MuJoCo Playground / MJX | TBD | package query on target + minimal step test |
| JAX | TBD | `python -c "import jax; print(jax.__version__)"` on target |
| jaxlib (CUDA) | TBD | `python -c "import jaxlib; print(jaxlib.__version__)"` on target |
| mjlab / MuJoCo Warp | TBD | package query on target + minimal step test |
| numpy | 2.4.6 | `python -c "import numpy; print(numpy.__version__)"` |
| onnx | 1.23.2 | `python -c "import onnx; print(onnx.__version__)"` |
| onnxruntime | 1.31.0 | `python -c "import onnxruntime; print(onnxruntime.__version__)"` |
| pytest | 9.1.1 | `pytest --version` |
| hypothesis | 6.169.0 | `python -c "import hypothesis; print(hypothesis.__version__)"` |
| pyyaml | 6.0.3 | `python -c "import yaml; print(yaml.__version__)"` |
| matplotlib | 3.11.2 | `python -c "import matplotlib; print(matplotlib.__version__)"` |
| Unreal Engine | TBD | editor version string + evidence on target |
| In-engine inference runtime | TBD | runtime name + version + passing in-engine test (Godot fallback if blocked) |

> Licenses for every pinned version are tracked in `THIRD_PARTY.md` and must be
> re-verified against the **exact** pinned version.

## RTX 30-series precision requirement

`JAX_DEFAULT_MATMUL_PRECISION=highest`

This environment variable is **required** on RTX 30-series GPUs when running
under MJX (per the MuJoCo Playground README — Requirements 0.10 and 1.5). It is
recorded here, set during training/eval on the target machine, and its value is
captured during the M0 smoke test as part of the evidence below.

## M0 smoke-test evidence (TBD — owner [ME] task 2)

Captured on the owner's Ubuntu 24.04 + RTX 3080 machine. Until task 2 is run,
every item below is a placeholder.

```text
# nvidia-smi (TBD — paste full capture here)
```

- Chosen framework: TBD (MuJoCo Playground / MJX  or  mjlab / MuJoCo Warp)
- Parallel-env count (>= 1024): TBD
- Step throughput (steps/s, with command + seed + commit): TBD
- Range_Rays returned per environment: TBD (confirm shape `(num_envs, N_rays)`)
- `JAX_DEFAULT_MATMUL_PRECISION` value during smoke test: TBD (expected `highest`)
- If both frameworks fail: record each framework's first-failing condition here (TBD)
