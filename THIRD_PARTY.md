# Third-party dependencies

Licenses come from each package's installed metadata on this machine
(`~/.venvs/learned-opponents/bin/python scripts/license_report.py`). SPDX
identifiers are used where the metadata gives one. Nothing from these
packages is vendored into the repository; they are installed from PyPI.

## Python, project environment (`~/.venvs/learned-opponents`)

| Name | Version | License identifier |
| --- | --- | --- |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| PyYAML | 6.0.3 | MIT |
| onnx | 1.23.2 | Apache-2.0 |
| onnxruntime | 1.31.0 | MIT |
| matplotlib | 3.11.2 | PSF-based matplotlib license (metadata: "Python Software Foundation License") |
| pytest | 9.1.1 | MIT |
| hypothesis | 6.170.1 | MPL-2.0 |
| mjlab | 1.6.0 | Apache-2.0 |
| mujoco | 3.11.0 | Apache-2.0 |
| mujoco-warp | 3.11.0 | Apache-2.0 |
| warp-lang | 1.18.0 | Apache-2.0 |
| torch | 2.14.1 (+cu130 build) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| rsl-rl-lib | 5.4.2 | BSD-3-Clause |
| tensordict | 0.15.1 | MIT |
| tensorboard | 2.21.0 | Apache-2.0 |

Transitive pins for the project environment are in `requirements-gpu.lock`.

## Python, M0 smoke-test environments only

Used only by `scripts/smoke_test.sh`; frozen in `docs/evidence/freeze_lo-mjx.txt`
and `docs/evidence/freeze_lo-mjlab.txt`.

| Name | Version | License identifier |
| --- | --- | --- |
| playground (MuJoCo Playground) | 0.2.0 | Apache-2.0 |
| mujoco / mujoco-mjx | 3.16.0 | Apache-2.0 |
| jax / jaxlib / jax-cuda12-plugin | 0.11.2 | Apache-2.0 |
| brax | 0.14.2 | Apache-2.0 |
| flax | 0.12.10 | Apache-2.0 |

## Engine and tools (Windows)

| Name | Version | License identifier |
| --- | --- | --- |
| Unreal Engine (incl. NNE / NNERuntimeORT plugins) | 5.8.3 | Unreal Engine EULA (proprietary; engine content is not redistributed here) |
| FFmpeg (GIF capture only, not distributed) | see `docs/ENVIRONMENT.md` | LGPL-2.1-or-later / GPL (build-dependent) |
