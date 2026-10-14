# Third-Party Dependencies

> Licenses must be **re-verified against the exact pinned versions** listed
> here. License identifiers use SPDX where a standard identifier exists.
> `TBD` rows are proven on the owner's target machine (M0, task 2 `[ME]`) once
> the GPU/engine stack is pinned.

| Name | Version | License identifier |
|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause |
| onnx | 1.23.2 | Apache-2.0 |
| onnxruntime | 1.31.0 | MIT |
| pytest | 9.1.1 | MIT |
| hypothesis | 6.169.0 | MPL-2.0 |
| PyYAML | 6.0.3 | MIT |
| matplotlib | 3.11.2 | matplotlib license (PSF-based, BSD-compatible) |
| MuJoCo | TBD | Apache-2.0 |
| JAX | TBD | Apache-2.0 |
| MuJoCo Playground / MJX | TBD | TBD (re-verify on pinned version) |
| mjlab / MuJoCo Warp | TBD | TBD (re-verify on pinned version) |
| Unreal Engine / inference runtime (Godot fallback) | TBD | TBD (re-verify on pinned version) |

Versions above match the pins in `pyproject.toml`. GPU/sim/engine versions are
proven on the owner's Ubuntu 24.04 + RTX 3080 machine and recorded in
`docs/ENVIRONMENT.md`.
