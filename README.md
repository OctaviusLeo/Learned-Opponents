# Learned Opponents

> **Private portfolio repository.** Not published, not packaged for PyPI, no
> public license. All contents are for the owner's portfolio only.

Sim-trained navigation policies for a differential-drive robot, trained in a
MuJoCo-based environment and raced against the player in-engine (Unreal Engine 5,
with Godot as the recorded fallback). The project studies whether domain
randomization closes the sim-to-sim transfer gap.

Every reported number in this README is produced by a repository command and
is cited with its **command, fixed seed, and full commit hash**. Placeholders
marked `TBD` are filled only from such a command — never typed or estimated.

## Research question

Does domain randomization during training improve the robustness and in-engine
transfer of a learned navigation policy, compared with a baseline trained at
nominal parameters only? (TBD: finalized statement.)

## Method

- Engine-independent observation/action spec (`learned_opponents/spec/`), mirrored
  by the game's in-engine observation builder.
- Differential-drive navigation environment with MuJoCo rangefinder rays
  (`learned_opponents/env/`) — owner-implemented core module `[ME]`.
- Domain-randomization sampler (`learned_opponents/randomization/`): baseline vs.
  randomized variants, all draws seeded from the run seed.
- PPO training (`learned_opponents/train/`) — owner-implemented core module `[ME]`.
- Robustness grid, metrics with 95% CIs, plots, and a generated summary
  (`learned_opponents/eval/`, `learned_opponents/plotting/`).
- ONNX export + Python/ONNX parity check (`learned_opponents/export/`).
- In-engine inference and an automated transfer check.

## Robustness plots

TBD — generated into `results/` by a single `eval.robustness` command and
sourced from the generated `results/summary.md`. Cited with command/seed/commit.

## Transfer table

TBD — sourced from the generated `results/summary.md` (never hand-edited):
per-variant in-game Success_Rate with 95% CI and the Transfer_Gap
(MuJoCo − in-game). Cited with command/seed/commit.

## Gameplay GIF

TBD — recorded from the in-engine race level.

## Reproduction steps

Each number is reproduced by one command with a fixed seed; the full commit
hash is recorded alongside. (Per-number commands are TBD, filled as each
milestone lands.)

1. Set up the environment per `docs/ENVIRONMENT.md` (GPU stack on the target
   machine; `JAX_DEFAULT_MATMUL_PRECISION=highest` on RTX 30-series).
2. Train: TBD command, per variant and seed.
3. Robustness grid + summary: TBD command.
4. ONNX export + parity: TBD command.
5. In-engine play + transfer check: TBD command.

## Test commands

- Python test suite (from the repo root):

  ```
  pytest
  ```

- In-engine (game-side) tests: TBD — run from the engine's test runner; the
  runtime name and version are recorded in `docs/ENVIRONMENT.md`.

## Known limitations

- **Sim-to-sim, not sim-to-real.** Transfer is measured between the training
  simulator and the game engine, not to physical hardware.
- **Domain randomization may not close the transfer gap.** The gap is reported
  as measured; nothing is tuned to shrink it.
- **RTX 30-series precision workaround.** `JAX_DEFAULT_MATMUL_PRECISION=highest`
  is required under MJX on RTX 30-series; results depend on it.
- **Determinism is bounded by GPU/framework nondeterminism.** Fixed-seed results
  are held to a stated tolerance, not byte-for-byte.
- **In-engine inference is Unreal-version-dependent.** If Unreal's inference
  runtime blocks progress, the Game and inference are implemented in the Godot
  fallback and that is recorded.
