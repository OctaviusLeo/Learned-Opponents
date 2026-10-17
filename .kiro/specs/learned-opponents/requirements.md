# Requirements Document

## Introduction

Learned Opponents trains a navigation policy for a differential-drive robot in a GPU-accelerated MuJoCo simulation on a single NVIDIA RTX 3080, in two variants — a baseline with fixed parameters and a domain-randomized variant — then measures how much the randomization improves robustness on a perturbation grid, exports the trained policy to ONNX, and uses it as the opponent in a small playable Unreal Engine 5 racing game. The central question the project answers with real, reproducible numbers is: how much does domain randomization improve robustness, and how large is the sim-to-sim transfer gap when the same policy drives a robot inside a game engine.

The project is portfolio evidence for robot-learning and sim-to-real roles, so every reported number must be reproducible from a single repository command with a fixed seed, recorded with its command, seed, and full commit hash; every third-party dependency must be proven on the target machine and license-checked; and no result is ever hand-typed, estimated, or borrowed.

The work is organized into milestones M0 through M5. M0 selects and proves the GPU training framework with a smoke test. M1 delivers the navigation environment with tests (reset, step, reward, termination, ray readings). M2 trains the baseline to a stated nominal success rate. M3 adds the domain-randomized training and the robustness grid with confidence intervals. M4 exports the policy to ONNX with a parity test. M5 delivers the Unreal game level, in-engine inference, the transfer check, and the README. Each milestone ends with an automated verification gate (a repository command whose exit status decides pass or fail).

This document defines the requirements only. Framework choice (MuJoCo Playground / MJX versus mjlab / MuJoCo Warp), in-engine inference runtime choice, and implementation details are deferred to the design phase.

## Glossary

- **System**: The complete Learned Opponents project, comprising the MuJoCo training environment, the PPO training pipeline, the robustness-evaluation and analysis tooling, the ONNX export and parity test, and the Unreal Engine 5 game.
- **Environment**: The custom GPU-steppable MuJoCo task in which a differential-drive robot must reach a goal in an arena with randomized obstacles.
- **Robot**: The simulated differential-drive ground vehicle controlled by wheel-velocity commands.
- **Arena**: The bounded simulated space containing the Robot, the goal, and randomized obstacles.
- **Range_Rays**: The fixed set of ray (range) sensors emitted from the Robot that measure distance to the nearest obstacle or wall along each ray direction.
- **Observation**: The engine-independent observation vector supplied to the policy: the Range_Rays readings, the goal direction and distance, and the Robot's own velocity.
- **Action**: The engine-independent action vector produced by the policy: wheel-velocity commands.
- **Obs_Action_Spec**: The written, engine-independent definition of the Observation and Action vectors (ordering, units, ranges, and construction rules) that both the training environment and the game use to build identical vectors.
- **Policy**: The neural-network navigation controller trained by PPO that maps an Observation to an Action.
- **PPO**: Proximal Policy Optimization, the on-policy reinforcement-learning algorithm used for training.
- **Baseline_Variant**: The Policy trained with fixed (non-randomized) simulation parameters.
- **Randomized_Variant**: The Policy trained with domain randomization applied to mass, wheel friction, motor strength, sensor noise, action latency, and observation latency.
- **Training_Budget**: The fixed quantity of training (environment steps or equivalent) applied identically to each training run so the two variants are compared fairly.
- **Seed**: The integer random seed that makes a training or evaluation run reproducible.
- **Checkpoint**: A saved set of Policy parameters produced by a training run.
- **Run_Artifacts**: The per-run saved configuration, Seed, and Checkpoint for a training run.
- **Domain_Randomization**: The sampling of simulation parameters (mass, wheel friction, motor strength, sensor noise, action latency, observation latency) from configured ranges at reset during training.
- **Robustness_Grid**: The grid of perturbed mass, wheel friction, and latency values — including values outside the training randomization ranges — on which each Policy variant is evaluated.
- **Success_Rate**: The fraction of evaluation episodes in which the Robot reaches the goal without collision before timeout.
- **Collision_Rate**: The fraction of evaluation episodes that end in a collision.
- **Time_To_Goal**: The elapsed simulated time for the Robot to reach the goal in a successful episode.
- **Confidence_Interval**: A 95% confidence interval reported for an estimated rate or mean.
- **ONNX_Model**: The Policy exported to the ONNX interchange format.
- **Parity_Test**: The test that runs a fixed set of saved Observations through the Python Policy and the exported ONNX_Model and checks that the Actions agree within a stated tolerance.
- **Game**: The Unreal Engine 5 level in which the player drives one Robot and races the Policy-driven Robot to checkpoints through obstacles.
- **In_Engine_Inference**: Running the exported ONNX_Model inside the Game to drive the opponent Robot.
- **In_Engine_Observation**: The Observation vector rebuilt inside the Game from in-engine raycasts using the Obs_Action_Spec.
- **Transfer_Check**: The automated in-Game evaluation comparing each Policy variant's in-Game Success_Rate against its MuJoCo Success_Rate.
- **Transfer_Gap**: The reported difference between the MuJoCo Success_Rate and the in-Game Success_Rate for a Policy variant.
- **Measured_Value**: A number produced by a script in this repository, recorded with its command, Seed, and full commit hash.
- **Configured_Value**: A parameter value set in a configuration file before a run.
- **Summary_File**: The generated file `results/summary.md` containing the measured robustness and transfer results.
- **ENVIRONMENT_Doc**: The file `docs/ENVIRONMENT.md` recording exact tool and dependency versions proven on the machine.
- **THIRD_PARTY_Doc**: The file `THIRD_PARTY.md` recording each third-party dependency and its license.
- **Smoke_Test**: The M0 procedure that selects and validates the GPU training framework.

## Requirements

### Requirement 0: Governing non-functional requirements

**User Story:** As the repository owner, I want every result reproducible and every dependency proven and license-checked, so that I can defend every line of the project in an interview.

#### Acceptance Criteria

1. WHERE a number is reported in the README, in `docs/`, or in the Summary_File, THE System SHALL produce that number from a script in this repository and SHALL record, in the same document and within the same section as the number, the generating command, the Seed, and the full commit hash.
2. WHERE a System component uses randomness, THE System SHALL expose a configurable Seed through a documented configuration parameter.
3. WHEN two runs of a System component execute with the same Seed and the same configuration on the same machine, THE System SHALL produce results that are identical to the run's stated reproducibility tolerance, and SHALL state that tolerance alongside the result.
4. IF a run fails or misses a stated target, THEN THE System SHALL report, in the run output, the actual result and the identified cause, and SHALL leave the stated target and the test unchanged.
5. THE System SHALL record, in the ENVIRONMENT_Doc, the exact version string of every tool, library, API, and dependency it relies on.
6. WHEN a dependency version is recorded in the ENVIRONMENT_Doc, THE System SHALL include, alongside that version, at least one of the following verifiable artifacts obtained on the target machine: a package-manager query result showing that exact version, a documentation reference for that exact version, or a minimal test that passes against that exact version.
7. WHERE a third-party package or asset is added to the repository, THE System SHALL record, in the THIRD_PARTY_Doc, the package or asset name, its version, and its license identifier.
8. THE System SHALL read every secret from an environment variable and SHALL list every secret-bearing file in `.gitignore` so that no such file is tracked in version control.
9. WHERE a module contains more than one public function or exceeds 50 source lines, THE System SHALL provide unit tests for that module, and THE README SHALL state the exact single command that runs those tests.
10. WHERE the training machine is an RTX 30-series card and the chosen framework is MuJoCo Playground (MJX), THE System SHALL set `JAX_DEFAULT_MATMUL_PRECISION=highest` for training runs and SHALL record that setting in the ENVIRONMENT_Doc.

### Requirement 1: Framework smoke test and choice (M0)

**User Story:** As the repository owner, I want the GPU training framework selected by a smoke test and justified with evidence, so that thousands of parallel environments with range sensors step correctly on the RTX 3080.

#### Acceptance Criteria

1. THE System SHALL evaluate MuJoCo Playground (MJX) and mjlab (MuJoCo Warp) for the custom navigation task and SHALL record, in `design.md`, the decision and the evidence supporting it.
2. WHEN the Smoke_Test runs, THE System SHALL build a minimal Environment that steps on the GPU and returns Range_Rays readings, and SHALL demonstrate stepping with at least 1024 parallel environments without error.
3. THE Smoke_Test SHALL pass for a framework only when all of the following hold: the output of `nvidia-smi` is saved in the ENVIRONMENT_Doc; the minimal Environment steps on the GPU for the stated parallel-environment count; the step returns a Range_Rays reading for each environment; and the exact versions of the framework and its GPU backend are recorded in the ENVIRONMENT_Doc with a verifiable artifact.
4. WHEN the Smoke_Test selects a framework, THE System SHALL record the chosen framework and the measured evidence (parallel-environment count, step throughput, and that Range_Rays were returned) in `design.md`.
5. IF the chosen framework is MuJoCo Playground (MJX) on the RTX 3080, THEN THE System SHALL set `JAX_DEFAULT_MATMUL_PRECISION=highest` during the Smoke_Test and SHALL record that setting in the ENVIRONMENT_Doc.
6. IF both frameworks fail the Smoke_Test, THEN THE System SHALL report that no framework passed and SHALL record, for each, the Smoke_Test condition that first failed in the ENVIRONMENT_Doc.

### Requirement 2: Navigation task definition

**User Story:** As a user, I want an engine-independent definition of the navigation task's observations and actions, so that the game can rebuild them exactly as training used them.

#### Acceptance Criteria

1. THE System SHALL define the Obs_Action_Spec as a written, engine-independent specification stating, for the Observation vector, the ordering, the units, and the value ranges of the Range_Rays readings, the goal direction, the goal distance, and the Robot's own velocity.
2. THE System SHALL define, in the Obs_Action_Spec, the Action vector as wheel-velocity commands with their ordering, units, and value ranges.
3. THE Observation SHALL contain a fixed number of Range_Rays, where that number is a Configured_Value and is identical for the training Environment and the Game.
4. THE System SHALL construct the Observation using the same construction rules in the training Environment and in the Game, such that the only intended difference between the two is the engine that produces the raycasts.
5. THE Obs_Action_Spec SHALL define the task domain as navigation to a goal with obstacle avoidance using differential-drive wheel-velocity actions, and SHALL NOT include legged-locomotion observations or actions.

### Requirement 3: Navigation environment

**User Story:** As a user, I want a differential-drive navigation environment that resets, steps, rewards, terminates, and returns ray readings, so that I can train and evaluate the policy on the GPU.

#### Acceptance Criteria

1. WHEN the Environment resets, THE Environment SHALL place the Robot and the goal, generate the randomized obstacles for the episode, and return an initial Observation conforming to the Obs_Action_Spec.
2. WHEN the Environment steps with an Action, THE Environment SHALL advance the simulation, apply the differential-drive wheel-velocity commands, and return the next Observation, the reward, and the termination indication for that step.
3. WHEN the Robot reaches the goal within the configured arrival tolerance, THE Environment SHALL terminate the episode with a success indication.
4. WHEN the Robot collides with an obstacle or an Arena wall, THE Environment SHALL terminate the episode with a collision indication.
5. IF the episode reaches the configured step or time limit without success or collision, THEN THE Environment SHALL terminate the episode with a timeout indication.
6. WHEN the Environment returns an Observation, THE Observation SHALL contain exactly the configured number of Range_Rays readings, each within the configured maximum range.
7. THE Environment SHALL compute the reward from the configured reward terms (at least progress toward the goal, a success term, and a collision penalty), and SHALL expose each reward term's weight as a Configured_Value.
8. THE Environment SHALL step on the GPU for at least the parallel-environment count demonstrated in the Smoke_Test.

### Requirement 4: Training (baseline and domain-randomized)

**User Story:** As a user, I want PPO training in a baseline and a domain-randomized variant under one fixed budget, so that I can attribute robustness differences to the randomization alone.

#### Acceptance Criteria

1. THE System SHALL train the Policy with PPO in two variants — the Baseline_Variant with fixed simulation parameters and the Randomized_Variant with Domain_Randomization — using the same Training_Budget for each.
2. THE Randomized_Variant SHALL sample, at episode reset during training, the Robot mass, the wheel friction, the motor strength, the sensor noise, the action latency, and the observation latency, each from a configured range (_Requirements: 4 — domain-randomization parameter set_).
3. THE System SHALL run at least 3 Seeds for the Baseline_Variant and at least 3 Seeds for the Randomized_Variant.
4. WHEN a training run executes, THE System SHALL save the Run_Artifacts for that run: the full configuration, the Seed, and the Checkpoint.
5. THE Training_Budget, the PPO hyperparameters, and the Domain_Randomization ranges SHALL each be Configured_Values recorded in a configuration file under version control.
6. WHEN a training run is launched, THE System SHALL be invocable from a single command that takes the variant and the Seed and writes the Run_Artifacts to a run-specific location.
7. WHERE the chosen framework is MuJoCo Playground (MJX) on the RTX 3080, THE training runs SHALL execute with `JAX_DEFAULT_MATMUL_PRECISION=highest`.

### Requirement 5: Nominal success target (M2)

**User Story:** As a user, I want the baseline to reach a stated success rate at nominal parameters, so that there is a trained policy worth evaluating for robustness.

#### Acceptance Criteria

1. THE System SHALL state, as a Configured_Value, the nominal success-rate target the Baseline_Variant must reach at nominal (unperturbed) simulation parameters.
2. WHEN the Baseline_Variant is evaluated at nominal parameters over a stated number of episodes, THE System SHALL report the Measured_Value Success_Rate with its 95% Confidence_Interval against the stated target.
3. IF the Baseline_Variant does not reach the stated nominal target, THEN THE System SHALL report the actual Measured_Value and the identified cause, and SHALL NOT loosen the target or the evaluation to force a pass.

### Requirement 6: Robustness evaluation (M3)

**User Story:** As a user, I want each variant evaluated on a perturbation grid with confidence intervals, so that I can measure how much domain randomization improves robustness.

#### Acceptance Criteria

1. THE System SHALL evaluate each Policy variant on a Robustness_Grid of perturbed mass, wheel friction, and latency values that includes at least one value outside the training Domain_Randomization range for each perturbed parameter.
2. WHEN the System evaluates a Robustness_Grid cell for a Policy variant, THE System SHALL report the Success_Rate, the Collision_Rate, and the mean Time_To_Goal, each with its 95% Confidence_Interval, computed over a stated number of evaluation episodes.
3. THE System SHALL aggregate every Robustness_Grid result across the at-least-3 Seeds per variant and SHALL state how the per-Seed results are aggregated.
4. WHEN the robustness results are produced, THE System SHALL save plots of Success_Rate against each perturbed parameter with the Baseline_Variant and the Randomized_Variant on the same axes.
5. WHEN the robustness evaluation completes, THE System SHALL write the Summary_File containing the Robustness_Grid results for both variants.
6. THE robustness evaluation SHALL be invocable from a single command, and every reported number SHALL be recorded with its command, Seed, and full commit hash.

### Requirement 7: Policy export and parity (M4)

**User Story:** As a user, I want the policy exported to ONNX and a parity test, so that the in-engine model provably matches the trained Python model.

#### Acceptance Criteria

1. THE System SHALL export the trained Policy to an ONNX_Model from a single command.
2. THE System SHALL save a fixed set of Observations and the corresponding Actions produced by the Python Policy, for use as the Parity_Test fixtures.
3. WHEN the Parity_Test runs, THE Parity_Test SHALL feed the saved Observations through the Python Policy and through the ONNX_Model and SHALL compare the resulting Actions.
4. THE Parity_Test SHALL state a numeric tolerance and SHALL pass only when every compared Action agrees within that tolerance.
5. IF any compared Action exceeds the stated tolerance, THEN THE Parity_Test SHALL fail and SHALL report the Observation index and the magnitude of the largest disagreement.
6. THE ONNX_Model SHALL take an input conforming to the Observation defined in the Obs_Action_Spec and SHALL produce an output conforming to the Action defined in the Obs_Action_Spec.

### Requirement 8: Game and in-engine observation (M5)

**User Story:** As a player, I want a small Unreal level where I race the policy-driven robot to checkpoints through obstacles, so that the trained opponent is playable.

#### Acceptance Criteria

1. THE Game SHALL provide a level in which the player drives one Robot and a Policy-driven Robot races to checkpoints through obstacles.
2. THE Game SHALL run In_Engine_Inference with the exported ONNX_Model to produce the opponent Robot's Actions.
3. THE System SHALL record, in the ENVIRONMENT_Doc, which in-engine inference runtime was used (whether Unreal's Neural Network Engine with an ONNX Runtime backend supports the installed Unreal version, or the recorded fallback), with its exact version and a verifiable artifact.
4. WHEN the Game builds the In_Engine_Observation, THE Game SHALL construct it from in-engine raycasts using the Obs_Action_Spec construction rules so that it matches the training Observation ordering, units, and ranges.
5. THE System SHALL provide unit tests that verify the In_Engine_Observation construction matches the Obs_Action_Spec for a set of known raycast inputs.
6. IF In_Engine_Inference with Unreal cannot be made to work on the installed version, THEN THE System SHALL record the blocking problem in the ENVIRONMENT_Doc and SHALL implement the Game and In_Engine_Inference in the recorded fallback engine instead.

### Requirement 9: Transfer check (M5)

**User Story:** As a user, I want an automated in-game transfer check, so that I can report the sim-to-sim transfer gap honestly.

#### Acceptance Criteria

1. THE System SHALL run at least 50 automated in-Game episodes for the Baseline_Variant and at least 50 automated in-Game episodes for the Randomized_Variant.
2. WHEN the Transfer_Check runs for a Policy variant, THE System SHALL report that variant's in-Game Success_Rate with its 95% Confidence_Interval.
3. WHEN the in-Game Success_Rate for a variant is produced, THE System SHALL report the Transfer_Gap between that variant's MuJoCo Success_Rate and its in-Game Success_Rate.
4. THE System SHALL report the Transfer_Gap as the Measured_Value it actually is, and SHALL NOT adjust the Game, the Policy, or the evaluation to shrink the reported gap.
5. THE Transfer_Check SHALL be invocable from a single command and SHALL record every reported number with its command, Seed, and full commit hash.

### Requirement 10: Tests and continuous integration

**User Story:** As the repository owner, I want automated tests and CI, so that correctness is verified on every change.

#### Acceptance Criteria

1. THE System SHALL provide tests covering the Environment (reset, step, reward, termination, and Range_Rays readings), the Observation and Action construction, the Domain_Randomization sampling, the ONNX parity, and the In_Engine_Observation construction.
2. THE System SHALL provide pytest tests for the Python components (Environment logic, Observation/Action construction, metrics, and ONNX parity).
3. THE System SHALL provide unit tests for the In_Engine_Observation construction in the Game.
4. WHEN a change is pushed to the repository, THE continuous-integration pipeline SHALL run the pytest tests.
5. IF any pytest test fails during a continuous-integration run, THEN THE continuous-integration pipeline SHALL report a failed status for that run.
6. THE README SHALL state the exact command for each test suite.

### Requirement 11: README and reproducibility

**User Story:** As a reader, I want a README with the question, method, results, and reproduction steps, so that I can understand and reproduce every number.

#### Acceptance Criteria

1. THE README SHALL contain the research question, the method, the robustness plots, the transfer table, a gameplay GIF, how to reproduce each number, and a known-limitations section.
2. THE README SHALL present the robustness results and the Transfer_Gap table sourced from the generated Summary_File, not from hand edits.
3. WHEN a result is cited in the README, THE README SHALL record the generating command, the Seed, and the full commit hash in the same section as the result.
4. WHEN M5 completes, THE System SHALL demonstrate the end-to-end pipeline — train, evaluate robustness, export to ONNX, pass the Parity_Test, play the Game with In_Engine_Inference, and run the Transfer_Check — with every reported number produced by a repository command.

## Out of Scope

The following are explicitly out of scope: legged locomotion, Isaac Sim, multiplayer, store releases, and photorealism. Publishing the Game or the repository is a separate step that waits for the owner's approval; the repository stays private with no public license, packaging, or releases until then.
