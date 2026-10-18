# Pilot evidence

Pilot runs use the baseline variant with seed 100 and are never reported as results.

## Pilot 1 (stopped): `entropy_coef: 0.005`

- Training command: `python -m learned_opponents.train --variant baseline --seed 100 --out runs/pilot/curve --force --set train.training_budget_steps=400000000 --checkpoint-every-steps 10000000` at commit `cfb6745` (`pilot1_entropy0.005_config.yaml`).
- Stopped by hand at 440 iterations (43.3M env steps). The policy's action std rose from 1.00 to 6.07 (`pilot1_entropy0.005_train_log.csv`, column `action_std`). The environment clips actions to [-1, 1], so a wider Gaussian costs nothing in return while the entropy bonus rewards it.
- Its four checkpoints, evaluated deterministically at nominal parameters on 4096 pilot layouts (`pilot1_entropy0.005_checkpoints.json`), lost success as the std grew: 0.951 (10M steps, std 0.69), 0.950 (20M, 1.28), 0.938 (30M, 2.76), 0.922 (40M, 5.11).
- Change: `entropy_coef: 0.0` in `config/train.yaml` before pilot 2. Nothing else changed.

## Pilot 2 (stopped): `entropy_coef: 0.0`, Gaussian policy

- Same command with `train.training_budget_steps=300000000`, at commit `6ee2b04` (`pilot2_entropy0_config.yaml`).
- Stopped by hand at 1271 iterations (125M env steps). Without the entropy bonus the std first fell (1.00 -> 0.31 by 18M steps) and then rose again, reaching 2.0 at 122M (`pilot2_entropy0_train_log.csv`).
- Deterministic nominal Success_Rate on 4096 pilot layouts (`pilot2_entropy0_checkpoints.json`): 0.966 at 10M steps, a peak of 0.974 at 40M (std 0.48), then 0.955 at 120M (std 1.89).
- Cause: a Gaussian whose samples the environment clips to [-1, 1] pays nothing for extra width once its mean saturates, so the width random-walks upward and the deterministic mean degrades.
- Change: `policy.distribution: beta` (rsl-rl `BetaDistribution`, support exactly [-1, 1], deterministic action = Beta mean). A 5M-step check trained normally at ~150k env-steps/s and its ONNX export matched the torch policy within 1.0e-6.
