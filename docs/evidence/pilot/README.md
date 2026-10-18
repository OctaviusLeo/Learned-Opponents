# Pilot evidence

Pilot runs use the baseline variant with seed 100 and are never reported as results.

## Pilot 1 (stopped): `entropy_coef: 0.005`

- Training command: `python -m learned_opponents.train --variant baseline --seed 100 --out runs/pilot/curve --force --set train.training_budget_steps=400000000 --checkpoint-every-steps 10000000` at commit `cfb6745` (`pilot1_entropy0.005_config.yaml`).
- Stopped by hand at 440 iterations (43.3M env steps). The policy's action std rose from 1.00 to 6.07 (`pilot1_entropy0.005_train_log.csv`, column `action_std`). The environment clips actions to [-1, 1], so a wider Gaussian costs nothing in return while the entropy bonus rewards it.
- Its four checkpoints, evaluated deterministically at nominal parameters on 4096 pilot layouts (`pilot1_entropy0.005_checkpoints.json`), lost success as the std grew: 0.951 (10M steps, std 0.69), 0.950 (20M, 1.28), 0.938 (30M, 2.76), 0.922 (40M, 5.11).
- Change: `entropy_coef: 0.0` in `config/train.yaml` before pilot 2. Nothing else changed.
