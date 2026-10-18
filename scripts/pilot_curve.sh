#!/usr/bin/env bash
# Pilot phase 1 (task 8.3): train the pilot seed for a long horizon with
# intermediate checkpoints, then evaluate them and apply the budget rule.
#   bash scripts/run_detached.sh pilot_curve bash scripts/pilot_curve.sh
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
HORIZON="${HORIZON:-400000000}"
EVERY="${EVERY:-10000000}"
export LO_COMMIT="$(repo_commit)"
"$PY" -m learned_opponents.train --variant baseline --seed 100 --out runs/pilot/curve --force \
  --set "train.training_budget_steps=$HORIZON" --checkpoint-every-steps "$EVERY"
"$PY" -m learned_opponents.train.pilot curve
