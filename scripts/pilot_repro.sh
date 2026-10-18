#!/usr/bin/env bash
# Pilot phase 2 (task 8.3): train the frozen setting twice with the pilot seed
# and measure the difference in nominal Success_Rate (training reproducibility).
# Run after the chosen budget is committed in config/train.yaml:
#   bash scripts/run_detached.sh pilot_repro bash scripts/pilot_repro.sh
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
require_clean_tree
export LO_COMMIT="$(repo_commit)"
for name in repro_a repro_b; do
  "$PY" -m learned_opponents.train --variant baseline --seed 100 --out "runs/pilot/$name" --force
done
"$PY" -m learned_opponents.train.pilot repro
