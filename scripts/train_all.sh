#!/usr/bin/env bash
# Reproduce all six reported training runs from one command (task 10):
#   bash scripts/run_detached.sh train_all bash scripts/train_all.sh
# Trains, one after another, every seed in config/train.yaml for each variant
# (VARIANTS="baseline randomized" by default) with the frozen configs. A seed
# whose run already finished (summary.json present) is skipped unless FORCE=1.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"

if gpu_job_running; then
  echo "refusing: another GPU job is running" >&2
  exit 2
fi
require_clean_tree
export LO_COMMIT="$(repo_commit)"
echo "commit $LO_COMMIT"

for variant in ${VARIANTS:-baseline randomized}; do
  seeds="$("$PY" -c "import yaml; print(' '.join(map(str, yaml.safe_load(open('config/train.yaml'))['seeds']['$variant'])))")"
  for seed in $seeds; do
    out="runs/$variant/seed$seed"
    if [ -f "$out/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "skip $out (finished; FORCE=1 retrains)"
      continue
    fi
    echo "== train $variant seed $seed"
    "$PY" -m learned_opponents.train --variant "$variant" --seed "$seed" --force
  done
done
echo "all runs finished"
