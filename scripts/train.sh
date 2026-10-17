#!/usr/bin/env bash
# Start one training run detached inside WSL (task 8.1):
#   bash scripts/train.sh <baseline|randomized> <seed> [extra train args...]
# Writes runs/logs/train_<variant>_seed<seed>.{log,pid,exit}; poll it with
#   bash scripts/poll_job.sh train_<variant>_seed<seed>
# Refuses to start while another GPU job runs. For a reported seed (listed in
# config/train.yaml seeds) it also refuses a dirty tree and any extra arguments,
# so reported runs always use the committed, frozen configuration.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"

variant="${1:?variant}"; seed="${2:?seed}"; shift 2
name="train_${variant}_seed${seed}"

if gpu_job_running; then
  echo "refusing: another GPU job is running" >&2
  exit 2
fi
reported="$("$PY" - "$variant" "$seed" <<'EOF'
import sys, yaml
seeds = yaml.safe_load(open("config/train.yaml"))["seeds"]
print("yes" if int(sys.argv[2]) in seeds.get(sys.argv[1], []) else "no")
EOF
)"
if [ "$reported" = "yes" ]; then
  require_clean_tree || exit 3
  if [ "$#" -gt 0 ]; then
    echo "refusing: reported seed $seed takes no extra arguments ($*)" >&2
    exit 4
  fi
fi

export LO_COMMIT="$(repo_commit)"
bash scripts/run_detached.sh "$name" "$PY" -m learned_opponents.train --variant "$variant" --seed "$seed" "$@"
