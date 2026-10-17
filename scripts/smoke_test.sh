#!/usr/bin/env bash
# M0 GPU framework smoke test (task 2.1). Run inside WSL:
#   bash scripts/smoke_test.sh
# Steps the shared smoke arena with >=1024 parallel environments in both
# candidate frameworks, writes results/smoke_result.json, and regenerates the
# smoke-test block in docs/ENVIRONMENT.md. Exits non-zero if no framework passed.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"

NUM_ENVS="${NUM_ENVS:-4096}"
STEPS="${STEPS:-200}"
mkdir -p docs/evidence results runs/smoke

if gpu_job_running; then
  echo "another GPU job is running; refusing to start" >&2
  exit 2
fi

COMMIT="$(repo_commit)"
CMD="bash scripts/smoke_test.sh (NUM_ENVS=$NUM_ENVS STEPS=$STEPS)"

nvidia-smi.exe > docs/evidence/nvidia-smi_windows.txt 2>&1
nvidia-smi > docs/evidence/nvidia-smi_wsl.txt 2>&1

echo "== MJX (lo-mjx)"
# Ampere precision setting from the Playground README. Under WSL2 a single
# multi-GB cudaMalloc fails even with free memory, so JAX allocates on demand.
JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$HOME/.venvs/lo-mjx/bin/python" scripts/smoke/smoke_mjx.py \
  --num-envs "$NUM_ENVS" --steps "$STEPS" --out runs/smoke/mjx.json > docs/evidence/smoke_mjx.log 2>&1
echo "exit $?"

echo "== mjlab / MuJoCo Warp (lo-mjlab)"
"$HOME/.venvs/lo-mjlab/bin/python" scripts/smoke/smoke_warp.py \
  --num-envs "$NUM_ENVS" --steps "$STEPS" --out runs/smoke/mjlab.json > docs/evidence/smoke_mjlab.log 2>&1
echo "exit $?"

python3 scripts/smoke/merge_results.py results/smoke_result.json runs/smoke/env_block.md \
  docs/evidence/nvidia-smi_windows.txt docs/evidence/nvidia-smi_wsl.txt "$CMD" "$COMMIT" \
  runs/smoke/mjx.json runs/smoke/mjlab.json
status=$?
python3 scripts/lib/replace_block.py docs/ENVIRONMENT.md smoke runs/smoke/env_block.md
exit $status
