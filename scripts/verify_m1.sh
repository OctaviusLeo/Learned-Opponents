#!/usr/bin/env bash
# Automated M1 gate (task 7g). Run inside WSL: bash scripts/verify_m1.sh
#   1. the full pytest suite (GPU tests included) in ~/.venvs/learned-opponents;
#   2. pytest -m "not gpu" in a throwaway venv holding only requirements-ci.txt,
#      exactly what CI installs.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
UV="${UV:-$HOME/.local/bin/uv}"
CI_VENV="$HOME/.venvs/lo-ci-check"
fail=0

echo "== full pytest (project env, GPU tests included)"
"$PY" -m pytest -q -p no:cacheprovider tests || fail=1

echo "== pytest -m 'not gpu' with requirements-ci.txt only"
rm -rf "$CI_VENV"
"$UV" venv -q --python 3.12 "$CI_VENV" && \
  "$UV" pip install -q --python "$CI_VENV/bin/python" -r requirements-ci.txt && \
  "$UV" pip install -q --python "$CI_VENV/bin/python" --no-deps -e . && \
  "$CI_VENV/bin/python" -m pytest -q -p no:cacheprovider -m "not gpu" tests || fail=1
rm -rf "$CI_VENV"

if [ "$fail" -eq 0 ]; then echo "M1 gate: PASS"; else echo "M1 gate: FAIL"; fi
exit "$fail"
