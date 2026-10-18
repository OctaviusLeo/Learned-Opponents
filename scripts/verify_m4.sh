#!/usr/bin/env bash
# Automated M4 gate (task 14g): the parity command on all six models and the full pytest suite.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
fail=0
echo "== parity (Python policy vs ONNX Runtime, tolerance 1e-4)"
"$PY" -m learned_opponents.export.parity || fail=1
n="$(ls results/models/*_seed*.onnx 2>/dev/null | wc -l)"
if [ "$n" -eq 6 ]; then echo "PASS  six exported models"; else echo "FAIL  six exported models (found $n)"; fail=1; fi
echo "== full pytest"
"$PY" -m pytest -q -p no:cacheprovider tests || fail=1
if [ "$fail" -eq 0 ]; then echo "M4 gate: PASS"; else echo "M4 gate: FAIL"; fi
exit "$fail"
