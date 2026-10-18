#!/usr/bin/env bash
# Automated M2 gate (task 9g): complete baseline Run_Artifacts, the nominal result in
# results/summary.md with command, seed, and commit, and the full pytest suite.
# A missed nominal target is reported in the summary, not a gate failure.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
fail=0
check() { local d="$1"; shift; if "$@" > /dev/null 2>&1; then echo "PASS  $d"; else echo "FAIL  $d"; fail=1; fi; }

for s in 0 1 2; do
  for f in config.yaml seed commit checkpoint.pt train_log.csv summary.json; do
    check "runs/baseline/seed$s/$f" test -s "runs/baseline/seed$s/$f"
  done
done
check "summary.md has the baseline nominal section" grep -q "^## Nominal evaluation: baseline" results/summary.md
check "nominal section names its command" grep -q "^- Command: \`python -m learned_opponents.eval.robustness --nominal" results/summary.md
check "nominal section has a full commit hash" grep -Eq "^- Commit: \`[0-9a-f]{40}\`" results/summary.md
check "nominal section lists training seeds" grep -q "^- Training seeds: 0, 1, 2" results/summary.md
echo "== full pytest"
"$PY" -m pytest -q -p no:cacheprovider tests || fail=1
if [ "$fail" -eq 0 ]; then echo "M2 gate: PASS"; else echo "M2 gate: FAIL"; fi
exit "$fail"
