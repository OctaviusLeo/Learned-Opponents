#!/usr/bin/env bash
# Automated M3 gate (task 12g): every grid cell for both variants in results/summary.md,
# one plot per grid parameter, and the full pytest suite.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
fail=0
check() { local d="$1"; shift; if "$@" > /dev/null 2>&1; then echo "PASS  $d"; else echo "FAIL  $d"; fail=1; fi; }

for v in baseline randomized; do
  for s in 0 1 2; do check "runs/$v/seed$s complete" test -s "runs/$v/seed$s/summary.json"; done
done
check "every grid cell of both variants is in summary.md" "$PY" - <<'EOF'
import sys
from learned_opponents.config import load_yaml
from learned_opponents.eval.grid_config import parse_grid
from learned_opponents.eval.summary_sections import LABELS

grid = parse_grid(load_yaml("robustness_grid.yaml"))
text = open("results/summary.md", encoding="utf-8").read()
sections = {chunk.split("\n", 1)[0]: chunk for chunk in text.split("\n### ")[1:]}
missing = []
for p, values in grid.items():
    body = sections.get(LABELS[p], "")
    for variant in ("baseline", "randomized"):
        for v in values:
            if f"| {variant} | {int(v) if p == 'latency_steps' else f'{v:g}'} | " not in body:
                missing.append((p, variant, v))
print("missing:", missing)
sys.exit(1 if missing else 0)
EOF
for p in mass_scale friction_scale latency_steps; do
  check "plot results/robustness_$p.png" test -s "results/robustness_$p.png"
done
check "robustness section has command and commit" grep -q "^## Robustness grid" results/summary.md
echo "== full pytest"
"$PY" -m pytest -q -p no:cacheprovider tests || fail=1
if [ "$fail" -eq 0 ]; then echo "M3 gate: PASS"; else echo "M3 gate: FAIL"; fi
exit "$fail"
