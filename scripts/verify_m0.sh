#!/usr/bin/env bash
# Automated M0 gate (task 3g). Run inside WSL: bash scripts/verify_m0.sh
# Exits 0 only when every check passes; prints one line per check.
set -uo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
WIN_REPO="$(wslpath -w "$REPO")"
fail=0
check() {  # check <description> <command...>
  local desc="$1"; shift
  if "$@" > /dev/null 2>&1; then echo "PASS  $desc"; else echo "FAIL  $desc"; fail=1; fi
}

for d in learned_opponents/spec learned_opponents/env learned_opponents/randomization \
         learned_opponents/train learned_opponents/eval learned_opponents/export \
         learned_opponents/plotting tests config runs results game/LearnedOpponents docs \
         scripts .github/workflows; do
  check "layout: $d/" test -d "$d"
done

for p in .env secrets.yaml credentials.json id.key cert.pem api.token my.local \
         game/LearnedOpponents/Binaries/x game/LearnedOpponents/Intermediate/x \
         game/LearnedOpponents/Saved/x game/LearnedOpponents/DerivedDataCache/x \
         game/LearnedOpponents/LearnedOpponents.sln game/LearnedOpponents/.vs/x \
         learned_opponents/__pycache__/x.pyc runs/baseline/seed0/checkpoint.pt; do
  check ".gitignore covers $p" git.exe -C "$WIN_REPO" check-ignore -q "$p"
done
check ".gitignore keeps results/models/*.onnx tracked" \
  bash -c "! git.exe -C '$WIN_REPO' check-ignore -q results/models/baseline_seed0.onnx"

for ext in sh py yaml yml md json; do
  check ".gitattributes: *.$ext eol=lf" \
    bash -c "git.exe -C '$WIN_REPO' check-attr eol -- x.$ext | tr -d '\r' | grep -q 'eol: lf$'"
done

check "git remote origin is OctaviusLeo/Learned-Opponents" \
  bash -c "[ \"\$(git.exe -C '$WIN_REPO' remote get-url origin | tr -d '\r')\" = https://github.com/OctaviusLeo/Learned-Opponents.git ]"

check "results/smoke_result.json names a passing framework" python3 - <<'EOF'
import json, sys
r = json.load(open("results/smoke_result.json"))
c = r.get("chosen_framework")
sys.exit(0 if c and r["frameworks"][c]["passed"] else 1)
EOF

check "design.md records the chosen framework" \
  grep -q "Chosen framework: mjlab (MuJoCo Warp)" .kiro/specs/learned-opponents/design.md
for f in docs/evidence/smoke_mjx.log docs/evidence/smoke_mjlab.log docs/evidence/nvidia-smi_windows.txt \
         docs/evidence/nvidia-smi_wsl.txt docs/evidence/freeze_lo-mjx.txt docs/evidence/freeze_lo-mjlab.txt; do
  check "evidence exists: $f" test -s "$f"
done
check "ENVIRONMENT.md has the generated smoke block" grep -q "Chosen framework: \*\*mjlab\*\*" docs/ENVIRONMENT.md

check "project env imports learned_opponents" "$PY" -c "import learned_opponents"

if [ "$fail" -eq 0 ]; then echo "M0 gate: PASS"; else echo "M0 gate: FAIL"; fi
exit "$fail"
