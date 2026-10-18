#!/usr/bin/env bash
# Write the files the Unreal game and its tests read (task 15.2): results/game/task.json,
# results/fixtures/obs_cases.json (Property 3 cases, fixed seed), results/game/race_course.json.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO"
export LO_COMMIT="${LO_COMMIT:-$(repo_commit)}"
"$PY" -m learned_opponents.export.game_data
