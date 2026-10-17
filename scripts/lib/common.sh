# Shared helpers for the WSL bash scripts. Source this file; do not execute it.
#
# Git rule: Git is only ever Windows Git. From WSL it is called through
# interop as git.exe (read-only queries), so the index is never touched by a
# second Git install.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/learned-opponents}"
PY="$VENV/bin/python"

repo_commit() {
  git.exe -C "$(wslpath -w "$REPO")" rev-parse HEAD | tr -d '\r'
}

# Paths that a reported run may leave modified (design "Numbers come from
# committed code").
_allowed_dirty='^(results/|runs/|docs/|README\.md)'

# Prints offending paths and returns 1 when the tree has changes outside the
# allowed generated paths.
require_clean_tree() {
  local dirty
  dirty="$(git.exe -C "$(wslpath -w "$REPO")" status --porcelain | tr -d '\r' | cut -c4- | grep -Ev "$_allowed_dirty" || true)"
  if [ -n "$dirty" ]; then
    echo "refusing: uncommitted changes outside results/, runs/, docs/, README.md:" >&2
    echo "$dirty" >&2
    return 1
  fi
}

# Returns 0 (true) when another GPU job is running: a project Python process
# in WSL other than the caller's own process tree, or an Unreal process on Windows.
gpu_job_running() {
  local self="$$"
  if pgrep -f "learned_opponents\.(train|eval|export)|scripts/smoke/smoke_" | grep -vx "$self" | grep -q .; then
    return 0
  fi
  if tasklist.exe 2>/dev/null | tr -d '\r' | grep -Eiq '^Unreal(Editor|Editor-Cmd|BuildTool)?'; then
    return 0
  fi
  return 1
}
