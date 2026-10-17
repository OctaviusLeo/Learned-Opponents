#!/usr/bin/env bash
# Start a long command detached inside WSL and return immediately:
#   bash scripts/run_detached.sh <name> <command> [args...]
# Writes runs/logs/<name>.log (stdout+stderr), runs/logs/<name>.pid, and, when
# the command ends, runs/logs/<name>.exit with its exit status. Poll with
#   bash scripts/poll_job.sh <name>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
name="$1"; shift
mkdir -p "$REPO/runs/logs"
log="$REPO/runs/logs/$name.log"
rm -f "$REPO/runs/logs/$name.exit"
cmd="$(printf '%q ' "$@")"
cd "$REPO"
setsid nohup bash -c "$cmd; echo \$? > runs/logs/$name.exit" > "$log" 2>&1 < /dev/null &
echo $! > "runs/logs/$name.pid"
echo "started $name (pid $(cat "runs/logs/$name.pid")), log runs/logs/$name.log"
