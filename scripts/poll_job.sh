#!/usr/bin/env bash
# Report the state of a job started with scripts/run_detached.sh:
#   bash scripts/poll_job.sh <name> [log_lines]
# Prints RUNNING, DONE <exit>, or LOST (process gone with no exit file, an
# environment failure), followed by the last lines of the log.
# Exit status: 0 done with exit 0, 1 done with non-zero, 2 running, 3 lost.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
name="$1"; lines="${2:-15}"
d="$REPO/runs/logs"
tail -n "$lines" "$d/$name.log" 2>/dev/null
if [ -f "$d/$name.exit" ]; then
  code="$(cat "$d/$name.exit")"
  echo "DONE $code"
  [ "$code" = "0" ] && exit 0 || exit 1
fi
if [ -f "$d/$name.pid" ] && kill -0 "$(cat "$d/$name.pid")" 2>/dev/null; then
  echo "RUNNING"
  exit 2
fi
echo "LOST (no exit file and process $(cat "$d/$name.pid" 2>/dev/null) is gone)"
exit 3
