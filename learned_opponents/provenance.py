"""Commit hash and tree state for every reported number (standard library only).

Git is only ever Windows Git. Inside WSL it is reached through interop as
``git.exe`` (read-only queries); ``LO_COMMIT`` set by the calling script wins.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRTY = ("results/", "runs/", "docs/", "README.md")


def _git(*args: str) -> str | None:
    exe = shutil.which("git.exe") or shutil.which("git")
    if exe is None:
        return None
    repo = str(REPO_ROOT)
    if exe.endswith(".exe") and repo.startswith("/mnt/"):
        try:
            repo = subprocess.run(["wslpath", "-w", repo], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    try:
        out = subprocess.run([exe, "-C", repo, *args], capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.replace("\r", "")


def repo_commit() -> str:
    env = os.environ.get("LO_COMMIT")
    if env:
        return env.strip()
    out = _git("rev-parse", "HEAD")
    return out.strip() if out else "unknown"


def dirty_paths() -> list[str] | None:
    """Changed paths outside the generated directories, or None if Git is unavailable."""
    out = _git("status", "--porcelain")
    if out is None:
        return None
    paths = [line[3:] for line in out.splitlines() if line.strip()]
    return [p for p in paths if not p.startswith(ALLOWED_DIRTY)]
