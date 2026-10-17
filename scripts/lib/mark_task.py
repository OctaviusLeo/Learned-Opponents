"""Check off tasks in the spec's tasks.md: python mark_task.py <tasks.md> <id>...

A task id matches a line such as ``- [ ] 2.1 ...``, ``- [ ] 3. ...`` or
``- [ ] 3g. ...``; the box becomes ``[x]``. Exits 1 if an id is not found.
"""

from __future__ import annotations

import re
import sys


def mark(text: str, task_id: str) -> tuple[str, bool]:
    pat = re.compile(rf"^(\s*- )\[[ x]\]( {re.escape(task_id)}\.?\s)", re.M)
    new, n = pat.subn(r"\1[x]\2", text, count=1)
    return new, n == 1


def main(path: str, *ids: str) -> int:
    text = open(path, encoding="utf-8").read()
    missing = []
    for i in ids:
        text, ok = mark(text, i)
        if not ok:
            missing.append(i)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if missing:
        print("not found:", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
