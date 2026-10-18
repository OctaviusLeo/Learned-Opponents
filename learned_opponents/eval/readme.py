"""Copy result sections from results/summary.md into README.md (task 18.3).

    python -m learned_opponents.eval.readme          # rewrite the README's result blocks
    python -m learned_opponents.eval.readme --check  # exit 1 if any block differs from summary.md

A README block ``<!-- BEGIN summary:<title> --> ... <!-- END summary:<title> -->`` holds the
summary section whose heading is ``## <title>`` verbatim, so every number in it is the generated
number, with its command, seeds, and commit. Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
SUMMARY = REPO / "results" / "summary.md"
BLOCK = re.compile(r"<!-- BEGIN summary:(?P<title>[^>]+?) -->\n(?P<body>.*?)<!-- END summary:(?P=title) -->", re.S)


def sections(summary_text: str) -> dict[str, str]:
    """``## Title`` sections of summary.md (text from the heading line to the next ``## ``)."""
    out = {}
    for chunk in re.split(r"(?m)^(?=## )", summary_text):
        if chunk.startswith("## "):
            title = chunk.split("\n", 1)[0][3:].strip()
            out[title] = chunk.rstrip() + "\n"
    return out


def sync(readme_text: str, summary_text: str) -> tuple[str, list[str]]:
    secs = sections(summary_text)
    missing = []

    def repl(m: re.Match) -> str:
        title = m.group("title")
        if title not in secs:
            missing.append(title)
            return m.group(0)
        return f"<!-- BEGIN summary:{title} -->\n{secs[title]}<!-- END summary:{title} -->"

    return BLOCK.sub(repl, readme_text), missing


def check(readme_text: str, summary_text: str) -> list[str]:
    secs = sections(summary_text)
    problems = []
    blocks = list(BLOCK.finditer(readme_text))
    if not blocks:
        problems.append("README has no summary blocks")
    for m in blocks:
        title = m.group("title")
        if secs.get(title) != m.group("body"):
            problems.append(f"block '{title}' differs from results/summary.md")
    return problems


def main(argv: list[str]) -> int:
    readme = README.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    if "--check" in argv:
        problems = check(readme, summary)
        for p in problems:
            print("FAIL", p)
        print("README blocks match results/summary.md" if not problems else "README out of date")
        return 1 if problems else 0
    new, missing = sync(readme, summary)
    with open(README, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    for t in missing:
        print(f"no summary section '{t}' yet; block left unchanged")
    print("README result blocks synced from results/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
