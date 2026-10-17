"""Replace a generated block in a Markdown file.

Usage: python replace_block.py <file> <name> <content-file>

The block is delimited by ``<!-- BEGIN <name> -->`` and ``<!-- END <name> -->``.
If the markers are missing, the block is appended at the end of the file.
Only the text between the markers is generated; the rest stays hand-written.
"""

from __future__ import annotations

import sys


def replace_block(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        return head + block + tail
    return text.rstrip() + "\n\n" + block + "\n"


def main(path: str, name: str, content_path: str) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(content_path, encoding="utf-8") as f:
        body = f.read()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(replace_block(text, name, body))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
