"""results/summary.md is generated from results/eval/*.json, never edited by hand (GPU-free)."""

from __future__ import annotations

from learned_opponents.eval import summary


def test_committed_summary_regenerates_from_raw_results():
    # Fails if a number in summary.md was typed by hand, or a raw result changed without
    # regenerating it: run `python -m learned_opponents.eval.summary` and commit both files.
    assert summary.render() == summary.SUMMARY.read_text(encoding="utf-8")
