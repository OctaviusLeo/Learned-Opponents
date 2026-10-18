"""README result blocks are verbatim copies of summary.md sections (GPU-free)."""

from __future__ import annotations

from learned_opponents.eval import readme

SUMMARY = "# Results\n\nintro\n\n## Alpha\n\n| a | 1 |\n\n## Beta\n\nx = 0.5\n"
README = ("# Project\n\n<!-- BEGIN summary:Alpha -->\nold\n<!-- END summary:Alpha -->\n\ntext\n\n"
          "<!-- BEGIN summary:Beta -->\n<!-- END summary:Beta -->\n")


def test_sections_split_on_level_two_headings():
    s = readme.sections(SUMMARY)
    assert list(s) == ["Alpha", "Beta"]
    assert s["Beta"] == "## Beta\n\nx = 0.5\n"


def test_sync_then_check_round_trip():
    assert readme.check(README, SUMMARY)                     # stale blocks are reported
    new, missing = readme.sync(README, SUMMARY)
    assert not missing and "| a | 1 |" in new and "old" not in new
    assert readme.check(new, SUMMARY) == []
    edited = new.replace("x = 0.5", "x = 0.6")
    assert readme.check(edited, SUMMARY) == ["block 'Beta' differs from results/summary.md"]


def test_unknown_section_is_left_alone():
    text = "<!-- BEGIN summary:Gamma -->\nkeep\n<!-- END summary:Gamma -->\n"
    new, missing = readme.sync(text, SUMMARY)
    assert new == text and missing == ["Gamma"]
