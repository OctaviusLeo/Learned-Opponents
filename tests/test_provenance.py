"""Commit and tree-state helpers (GPU-free)."""

from __future__ import annotations

from learned_opponents import provenance


def test_lo_commit_environment_variable_wins(monkeypatch):
    monkeypatch.setenv("LO_COMMIT", "a" * 40 + "\n")
    assert provenance.repo_commit() == "a" * 40


def test_dirty_filter_allows_only_generated_paths():
    porcelain = (" M results/summary.md\n?? runs/logs/x.log\n M docs/ENVIRONMENT.md\n M README.md\n"
                 " M learned_opponents/env/nav_env.py\n?? config/new.yaml\n")
    assert provenance.filter_dirty(porcelain) == ["learned_opponents/env/nav_env.py", "config/new.yaml"]
    assert provenance.filter_dirty("") == []
