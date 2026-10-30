"""Argument parsing of the GPU-free eval entry points: ``--help`` and ``--allow-dirty``."""

from __future__ import annotations

import pytest

from learned_opponents.eval import training_runs, transfer


@pytest.mark.parametrize("module", [training_runs, transfer])
def test_help_exits_zero_and_documents_allow_dirty(module, capsys):
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "--allow-dirty" in capsys.readouterr().out


def test_training_runs_passes_allow_dirty_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(training_runs, "collect", lambda allow_dirty: {"allow_dirty": allow_dirty})
    monkeypatch.setattr(training_runs, "write_json", lambda path, data: seen.update(data))
    assert training_runs.main([]) == 0 and seen == {"allow_dirty": False}
    assert training_runs.main(["--allow-dirty"]) == 0 and seen == {"allow_dirty": True}


def test_transfer_dispatches_commands_and_rejects_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(transfer, "episodes", lambda: calls.append(("episodes", None)) or 0)
    monkeypatch.setattr(transfer, "mujoco", lambda allow: calls.append(("mujoco", allow)) or 0)
    monkeypatch.setattr(transfer, "summarize", lambda allow: calls.append(("summarize", allow)) or 0)
    assert transfer.main(["episodes"]) == 0
    assert transfer.main(["mujoco", "--allow-dirty"]) == 0
    assert transfer.main(["summarize"]) == 0
    assert calls == [("episodes", None), ("mujoco", True), ("summarize", False)]
    with pytest.raises(SystemExit) as exc:
        transfer.main(["bogus"])
    assert exc.value.code == 2
