"""Transfer-gap computation (task 17.2): Property 14 plus the summary table. GPU-free."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.eval import summary
from learned_opponents.eval import transfer_stats as T

REASONS = st.sampled_from(["success", "collision", "timeout"])


@st.composite
def transfer_data(draw):
    seeds = [0, 1, 2]
    n = draw(st.integers(1, 60))
    data = {"seeds": seeds, "variants": ["baseline", "randomized"], "mujoco": {}, "game": {}}
    for v in data["variants"]:
        for s in seeds:
            data["mujoco"][f"{v}_seed{s}"] = draw(st.lists(REASONS, min_size=n, max_size=n))
            data["game"][f"{v}_seed{s}"] = draw(st.lists(REASONS, min_size=n, max_size=n))
    return data


# Feature: learned-opponents, Property 14: Transfer gap equals the measured difference
# (MuJoCo Success_Rate minus in-game Success_Rate on the same episodes, with no adjustment).
@settings(max_examples=200, deadline=None)
@given(data=transfer_data())
def test_property_14_gap_is_the_measured_difference(data):
    for row in T.transfer_rows(data):
        v = row["variant"]
        mj = [r for s in data["seeds"] for r in data["mujoco"][f"{v}_seed{s}"]]
        gm = [r for s in data["seeds"] for r in data["game"][f"{v}_seed{s}"]]
        assert row["n"] == len(mj) == len(gm)
        assert row["mujoco"].rate == mj.count("success") / len(mj)
        assert row["game"].rate == gm.count("success") / len(gm)
        assert row["gap"] == row["mujoco"].rate - row["game"].rate
        assert row["gap"] == T.transfer_gap(row["mujoco"].rate, row["game"].rate)


def test_unequal_episode_counts_are_rejected():
    data = {"seeds": [0], "variants": ["baseline"], "mujoco": {"baseline_seed0": ["success"]},
            "game": {"baseline_seed0": ["success", "timeout"]}}
    with pytest.raises(ValueError):
        T.transfer_rows(data)


def test_transfer_section_in_summary(tmp_path):
    data = {"kind": "transfer", "command": "powershell -File scripts/run_transfer.ps1", "commit": "c" * 40,
            "layout_seed": 50505, "n_layouts": 2, "seeds": [0, 1, 2], "variants": ["baseline", "randomized"],
            "mujoco": {f"{v}_seed{s}": ["success", "success"] for v in ("baseline", "randomized") for s in (0, 1, 2)},
            "game": {f"{v}_seed{s}": ["success", "collision"] for v in ("baseline", "randomized") for s in (0, 1, 2)}}
    (tmp_path / "transfer.json").write_text(json.dumps(data))
    text = summary.render(tmp_path)
    assert "## Transfer check" in text and "+0.500" in text and "c" * 40 in text
    assert "| +0.500 | 3 / 0 | 0.25 |" in text            # 3 MuJoCo-only successes per variant; 2 * 0.5**3


@settings(max_examples=200, deadline=None)
@given(data=transfer_data())
def test_discordant_counts_account_for_the_gap(data):
    for row in T.transfer_rows(data):
        assert row["mujoco_only"] - row["game_only"] == round(row["gap"] * row["n"])
        assert row["mujoco_only"] + row["game_only"] <= row["n"]
        assert 0.0 <= row["mcnemar_p"] <= 1.0
        assert row["mcnemar_p"] == T.mcnemar_exact_p(row["game_only"], row["mujoco_only"])   # symmetric


def test_mcnemar_exact_known_values():
    assert T.mcnemar_exact_p(0, 0) == 1.0
    assert T.mcnemar_exact_p(3, 3) == 1.0
    assert T.mcnemar_exact_p(5, 0) == pytest.approx(2 / 32)            # 2 * 0.5**5
    assert T.mcnemar_exact_p(2, 0) == pytest.approx(0.5)               # 2 * 0.5**2
    assert T.mcnemar_exact_p(10, 2) == pytest.approx(2 * 79 / 4096)    # 2 * P(X <= 2), X ~ Bin(12, 0.5)
