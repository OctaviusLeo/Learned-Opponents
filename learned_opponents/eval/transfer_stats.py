"""Transfer-gap arithmetic (numpy/stdlib only; Property 14).

Transfer_Gap = MuJoCo Success_Rate - in-game Success_Rate, computed per variant over the
same (training seed, layout) episodes in both engines and reported as measured. Because the
episodes are paired, the discordant counts (success in one engine only) and an exact McNemar
test show how many episodes the gap actually rests on.
"""

from __future__ import annotations

from math import comb
from typing import Any, Mapping

from learned_opponents.eval import metrics as M


def transfer_gap(mujoco_rate: float, game_rate: float) -> float:
    return float(mujoco_rate) - float(game_rate)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` and ``c`` discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def variant_counts(data: Mapping[str, Any], variant: str) -> dict[str, int]:
    """Pool successes over the variant's seeds; each engine contributes the same episodes."""
    out = {"n": 0, "mujoco_success": 0, "game_success": 0, "game_collision": 0, "game_timeout": 0,
           "mujoco_only": 0, "game_only": 0}
    for seed in data["seeds"]:
        name = f"{variant}_seed{seed}"
        mj = data["mujoco"][name]
        gm = data["game"][name]
        if len(mj) != len(gm):
            raise ValueError(f"{name}: {len(mj)} MuJoCo episodes vs {len(gm)} in-game episodes")
        out["n"] += len(mj)
        out["mujoco_success"] += sum(r == "success" for r in mj)
        out["game_success"] += sum(r == "success" for r in gm)
        out["game_collision"] += sum(r == "collision" for r in gm)
        out["game_timeout"] += sum(r == "timeout" for r in gm)
        out["mujoco_only"] += sum(a == "success" and b != "success" for a, b in zip(mj, gm))
        out["game_only"] += sum(a != "success" and b == "success" for a, b in zip(mj, gm))
    return out


def transfer_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in data["variants"]:
        c = variant_counts(data, variant)
        mj = M.wilson(c["mujoco_success"], c["n"])
        gm = M.wilson(c["game_success"], c["n"])
        rows.append({"variant": variant, "n": c["n"], "mujoco": mj, "game": gm,
                     "game_collision": M.wilson(c["game_collision"], c["n"]),
                     "gap": transfer_gap(mj.rate, gm.rate),
                     "mujoco_only": c["mujoco_only"], "game_only": c["game_only"],
                     "mcnemar_p": mcnemar_exact_p(c["mujoco_only"], c["game_only"])})
    return rows
