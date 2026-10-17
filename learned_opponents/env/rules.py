"""Termination and reward rules (pure functions, ``xp`` = numpy or torch adapter).

Termination reasons are mutually exclusive, checked once per control step on
the state at the end of the step, in priority order collision > success >
timeout:

- collision: the robot footprint touches an obstacle or wall;
- success: the robot centre is within ``arrival_tolerance_m`` of the goal;
- timeout: neither, and the step count reached ``episode_limit``.

The reward is the configured weighted sum of its terms (Requirement 3.7):
``w_progress * progress + w_success * success - w_collision * collision - w_time``.
"""

from __future__ import annotations

from typing import Mapping

NONE, SUCCESS, COLLISION, TIMEOUT = 0, 1, 2, 3
REASON_NAMES = {NONE: "running", SUCCESS: "success", COLLISION: "collision", TIMEOUT: "timeout"}
REWARD_TERMS = ("progress", "success", "collision", "time")


def termination_reason_xp(xp, dist_to_goal, collided, step_count, arrival_tolerance_m: float,
                          episode_limit: int):
    """Integer reason per environment: NONE, SUCCESS, COLLISION, or TIMEOUT."""
    success = dist_to_goal <= arrival_tolerance_m
    timeout = step_count >= episode_limit
    reason = xp.where(timeout, TIMEOUT, NONE)
    reason = xp.where(success, SUCCESS, reason)
    return xp.where(collided, COLLISION, reason)


def reward_terms_xp(xp, prev_dist, dist, reason):
    """Unweighted reward terms; ``time`` is 1 every step (a per-step penalty)."""
    return {
        "progress": prev_dist - dist,
        "success": xp.where(reason == SUCCESS, 1.0, 0.0),
        "collision": xp.where(reason == COLLISION, 1.0, 0.0),
        "time": xp.where(reason >= NONE, 1.0, 0.0),
    }


# Sign of each term in the weighted sum: penalties are subtracted.
TERM_SIGNS = {"progress": 1.0, "success": 1.0, "collision": -1.0, "time": -1.0}


def combine_reward(terms: Mapping[str, object], weights: Mapping[str, float]):
    """``sum(sign * w_term * term)`` over the four configured terms."""
    total = 0.0
    for name in REWARD_TERMS:
        total = total + TERM_SIGNS[name] * float(weights[f"w_{name}"]) * terms[name]
    return total
