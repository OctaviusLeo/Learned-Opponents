"""Termination and reward rules (task 5.2): Properties 5 and 6, GPU-free."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from learned_opponents.env import rules

n_envs = st.integers(1, 64)


# Feature: learned-opponents, Property 5: Termination is mutually exclusive and reason-tagged
# (exactly one reason per ended episode; success only within the arrival tolerance; timeout only
# at the configured limit).
@settings(max_examples=200, deadline=None)
@given(data=st.data(), n=n_envs, tol=st.floats(0.01, 2.0), limit=st.integers(1, 1000))
def test_property_5_termination_exclusive_and_tagged(data, n, tol, limit):
    dist = data.draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 20.0)))
    collided = data.draw(hnp.arrays(np.bool_, n))
    steps = data.draw(hnp.arrays(np.int64, n, elements=st.integers(0, limit + 5)))
    reason = rules.termination_reason_xp(np, dist, collided, steps, tol, limit)

    assert reason.shape == (n,)
    assert set(np.unique(reason)) <= {rules.NONE, rules.SUCCESS, rules.COLLISION, rules.TIMEOUT}
    ended = reason != rules.NONE
    # Ended exactly when some condition holds; running otherwise.
    np.testing.assert_array_equal(ended, collided | (dist <= tol) | (steps >= limit))
    assert np.all(dist[reason == rules.SUCCESS] <= tol)
    assert not np.any(collided[reason == rules.SUCCESS])
    assert np.all(steps[reason == rules.TIMEOUT] >= limit)
    assert np.all(dist[reason == rules.TIMEOUT] > tol)
    assert not np.any(collided[reason == rules.TIMEOUT])
    np.testing.assert_array_equal(reason == rules.COLLISION, collided)


@settings(max_examples=100, deadline=None)
@given(data=st.data(), tol=st.floats(0.05, 1.0), limit=st.integers(1, 60))
def test_episode_reports_one_reason_at_first_end(data, tol, limit):
    """Rolling the rule over an episode: the first end step carries one reason."""
    length = limit + 5
    dist = data.draw(hnp.arrays(np.float64, length, elements=st.floats(0.0, 5.0)))
    hit = data.draw(hnp.arrays(np.bool_, length))
    reasons = [int(rules.termination_reason_xp(np, dist[t:t + 1], hit[t:t + 1], np.array([t + 1]),
                                               tol, limit)[0]) for t in range(length)]
    first = next(t for t, r in enumerate(reasons) if r != rules.NONE)
    assert first <= limit - 1                       # an episode always ends by the limit
    assert all(r == rules.NONE for r in reasons[:first])


weights_st = st.fixed_dictionaries({f"w_{t}": st.floats(-100.0, 100.0) for t in rules.REWARD_TERMS})


# Feature: learned-opponents, Property 6: Reward is a weighted sum of its configured terms
# (zeroing a weight removes exactly that term's contribution).
@settings(max_examples=200, deadline=None)
@given(data=st.data(), n=n_envs, weights=weights_st, term=st.sampled_from(rules.REWARD_TERMS))
def test_property_6_reward_is_weighted_sum(data, n, weights, term):
    prev = data.draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 12.0)))
    dist = data.draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 12.0)))
    reason = data.draw(hnp.arrays(np.int64, n, elements=st.sampled_from(
        [rules.NONE, rules.SUCCESS, rules.COLLISION, rules.TIMEOUT])))
    terms = rules.reward_terms_xp(np, prev, dist, reason)
    total = rules.combine_reward(terms, weights)
    expected = (weights["w_progress"] * (prev - dist) + weights["w_success"] * (reason == rules.SUCCESS)
                - weights["w_collision"] * (reason == rules.COLLISION) - weights["w_time"])
    np.testing.assert_allclose(total, expected, rtol=1e-9, atol=1e-9)

    zeroed = dict(weights, **{f"w_{term}": 0.0})
    removed = total - rules.combine_reward(terms, zeroed)
    np.testing.assert_allclose(removed, rules.TERM_SIGNS[term] * weights[f"w_{term}"] * terms[term],
                               rtol=1e-9, atol=1e-9)


def test_priority_collision_over_success_over_timeout():
    r = rules.termination_reason_xp(np, np.array([0.1, 0.1, 5.0, 5.0]), np.array([True, False, False, False]),
                                    np.array([400, 400, 400, 10]), 0.25, 400)
    assert r.tolist() == [rules.COLLISION, rules.SUCCESS, rules.TIMEOUT, rules.NONE]


def test_reward_terms_values():
    t = rules.reward_terms_xp(np, np.array([3.0, 2.0]), np.array([2.5, 2.0]),
                              np.array([rules.NONE, rules.COLLISION]))
    assert t["progress"].tolist() == [0.5, 0.0]
    assert t["collision"].tolist() == [0.0, 1.0] and t["success"].tolist() == [0.0, 0.0]
    assert t["time"].tolist() == [1.0, 1.0]
