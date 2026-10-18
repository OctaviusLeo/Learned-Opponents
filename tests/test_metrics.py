"""Confidence intervals (tasks 9.1, 9.2): Properties 9 and 10 plus known values. GPU-free."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learned_opponents.eval import metrics as M


@st.composite
def counts(draw):
    n = draw(st.integers(1, 100_000))
    s = draw(st.integers(0, n))
    c = draw(st.integers(0, n - s))
    return s, c, n


# Feature: learned-opponents, Property 9: Success/collision confidence intervals are valid proportions
# (rates in [0, 1]; each 95% Wilson interval contains its point estimate and lies within [0, 1]).
@settings(max_examples=300, deadline=None)
@given(c=counts())
def test_property_9_wilson_intervals_are_valid(c):
    s, col, n = c
    for k in (s, col):
        p = M.wilson(k, n)
        assert 0.0 <= p.rate <= 1.0
        assert 0.0 <= p.lo <= p.rate <= p.hi <= 1.0
        assert p.k == k and p.n == n
    assert M.wilson(s, n).rate + M.wilson(col, n).rate <= 1.0 + 1e-12


# Feature: learned-opponents, Property 10: Achieved time-to-goal CI brackets the mean.
@settings(max_examples=300, deadline=None)
@given(t=st.lists(st.floats(0.05, 20.0), min_size=1, max_size=500))
def test_property_10_time_to_goal_ci_brackets_mean(t):
    m = M.mean_t_ci(t)
    assert m.n == len(t)
    assert m.mean == pytest.approx(float(np.mean(t)))
    assert m.lo <= m.mean <= m.hi


def test_wilson_known_values():
    p = M.wilson(8, 10)
    assert (p.lo, p.hi) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9433, abs=1e-4))
    zero = M.wilson(0, 50)
    assert zero.lo == 0.0 and zero.hi == pytest.approx(0.0713, abs=1e-4)
    assert math.isnan(M.wilson(0, 0).rate)
    with pytest.raises(ValueError):
        M.wilson(5, 4)


@pytest.mark.parametrize("df,q", [(1, 12.7062047), (2, 4.30265273), (10, 2.22813885), (30, 2.04227246),
                                  (1000, 1.96233908)])
def test_t_quantile_matches_tables(df, q):
    assert M.t_quantile(0.975, df) == pytest.approx(q, abs=1e-6)
    assert M.t_quantile(0.025, df) == pytest.approx(-q, abs=1e-6)


def test_betainc_edges_and_symmetry():
    assert M.betainc(2.0, 3.0, 0.0) == 0.0 and M.betainc(2.0, 3.0, 1.0) == 1.0
    assert M.betainc(2.5, 2.5, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert M.betainc(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-12)     # uniform CDF


def test_mean_t_ci_example_and_degenerate_cases():
    m = M.mean_t_ci([1.0, 2.0, 3.0, 4.0])
    half = M.t_quantile(0.975, 3) * np.std([1, 2, 3, 4], ddof=1) / 2.0
    assert (m.mean, m.lo, m.hi) == (2.5, pytest.approx(2.5 - half), pytest.approx(2.5 + half))
    one = M.mean_t_ci([3.0])
    assert one.mean == 3.0 and one.lo == -math.inf and one.hi == math.inf
    assert math.isnan(M.mean_t_ci([]).mean)


def test_pooling_sums_episodes_across_seeds():
    p = M.pooled([800, 820, 790], [1024, 1024, 1024])
    assert p.k == 2410 and p.n == 3072
    assert p.rate == pytest.approx(2410 / 3072)


def test_formatting():
    assert M.fmt_rate(M.wilson(8, 10)) == "0.800 [0.490, 0.943]"
    assert M.fmt_rate(M.wilson(0, 0)) == "n/a"
    assert M.fmt_mean(M.mean_t_ci([3.0])) == "3.00 s (n=1, no interval)"


@pytest.mark.parametrize("module", ["learned_opponents.eval.metrics", "learned_opponents.eval.summary"])
def test_ci_modules_import_only_numpy_and_stdlib(module):
    import ast
    import importlib
    import sys

    src = open(importlib.import_module(module).__file__, encoding="utf-8").read()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names) | {"numpy", "__future__", "learned_opponents"}
