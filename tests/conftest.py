"""Shared fixtures. GPU fixtures build one environment per session (CUDA graph capture is slow)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def task():
    from learned_opponents.config import load_task

    return load_task()


@pytest.fixture(scope="session")
def dr_ranges():
    from learned_opponents.config import load_yaml
    from learned_opponents.randomization.domain_rand import parse_ranges

    return parse_ranges(load_yaml("domain_rand.yaml"))


@pytest.fixture(scope="session")
def gpu_env(task):
    """A 128-world baseline env without auto-reset, for layout replays."""
    pytest.importorskip("mjlab")
    from learned_opponents.env.nav_env import NavEnv

    return NavEnv(task, num_envs=128, seed=0, bank_size=128, auto_reset=False)
