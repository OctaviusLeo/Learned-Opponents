"""Domain-randomization sampler.

Samples physics/sensing parameters at episode reset from the configured ranges
in ``config/domain_rand.yaml``:

    - mass
    - wheel_friction
    - motor_strength
    - sensor_noise (sigma)
    - action_latency (steps)
    - observation_latency (steps)

Two variants:

    - RANDOMIZED: each parameter is drawn uniformly from its ``[min, max]``
      range (latency parameters are drawn as integers).
    - BASELINE: every parameter is left at its configured ``nominal`` value;
      no sampling is performed.

Reproducibility (Requirements 0.2, 0.3, 4.2): ALL draws derive from the run
seed via an explicit ``numpy.random.Generator``. The same ``(seed, config,
variant)`` reproduces the same draws exactly. The global numpy RNG state is
never touched.

Run as a script for a demo::

    python -m learned_opponents.randomization.domain_rand
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping

import numpy as np

# Parameters sampled as integers (latency in steps); the rest are continuous.
_INTEGER_PARAMS = frozenset({"action_latency_steps", "observation_latency_steps"})

# Canonical parameter order used by this sampler. Draws are consumed from the
# Generator in THIS order so a given (seed, config) is fully reproducible.
PARAM_ORDER = (
    "mass",
    "wheel_friction",
    "motor_strength",
    "sensor_noise_sigma",
    "action_latency_steps",
    "observation_latency_steps",
)

BASELINE = "baseline"
RANDOMIZED = "randomized"


@dataclass(frozen=True)
class SampledParams:
    """One episode's sampled (or nominal) physics/sensing parameters."""

    mass: float
    wheel_friction: float
    motor_strength: float
    sensor_noise_sigma: float
    action_latency_steps: int
    observation_latency_steps: int

    def as_dict(self) -> dict:
        return asdict(self)


def _nominal_params(config: Mapping) -> SampledParams:
    """Build the baseline (nominal) parameter set from the config."""
    values = {}
    for name in PARAM_ORDER:
        nominal = config[name]["nominal"]
        values[name] = int(nominal) if name in _INTEGER_PARAMS else float(nominal)
    return SampledParams(**values)


def _draw(rng: np.random.Generator, name: str, spec: Mapping):
    """Draw one parameter from its [min, max] range using ``rng``."""
    lo = spec["min"]
    hi = spec["max"]
    if name in _INTEGER_PARAMS:
        # integers() upper bound is exclusive -> +1 to include hi.
        return int(rng.integers(int(lo), int(hi) + 1))
    return float(rng.uniform(float(lo), float(hi)))


def sample(config: Mapping, seed: int, variant: str) -> SampledParams:
    """Sample the domain-randomization parameters for one episode.

    Args:
        config: parsed ``domain_rand.yaml`` mapping. Each parameter maps to a
            dict with ``nominal``, ``min``, and ``max``.
        seed: the run seed. All draws derive from this via a dedicated
            ``numpy.random.Generator``; the global RNG is not touched.
        variant: ``"baseline"`` (all nominal) or ``"randomized"`` (sample).

    Returns:
        A :class:`SampledParams` instance.

    Raises:
        ValueError: if ``variant`` is not recognized.
    """
    if variant == BASELINE:
        return _nominal_params(config)

    if variant != RANDOMIZED:
        raise ValueError(
            f"variant must be '{BASELINE}' or '{RANDOMIZED}', got {variant!r}"
        )

    rng = np.random.default_rng(seed)
    values = {}
    for name in PARAM_ORDER:
        values[name] = _draw(rng, name, config[name])
    return SampledParams(**values)


def _demo() -> None:
    """Task 7.1 demo: reproducibility, in-range draws, baseline == nominal."""
    # Self-contained config mirroring config/domain_rand.yaml so the demo runs
    # without reading files.
    config = {
        "mass": {"nominal": 2.0, "min": 1.5, "max": 2.5},
        "wheel_friction": {"nominal": 1.0, "min": 0.7, "max": 1.3},
        "motor_strength": {"nominal": 1.0, "min": 0.8, "max": 1.2},
        "sensor_noise_sigma": {"nominal": 0.0, "min": 0.0, "max": 0.05},
        "action_latency_steps": {"nominal": 0, "min": 0, "max": 2},
        "observation_latency_steps": {"nominal": 0, "min": 0, "max": 2},
    }
    seed = 42

    a = sample(config, seed, RANDOMIZED)
    b = sample(config, seed, RANDOMIZED)
    print("randomized draw a:", a.as_dict())
    print("randomized draw b:", b.as_dict())
    print("identical for fixed seed:", a == b)

    in_range = all(
        config[name]["min"] <= getattr(a, name) <= config[name]["max"]
        for name in PARAM_ORDER
    )
    print("all draws in configured range:", in_range)

    base = sample(config, seed, BASELINE)
    nominal_match = all(
        getattr(base, name) == config[name]["nominal"] for name in PARAM_ORDER
    )
    print("baseline draw:", base.as_dict())
    print("baseline == nominal:", nominal_match)


if __name__ == "__main__":
    _demo()
