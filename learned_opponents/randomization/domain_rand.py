"""Domain-randomization sampler (task 7.1).

At every episode reset the randomized variant draws, per environment, the
robot mass scale, wheel friction scale, motor strength scale, ray-noise sigma,
action latency, and observation latency from the ranges in
``config/domain_rand.yaml``. The baseline variant returns the nominal values
and consumes no randomness.

All draws come from an explicit ``numpy.random.Generator`` derived from the
run seed (``make_rng``), in a fixed parameter order, so a given
``(seed, config, variant)`` reproduces the draws exactly. Imports only numpy
and the standard library.

Demo: ``python -m learned_opponents.randomization.domain_rand``
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

PARAMS = ("mass_scale", "friction_scale", "motor_strength_scale", "sensor_noise_sigma_m",
          "action_latency_steps", "observation_latency_steps")
INTEGER_PARAMS = frozenset({"action_latency_steps", "observation_latency_steps"})
VARIANTS = ("baseline", "randomized")


@dataclass(frozen=True)
class ParamRange:
    nominal: float
    low: float
    high: float
    integer: bool = False

    def contains(self, x) -> np.ndarray:
        return (np.asarray(x) >= self.low) & (np.asarray(x) <= self.high)


def parse_ranges(cfg: Mapping[str, Any]) -> dict[str, ParamRange]:
    """Validate a parsed domain_rand.yaml; errors name the offending parameter."""
    out = {}
    for name in PARAMS:
        if name not in cfg:
            raise ValueError(f"domain_rand: missing parameter '{name}'")
        entry = cfg[name]
        try:
            nominal = entry["nominal"]
            low, high = entry["range"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"domain_rand: '{name}' needs 'nominal' and 'range: [min, max]'") from None
        integer = name in INTEGER_PARAMS
        if integer and not all(float(v).is_integer() for v in (nominal, low, high)):
            raise ValueError(f"domain_rand: '{name}' must use whole numbers of steps")
        nominal, low, high = float(nominal), float(low), float(high)
        if not low <= nominal <= high:
            raise ValueError(f"domain_rand: '{name}' needs min <= nominal <= max, got "
                             f"{low} <= {nominal} <= {high}")
        if low < 0:
            raise ValueError(f"domain_rand: '{name}' must be >= 0, got min {low}")
        if name.endswith("_scale") and low <= 0:
            raise ValueError(f"domain_rand: '{name}' must be > 0")
        out[name] = ParamRange(nominal, low, high, integer)
    extra = set(cfg) - set(PARAMS)
    if extra:
        raise ValueError(f"domain_rand: unknown parameters {sorted(extra)}")
    return out


def make_rng(seed: int, stream: str) -> np.random.Generator:
    """Independent, reproducible generator for one named use of a run seed."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(stream.encode())]))


def nominal_params(n: int, ranges: Mapping[str, ParamRange]) -> dict[str, np.ndarray]:
    return {k: np.full(n, r.nominal, dtype=np.int64 if r.integer else np.float64)
            for k, r in ranges.items()}


def sample_params(rng: np.random.Generator, n: int, ranges: Mapping[str, ParamRange],
                  variant: str) -> dict[str, np.ndarray]:
    """Draw ``n`` episodes of parameters. Baseline returns nominal values."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "baseline":
        return nominal_params(n, ranges)
    out = {}
    for name in PARAMS:          # fixed order keeps the stream reproducible
        r = ranges[name]
        if r.integer:
            out[name] = rng.integers(int(r.low), int(r.high) + 1, size=n, dtype=np.int64)
        else:
            out[name] = rng.uniform(r.low, r.high, size=n)
    return out


def _demo() -> None:
    from learned_opponents.config import load_yaml

    ranges = parse_ranges(load_yaml("domain_rand.yaml"))
    a = sample_params(make_rng(0, "domain_rand"), 5, ranges, "randomized")
    b = sample_params(make_rng(0, "domain_rand"), 5, ranges, "randomized")
    base = sample_params(make_rng(0, "domain_rand"), 5, ranges, "baseline")
    print("same seed -> identical draws:", all(np.array_equal(a[k], b[k]) for k in PARAMS))
    print("all draws inside their ranges:", all(ranges[k].contains(a[k]).all() for k in PARAMS))
    print("baseline equals nominal:", all(np.all(base[k] == ranges[k].nominal) for k in PARAMS))
    for k in PARAMS:
        print(f"  {k:26s} {np.round(a[k], 3).tolist()}")


if __name__ == "__main__":
    _demo()
