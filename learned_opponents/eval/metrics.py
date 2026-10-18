"""Rates and means with 95% confidence intervals (task 9.1).

- Success_Rate and Collision_Rate are binomial proportions; their intervals
  are Wilson score intervals, which stay inside [0, 1] and behave near 0 and 1.
- Mean Time_To_Goal over successful episodes uses a Student-t interval on the
  mean. The t quantile is computed here (regularized incomplete beta by
  continued fraction, then bisection), so this module needs only numpy and
  the standard library and CI can test it.
- Seeds are aggregated by pooling episodes: the pooled interval covers episode
  sampling, not seed-to-seed training variation, so per-seed rates are always
  reported next to it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054          # standard normal 0.975 quantile


@dataclass(frozen=True)
class Proportion:
    k: int
    n: int
    rate: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeanCI:
    n: int
    mean: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)


def wilson(k: int, n: int, z: float = Z95) -> Proportion:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return Proportion(0, 0, float("nan"), float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # Guard rounding so the interval always contains the point estimate.
    return Proportion(k, n, p, min(lo, p), max(hi, p))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection (|error| < 1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def mean_t_ci(values: Iterable[float], confidence: float = 0.95) -> MeanCI:
    """Mean with a Student-t interval. One value gives an unbounded interval; none gives NaN."""
    x = np.asarray(list(values), dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return MeanCI(0, float("nan"), float("nan"), float("nan"))
    m = float(x.mean())
    if n == 1:
        return MeanCI(1, m, float("-inf"), float("inf"))
    s = float(x.std(ddof=1))
    half = t_quantile(0.5 + confidence / 2.0, n - 1) * s / math.sqrt(n)
    return MeanCI(n, m, min(m - half, m), max(m + half, m))


def pooled(counts: Sequence[int], totals: Sequence[int]) -> Proportion:
    """Pool episodes across seeds and return the Wilson interval of the pooled rate."""
    return wilson(int(sum(counts)), int(sum(totals)))


def fmt_rate(p: Proportion) -> str:
    if p.n == 0:
        return "n/a"
    return f"{p.rate:.3f} [{p.lo:.3f}, {p.hi:.3f}]"


def fmt_mean(m: MeanCI, unit: str = "s") -> str:
    if m.n == 0:
        return "n/a"
    if not math.isfinite(m.lo):
        return f"{m.mean:.2f} {unit} (n=1, no interval)"
    return f"{m.mean:.2f} [{m.lo:.2f}, {m.hi:.2f}] {unit}"
