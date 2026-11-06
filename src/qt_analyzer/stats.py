"""Statistics for comparing pass counts between two states."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()

# Continuity correction: the difference is moved half a pass towards zero. At
# the small per-scope levels a monthly budget needs (around 1e-4), the normal
# approximation of a few discrete cases is otherwise too optimistic.
CONTINUITY = 0.5


def continuity(difference: float) -> float:
    return math.copysign(max(abs(difference) - CONTINUITY, 0.0), difference)


def jeffreys(passed: int, n: int) -> float:
    """Pass rate smoothed with a Jeffreys prior, never exactly 0 or 1."""
    return (passed + 0.5) / (n + 1)


@dataclass(frozen=True)
class Pair:
    """One comparable case: passes and executions in the new state and the baseline."""

    passed_s: int
    n_s: int
    passed_b: int
    n_b: int


def pooled_rate(p: Pair) -> float:
    """Pass rate of both samples together, smoothed so it is never 0 or 1."""
    return (p.passed_s + p.passed_b + 1) / (p.n_s + p.n_b + 2)


@dataclass(frozen=True)
class ScopeTest:
    cases: int
    rate_s: float  # mean observed pass rate in the new state
    rate_b: float  # mean baseline pass rate
    z: float

    @property
    def delta(self) -> float:
        return self.rate_s - self.rate_b

    @property
    def p_drop(self) -> float:
        """One-sided p-value for a drop."""
        return _NORMAL.cdf(self.z)


def scope_test(pairs: Iterable[Pair]) -> ScopeTest:
    """Compare observed passes with those expected under "no change".

    The expectation uses the raw baseline rate (unbiased). The variance uses the
    rate of both samples pooled, as "no change" assumes, smoothed (+1/+2) so it
    stays positive for one run per case and for rates at 0 or 100 %. A variance
    from the baseline alone collapses when a short baseline happens to look
    perfect and makes small families raise false alarms.
    """
    num = var = 0.0
    rate_s = rate_b = 0.0
    cases = 0
    for p in pairs:
        if p.n_s == 0 or p.n_b == 0:
            continue
        raw_b = p.passed_b / p.n_b
        pooled = pooled_rate(p)
        num += p.passed_s - p.n_s * raw_b
        var += p.n_s * pooled * (1 - pooled) * (1 + p.n_s / p.n_b)
        rate_s += p.passed_s / p.n_s
        rate_b += raw_b
        cases += 1
    if cases == 0:
        return ScopeTest(0, math.nan, math.nan, 0.0)
    return ScopeTest(cases, rate_s / cases, rate_b / cases, continuity(num) / math.sqrt(var))


def z_threshold(alpha: float) -> float:
    """Negative z below which a one-sided drop test rejects at level alpha."""
    return _NORMAL.inv_cdf(alpha)


def fisher_drop(passed_s: int, failed_s: int, passed_b: int, failed_b: int) -> float:
    """One-sided Fisher exact p-value that the new state passes less often."""
    n_s, n_b = passed_s + failed_s, passed_b + failed_b
    total_pass = passed_s + passed_b
    n = n_s + n_b
    if n_s == 0 or n_b == 0:
        return 1.0
    denominator = math.comb(n, n_s)
    low = max(0, total_pass - n_b)
    tail = sum(
        math.comb(total_pass, k) * math.comb(n - total_pass, n_s - k)
        for k in range(low, passed_s + 1)
    )
    return min(1.0, tail / denominator)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """q-values controlling the false discovery rate."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        q[i] = running
    return q


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a pass rate."""
    if n == 0:
        return (math.nan, math.nan)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
