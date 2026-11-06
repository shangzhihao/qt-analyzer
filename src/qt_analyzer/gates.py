"""Gate algorithms: decide whether a comparison shows a regression or an improvement.

Every gate looks at the same `Comparison` (paired cases of a new state and its
baseline) and returns the scopes ("overall" or a family) it judges regressed or
improved.

Adding a gate: write a class with `judge(comparison) -> Verdict` and decorate it
with `@register_gate("name")`, in this module or in your own module listed in
`analyzer.toml` as `gate_modules = ["my_gates"]`. Keyword arguments come from
`[gates.<name>]` in `analyzer.toml`. Every registered gate takes part in
`qt-analyzer gates --example ...` and `--benchmark`; `gate = "<name>"` selects the
one that makes the analysis decisions.

- `zscore`: the analyzer's own test. Per scope, observed passes against those
  expected from the baseline, pooled variance, continuity correction; flags a
  scope whose z passes a threshold in either direction.
- `testimate_threshold`: the gate testimate uses today (`Threshold4StdDev` with
  `OmegaCubic` and `StdDevAssumeIndependency`), ported exactly: a weighted
  failure rate with weights ω = p³ of the reference rates, failing when it
  exceeds the reference rate plus 10 standard deviations. One total, no
  families, no improvement.
- `poisson_binomial`: testimate's `PoissonBinomialGate` (testimate
  `pipeline/extensions/poisson_binomial.py`), ported exactly: the total passes of
  all cases against the Poisson-binomial distribution of the baseline's pass
  rates, taken as known; fails when the total is in the lower tail (α = 0.05).
  It has no families and no improvement.
- `bayes`: beta-binomial posterior per case on both sides (Jeffreys prior); per
  scope, the probability that the mean pass rate changed by more than `delta`;
  flags when that probability reaches `probability`.
"""

from __future__ import annotations

import importlib
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Protocol

from qt_analyzer.stats import z_threshold

if TYPE_CHECKING:
    from qt_analyzer.analyse import CaseResult, Comparison

_NORMAL = NormalDist()


@dataclass
class Verdict:
    regressed: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)  # scope -> human-readable evidence


class Gate(Protocol):
    def judge(self, comparison: Comparison) -> Verdict: ...


_REGISTRY: dict[str, Callable[..., Gate]] = {}


def register_gate(name: str) -> Callable[[type], type]:
    """Class decorator: make a gate available under `name`."""

    def decorate(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"gate {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorate


def load_gate_modules(modules: Sequence[str]) -> None:
    """Import modules that register their own gates."""
    for module in modules:
        importlib.import_module(module)


def available_gates() -> list[str]:
    return sorted(_REGISTRY)


def make_gate(name: str, params: dict[str, Any] | None = None) -> Gate:
    if name not in _REGISTRY:
        raise KeyError(f"unknown gate {name!r}; available: {', '.join(available_gates())}")
    return _REGISTRY[name](**(params or {}))


# ---------------------------------------------------------------------------
# zscore (current)


@register_gate("zscore")
@dataclass(frozen=True)
class ZScoreGate:
    """The analyzer's scope test: a fixed `z` cut, or `alpha` split over the scopes."""

    alpha: float = 0.05
    z: float | None = None

    def judge(self, comparison: Comparison) -> Verdict:
        cut = (
            self.z
            if self.z is not None
            else z_threshold(self.alpha / (1 + len(comparison.families)))
        )
        verdict = Verdict()
        scopes = [("overall", comparison.overall)] + sorted(comparison.families.items())
        for scope, test in scopes:
            if not test.cases:
                continue
            verdict.detail[scope] = f"z = {test.z:+.2f} (cut ±{-cut:.2f})"
            if test.z <= cut:
                verdict.regressed.append(scope)
            elif test.z >= -cut:
                verdict.improved.append(scope)
        return verdict


# ---------------------------------------------------------------------------
# testimate's gate in use: Threshold4StdDev + OmegaCubic (exact port)

THRESHOLD_STD_DEV_MARGIN = 10  # testimate update_threshold.py
EPSILON = 2**-20  # testimate helper.non_zero floor


@dataclass(frozen=True)
class ThresholdResult:
    passed: bool
    y: float  # weighted failure rate of the new run
    y_reference: float
    threshold: float


def testimate_threshold(reference: Sequence[float], observed: Sequence[float]) -> ThresholdResult:
    """testimate: ω = p³; y = Σω(1-x)/Σω; pass if y <= y_ref + 10·σ.

    `reference` are the reference pass probabilities (testimate's p_short_term),
    `observed` the new run's pass fractions per case (testimate's x).
    """
    omega = [p**3 for p in reference]
    norm = max(sum(omega), EPSILON)  # testimate's non_zero
    y_ref = sum(w * (1 - p) for w, p in zip(omega, reference, strict=True)) / norm
    sigma = (
        math.sqrt(sum(w * w * p * (1 - p) for w, p in zip(omega, reference, strict=True))) / norm
    )
    y = sum(w * (1 - x) for w, x in zip(omega, observed, strict=True)) / norm
    threshold = y_ref + THRESHOLD_STD_DEV_MARGIN * sigma
    return ThresholdResult(y <= threshold, y, y_ref, threshold)


@register_gate("testimate_threshold")
@dataclass(frozen=True)
class TestimateThresholdGate:
    """testimate's gate in use, with the baseline rates as reference."""

    def judge(self, comparison: Comparison) -> Verdict:
        cases = [c for c in comparison.cases if c.judged_s and c.judged_b]
        verdict = Verdict()
        if not cases:
            return verdict
        r = testimate_threshold(
            [c.passed_b / c.judged_b for c in cases], [c.passed_s / c.judged_s for c in cases]
        )
        verdict.detail["overall"] = (
            f"weighted failure rate {r.y:.4f}, reference {r.y_reference:.4f}, "
            f"threshold {r.threshold:.4f}"
        )
        if not r.passed:
            verdict.regressed.append("overall")
        return verdict


# ---------------------------------------------------------------------------
# testimate's Poisson-binomial gate, implemented but not used (exact port)


@dataclass(frozen=True)
class Observation:
    probability: float
    pass_count: int
    run_count: int


@dataclass(frozen=True)
class PoissonBinomialResult:
    passed: bool
    pass_count: int
    run_count: int
    expected_passes: float
    minimum_accepted_passes: int
    score: float  # P(passes <= observed) under the baseline probabilities


def poisson_binomial(
    observations: Sequence[Observation], significance_level: float = 0.05
) -> PoissonBinomialResult:
    """Same computation as testimate's `PoissonBinomialGate.evaluate` (offset 0)."""
    mass = [1.0]
    for o in observations:
        for _ in range(o.run_count):
            mass.append(0.0)
            for passes in range(len(mass) - 1, 0, -1):
                mass[passes] = mass[passes] * (1 - o.probability) + mass[passes - 1] * o.probability
            mass[0] *= 1 - o.probability
    run_count = sum(o.run_count for o in observations)
    pass_count = sum(o.pass_count for o in observations)
    cumulative = 0.0
    minimum = run_count
    for passes, probability in enumerate(mass):
        cumulative += probability
        if cumulative >= significance_level or math.isclose(
            cumulative, significance_level, rel_tol=1e-12, abs_tol=1e-15
        ):
            minimum = passes
            break
    return PoissonBinomialResult(
        passed=pass_count >= minimum,
        pass_count=pass_count,
        run_count=run_count,
        expected_passes=sum(o.probability * o.run_count for o in observations),
        minimum_accepted_passes=minimum,
        score=min(1.0, max(0.0, sum(mass[: pass_count + 1]))),
    )


@register_gate("poisson_binomial")
@dataclass(frozen=True)
class PoissonBinomialGate:
    """testimate's gate: total passes of all cases, baseline rates taken as known."""

    significance_level: float = 0.05

    def judge(self, comparison: Comparison) -> Verdict:
        observations = [
            Observation(c.passed_b / c.judged_b, c.passed_s, c.judged_s)
            for c in comparison.cases
            if c.judged_s and c.judged_b
        ]
        verdict = Verdict()
        if not observations:
            return verdict
        r = poisson_binomial(observations, self.significance_level)
        verdict.detail["overall"] = (
            f"{r.pass_count}/{r.run_count} passes, expected {r.expected_passes:.1f}, "
            f"minimum accepted {r.minimum_accepted_passes}"
        )
        if not r.passed:
            verdict.regressed.append("overall")
        return verdict


# ---------------------------------------------------------------------------
# Bayesian gate: posterior predictive check against the baseline


@register_gate("bayes")
@dataclass(frozen=True)
class BayesGate:
    """Are the new passes surprising under the baseline's posterior predictive?

    Per case, the baseline gives a Beta posterior (Jeffreys prior) for the pass
    rate, so a 10-run baseline is not taken as the truth. Under "no change by
    more than `delta`", the new passes of a scope follow a sum of beta-binomial
    distributions with the rates shifted by `delta`; a scope is flagged when the
    observed total lies beyond that distribution's tail at `1 - probability`
    (normal approximation). Unlike comparing posterior means, this does not
    depend on how many runs each side has.
    """

    delta: float = 0.03
    probability: float = 0.99

    def judge(self, comparison: Comparison) -> Verdict:
        by_scope: dict[str, list[CaseResult]] = defaultdict(list)
        for case in comparison.cases:
            if case.judged_s and case.judged_b:
                by_scope["overall"].append(case)
                by_scope[case.family].append(case)
        cut = _NORMAL.inv_cdf(1 - self.probability)
        verdict = Verdict()
        for scope in ["overall"] + sorted(s for s in by_scope if s != "overall"):
            cases = by_scope.get(scope, [])
            if not cases:
                continue
            observed = expected = var = runs = 0.0
            for c in cases:
                a, b = c.passed_b + 0.5, c.judged_b - c.passed_b + 0.5
                m = a / (a + b)
                n = c.judged_s
                observed += c.passed_s
                expected += n * m
                var += n * m * (1 - m) * (a + b + n) / (a + b + 1)  # beta-binomial
                runs += n
            sd = math.sqrt(var) or 1.0
            margin = self.delta * runs
            z_drop = (observed - (expected - margin)) / sd
            z_gain = (observed - (expected + margin)) / sd
            change = (observed - expected) / runs
            verdict.detail[scope] = (
                f"change {change * 100:+.1f} pts vs predicted; "
                f"P(no drop > {self.delta * 100:.0f} pts) = {_NORMAL.cdf(z_drop):.3f}"
            )
            if z_drop <= cut:
                verdict.regressed.append(scope)
            elif z_gain >= -cut:
                verdict.improved.append(scope)
        return verdict
