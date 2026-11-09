"""Candidate gates under exploration, registered like any user gate.

- `higher_criticism`: Donoho-Jin higher criticism over the per-case one-sided
  p-values (Fisher exact). Designed for sparse changes: a few cases among many
  shift, too few to move a total, too weak to stand out alone.
- `min_p_cases`: the smallest per-case p-value against a Bonferroni bound over
  all cases. Flags when any single case collapses.
- `zscore_split_hc`: `zscore_split` with part of its budget moved to a higher
  criticism test, to cover sparse changes as well.
- `zscore_strata`: `zscore_split` plus a scope of the reliable cases, taking over
  the strength of testimate's p^3 weighting without its blind spot. With
  `dispersion` (φ per kind of scope, estimated by the analysis) each z is
  divided by √φ, for runs that vary more than independent runs.
- `likelihood_ratio`: `zscore_strata`'s scopes and budget, but each scope is
  tested with a likelihood ratio against a drop of `delta` in every case's own
  pass rate, so a failure counts by how surprising it is for that case. Meant
  for 1-run comparisons, where a plain count wastes the evidence.
- `ci_gate`: the PR check without re-runs: `likelihood_ratio` tested against a
  margin (drops smaller than `margin` points pass), with a strict block level and
  a looser warn level.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

from qt_det.gates import Verdict, ZScoreSplitGate, register_gate
from qt_det.stats import fisher_drop

if TYPE_CHECKING:
    from qt_det.analyse import CaseResult, Comparison

_NORMAL = NormalDist()


def _drop_p(c: CaseResult) -> float:
    return c.p  # P(passes <= observed) given both samples (one-sided Fisher exact)


def _gain_p(c: CaseResult) -> float:
    failed_s, failed_b = c.judged_s - c.passed_s, c.judged_b - c.passed_b
    return fisher_drop(failed_s, c.passed_s, failed_b, c.passed_b)


def higher_criticism(p_values: list[float], fraction: float = 0.5) -> float:
    """HC+ = max over the smallest `fraction` of p-values (with p >= 1/k) of
    sqrt(k) (i/k - p_(i)) / sqrt(p_(i) (1 - p_(i)))."""
    k = len(p_values)
    if k == 0:
        return 0.0
    best = 0.0
    for i, p in enumerate(sorted(p_values)[: max(1, int(fraction * k))], start=1):
        if p < 1 / k or p >= 1:
            continue
        best = max(best, math.sqrt(k) * (i / k - p) / math.sqrt(p * (1 - p)))
    return best


def _hc_threshold(level: float) -> float:
    """A monotone map from a level to an HC threshold (exact calibration is left
    to the benchmark's equal-false-alarm mode)."""
    return -2 * _NORMAL.inv_cdf(level)


def _cases(comparison: Comparison) -> list[CaseResult]:
    return [c for c in comparison.cases if c.judged_s and c.judged_b]


@register_gate("higher_criticism")
@dataclass(frozen=True)
class HigherCriticismGate:
    threshold: float = 4.0

    def tuned(self, level: float) -> HigherCriticismGate:
        return HigherCriticismGate(_hc_threshold(level))

    def judge(self, comparison: Comparison) -> Verdict:
        cases = _cases(comparison)
        drop = higher_criticism([_drop_p(c) for c in cases])
        gain = higher_criticism([_gain_p(c) for c in cases])
        verdict = Verdict(detail={"overall": f"HC drop {drop:.2f}, gain {gain:.2f} "
                                             f"(threshold {self.threshold:.2f})"})  # fmt: skip
        if drop >= self.threshold:
            verdict.regressed.append("overall")
        elif gain >= self.threshold:
            verdict.improved.append("overall")
        return verdict


@register_gate("min_p_cases")
@dataclass(frozen=True)
class MinPCasesGate:
    alpha: float = 0.05

    def tuned(self, level: float) -> MinPCasesGate:
        return MinPCasesGate(level)

    def judge(self, comparison: Comparison) -> Verdict:
        cases = _cases(comparison)
        verdict = Verdict()
        if not cases:
            return verdict
        bound = self.alpha / len(cases)
        worst = min(cases, key=_drop_p)
        best = min(cases, key=_gain_p)
        verdict.detail["overall"] = (
            f"min drop p {_drop_p(worst):.2e} ({worst.hrid}), bound {bound:.2e}"
        )
        if _drop_p(worst) <= bound:
            verdict.regressed.append("overall")
        elif _gain_p(best) <= bound:
            verdict.improved.append("overall")
        return verdict


@register_gate("zscore_split_hc")
@dataclass(frozen=True)
class ZScoreSplitHCGate:
    """`zscore_split` on 80 % of the budget, higher criticism on 20 %."""

    alpha: float = 0.05
    hc_share: float = 0.2

    def tuned(self, level: float) -> ZScoreSplitHCGate:
        return ZScoreSplitHCGate(level, self.hc_share)

    def judge(self, comparison: Comparison) -> Verdict:
        split = ZScoreSplitGate().tuned(self.alpha * (1 - self.hc_share)).judge(comparison)
        hc = HigherCriticismGate().tuned(self.alpha * self.hc_share).judge(comparison)
        regressed = split.regressed + [s for s in hc.regressed if s not in split.regressed]
        improved = split.improved + [s for s in hc.improved if s not in split.improved]
        detail = split.detail | {"higher criticism": hc.detail.get("overall", "")}
        return Verdict(regressed, improved, detail)


def _reliable(c: CaseResult) -> bool:
    from qt_det.gates import pooled_rate

    return pooled_rate(c) >= 0.95


DISPERSION_KINDS = ("overall", "reliable cases", "flaky cases", "topics")


def strata_scopes(
    comparison: Comparison, shares: tuple[float, float, float]
) -> list[tuple[str, str, float, float]]:
    """(scope, kind, z, share of the budget) for zscore_strata's scopes."""
    from qt_det.gates import _flaky
    from qt_det.stats import Pair, scope_test

    total, reliable_share, flaky_share = shares
    topics = sorted((n, t) for n, t in comparison.families.items() if t.cases)
    topic_share = (1 - total - reliable_share - flaky_share) / max(len(topics), 1)

    def pairs(keep):
        return [Pair(c.passed_s, c.judged_s, c.passed_b, c.judged_b)
                for c in _cases(comparison) if keep(c)]  # fmt: skip

    scopes = [("overall", "overall", comparison.overall.z, total)]
    for name, keep, share in (("reliable cases", _reliable, reliable_share),
                              ("flaky cases", _flaky, flaky_share)):  # fmt: skip
        selected = pairs(keep)
        if selected:
            scopes.append((name, name, scope_test(selected).z, share))
    scopes += [(name, "topics", t.z, topic_share) for name, t in topics]
    return scopes


@register_gate("zscore_strata")
@dataclass(frozen=True)
class ZScoreStrataGate:
    """`zscore_split` plus a scope of the reliable cases (baseline >= 95 %).

    Budget for regressions: total 35 %, reliable cases 25 %, flaky cases 15 %,
    topics the rest; improvements get `improvement_alpha` split the same way.
    The reliable scope takes over the strength of testimate's p^3 weighting
    (reliable cases breaking) without its blindness to flaky cases.

    `dispersion` ({kind: phi}, kinds as in DISPERSION_KINDS) inflates the
    variance for night-to-night variation that the binomial model ignores:
    z is divided by sqrt(phi). The analysis estimates it from the history.
    """

    alpha: float = 0.05
    improvement_alpha: float = 0.01
    shares: tuple[float, float, float] = (0.35, 0.25, 0.15)  # total, reliable, flaky
    dispersion: dict[str, float] | None = None

    def tuned(self, level: float) -> ZScoreStrataGate:
        return ZScoreStrataGate(level, level * self.improvement_alpha / self.alpha, self.shares,
                                self.dispersion)  # fmt: skip

    def judge(self, comparison: Comparison) -> Verdict:
        from qt_det.stats import z_threshold

        verdict = Verdict()
        if not comparison.overall.cases:
            return verdict
        phi = self.dispersion or {}
        for scope, kind, raw_z, share in strata_scopes(comparison, tuple(self.shares)):
            z = raw_z / math.sqrt(max(phi.get(kind, 1.0), 1.0))
            drop_cut = z_threshold(self.alpha * share)
            gain_cut = -z_threshold(self.improvement_alpha * share)
            verdict.detail[scope] = f"z = {z:+.2f} (drop at {drop_cut:.2f})"
            if z <= drop_cut:
                verdict.regressed.append(scope)
            elif z >= gain_cut:
                verdict.improved.append(scope)
        return verdict


def _lr_weight(c: CaseResult, delta: float, gain: bool, cap: float) -> float:
    """How much one pass moves the log likelihood ratio of "pass rate fell by the
    factor 1 - delta" (or, with `gain`, "failure rate fell by it") against
    "unchanged": log(q/p) - log((1-q)/(1-p)), as a positive number. Large for a
    case that nearly always passes, small for a flaky one. p is the case's rate
    over both sides (choosing by one side biases the test), kept within
    [1 - cap, cap] so that no single case dominates its scope."""
    p = (c.passed_s + c.passed_b) / (c.judged_s + c.judged_b)
    p = min(max(p, 1 - cap), cap)
    q = 1 - (1 - p) * (1 - delta) if gain else p * (1 - delta)
    return abs(math.log(q / p) - math.log((1 - q) / (1 - p)))


def _lr_z(cases: list[CaseResult], delta: float, gain: bool, cap: float) -> float:
    """Weighted sum of each case's new passes minus those expected given both
    sides' total (hypergeometric, so exactly zero mean under "unchanged", also for
    cases that always pass), standardised. Negative = fewer passes than expected."""
    total = var = largest = 0.0
    for c in cases:
        n, both = c.judged_s, c.judged_s + c.judged_b
        passes = c.passed_s + c.passed_b
        if both < 2 or passes in (0, both):
            continue
        w = _lr_weight(c, delta, gain, cap)
        total += w * (c.passed_s - n * passes / both)
        var += w * w * n * c.judged_b * passes * (both - passes) / (both * both * (both - 1))
        largest = max(largest, w)
    if var <= 0:
        return 0.0
    # Continuity correction, as in the plain test, at the heaviest case: one pass
    # of it moves the sum in steps of its weight.
    correction = 0.5 * largest
    return math.copysign(max(abs(total) - correction, 0.0), total) / math.sqrt(var)


def _margin_z(
    cases: list[CaseResult], delta: float, cap: float, margin: float, weighted: bool
) -> float:
    """Did the scope drop by more than `margin` points? Negative = yes.

    Each case's new passes minus those expected after a drop of exactly `margin`
    (or of its whole rate, if lower), weighted like likelihood_ratio (or all 1
    without `weighted`). The variance is that of this boundary, not of
    "unchanged": the new run at the lowered rate, plus the baseline's own
    sampling error, with the baseline rate smoothed so a case that always passed
    still varies."""
    total = var = largest = 0.0
    for c in cases:
        n, n_b = c.judged_s, c.judged_b
        if not n or not n_b:
            continue
        raw = c.passed_b / n_b
        p = (c.passed_b + 0.5) / (n_b + 1)
        q = max(p - min(margin, raw), 0.0)
        w = _lr_weight(c, delta, False, cap) if weighted else 1.0
        total += w * (c.passed_s - n * (raw - min(margin, raw)))
        var += w * w * (n * q * (1 - q) + n * n * p * (1 - p) / n_b)
        largest = max(largest, w)
    if var <= 0:
        return 0.0
    correction = 0.5 * largest
    return math.copysign(max(abs(total) - correction, 0.0), total) / math.sqrt(var)


def _baseline_rate(c: CaseResult) -> float:
    return (c.passed_b + 0.5) / (c.judged_b + 1)


def _lr_scopes(
    cases: list[CaseResult],
    shares: tuple[float, float, float],
    by_baseline: bool = False,
) -> list[tuple[str, str, list[CaseResult], float]]:
    """(scope, kind, cases, share of the budget): zscore_strata's scopes.

    Reliable and flaky cases are chosen by the rate over both sides, or with
    `by_baseline` by the baseline's smoothed rate alone. With one new run against
    a short baseline the first lets the new run decide membership: a case that
    passed 10/10 and fails once (10/11) becomes "flaky", so the flaky cases fill
    with reliable cases that just failed."""
    from qt_det.gates import _flaky

    def reliable(c: CaseResult) -> bool:
        return _baseline_rate(c) >= 0.95 if by_baseline else _reliable(c)

    def flaky(c: CaseResult) -> bool:
        return 0.05 < _baseline_rate(c) < 0.95 if by_baseline else _flaky(c)

    total, reliable_share, flaky_share = shares
    topics = sorted({c.family for c in cases})
    topic_share = (1 - total - reliable_share - flaky_share) / max(len(topics), 1)
    scopes = [("overall", "overall", cases, total)]
    for name, keep, share in (("reliable cases", reliable, reliable_share),
                              ("flaky cases", flaky, flaky_share)):  # fmt: skip
        selected = [c for c in cases if keep(c)]
        if selected:
            scopes.append((name, name, selected, share))
    return scopes + [
        (t, "topics", [c for c in cases if c.family == t], topic_share) for t in topics
    ]


@register_gate("likelihood_ratio")
@dataclass(frozen=True)
class LikelihoodRatioGate:
    """Per-case weighted test for 1-run comparisons, with zscore_strata's scopes.

    The plain count treats every failure alike. Under the alternative "each case's
    pass rate drops by the factor 1 - delta", the most powerful test (Neyman-
    Pearson) weighs each case by its log likelihood ratio per pass: large for a
    case that nearly always passes, small for a flaky one. The weighted sum is
    taken conditionally on each case's total over both sides, standardised and
    cut like zscore_strata's z. The flaky cases keep the plain count: a drop there
    is a few points off each case, which these weights would miss.
    """

    alpha: float = 0.05
    improvement_alpha: float = 0.01
    delta: float = 0.2
    cap: float = 0.95  # weights are those of a case passing at most this often
    shares: tuple[float, float, float] = (0.3, 0.2, 0.3)  # total, reliable, flaky
    dispersion: dict[str, float] | None = None

    def tuned(self, level: float) -> LikelihoodRatioGate:
        return LikelihoodRatioGate(level, level * self.improvement_alpha / self.alpha,
                                   self.delta, self.cap, self.shares, self.dispersion)  # fmt: skip

    def judge(self, comparison: Comparison) -> Verdict:
        from qt_det.stats import Pair, scope_test, z_threshold

        verdict = Verdict()
        cases = _cases(comparison)
        if not cases:
            return verdict
        phi = self.dispersion or {}
        for scope, kind, selected, share in _lr_scopes(cases, tuple(self.shares)):
            scale = math.sqrt(max(phi.get(kind, 1.0), 1.0))
            if kind == "flaky cases":
                # A drop over flaky cases is a few points off each, not a share of
                # each rate: there the plain count is the better test.
                drop = gain = scope_test([Pair(c.passed_s, c.judged_s, c.passed_b, c.judged_b)
                                          for c in selected]).z / scale  # fmt: skip
            else:
                drop = _lr_z(selected, self.delta, False, self.cap) / scale
                gain = _lr_z(selected, self.delta, True, self.cap) / scale
            drop_cut = z_threshold(self.alpha * share)
            gain_cut = -z_threshold(self.improvement_alpha * share)
            verdict.detail[scope] = (
                f"drop z = {drop:+.2f} (at {drop_cut:.2f}), gain z = {gain:+.2f}"
            )
            if drop <= drop_cut:
                verdict.regressed.append(scope)
            elif gain >= gain_cut:
                verdict.improved.append(scope)
        return verdict


@register_gate("ci_gate")
@dataclass(frozen=True)
class CIGate:
    """The PR check, one run per case, no re-runs: likelihood_ratio's weights and
    scopes, tested against a margin, with two levels.

    - A scope is **blocked** when it dropped by more than `margin` points at level
      `alpha` (strict: a false block costs a developer time), and **warned** when
      it did so at the looser `warn_alpha`. Drops within the margin pass.
    - The reliable cases are tested without a margin: a few of them breaking
      hardly moves an average, and is exactly what the check must not forgive.
    - Reliable and flaky cases are chosen by the baseline alone, so the new run
      cannot move a case between them.
    - **New cases** (added, or edited so they have no baseline) are not compared;
      their passes together are tested against `new_case_rate`, the least a
      sensible new case should pass: "new cases" is blocked or warned when they
      pass less often at the same levels.
    - Improvements use `improvement_alpha` without a margin.

    `regressed` holds the blocked scopes, `warned` the warned ones. The gate only
    reports; whether a block stops the merge is CI's choice.
    """

    alpha: float = 0.01
    warn_alpha: float = 0.05
    improvement_alpha: float = 0.002
    margin: float = 0.03
    new_case_rate: float = 0.5
    delta: float = 0.2
    cap: float = 0.95
    shares: tuple[float, float, float] = (0.3, 0.2, 0.3)  # total, reliable, flaky
    dispersion: dict[str, float] | None = None

    def tuned(self, level: float) -> CIGate:
        """Block at `level`, warn at the same ratio; improvements are only
        reported, so their level stays."""
        warn = min(level * self.warn_alpha / self.alpha, 0.99)
        return dataclasses.replace(self, alpha=level, warn_alpha=warn)

    def _level(self, verdict: Verdict, scope: str, p: float) -> None:
        if p <= self.alpha:
            verdict.regressed.append(scope)
        elif p <= self.warn_alpha:
            verdict.warned.append(scope)

    def judge(self, comparison: Comparison) -> Verdict:
        from qt_det.stats import Pair, scope_test, z_threshold

        verdict = Verdict()
        passed, judged = comparison.new_runs
        if judged:
            p = _binomial_below(passed, judged, self.new_case_rate)
            verdict.detail["new cases"] = (
                f"{passed}/{judged} passed; P(this few | {self.new_case_rate:.0%}) = {p:.3f}"
            )
            self._level(verdict, "new cases", p)
        cases = _cases(comparison)
        if not cases:
            return verdict
        phi = self.dispersion or {}
        for scope, kind, selected, share in _lr_scopes(cases, tuple(self.shares), True):
            scale = math.sqrt(max(phi.get(kind, 1.0), 1.0))
            # Reliable and flaky cases weigh about the same within their group:
            # the plain count, as in likelihood_ratio's flaky cases.
            weighted = kind not in ("reliable cases", "flaky cases")
            plain = scope_test([Pair(c.passed_s, c.judged_s, c.passed_b, c.judged_b)
                                for c in selected]).z / scale  # fmt: skip
            if kind == "reliable cases":
                drop = plain
            else:
                drop = _margin_z(selected, self.delta, self.cap, self.margin, weighted) / scale
            gain = _lr_z(selected, self.delta, True, self.cap) / scale if weighted else plain
            block_cut = z_threshold(self.alpha * share)
            warn_cut = z_threshold(self.warn_alpha * share)
            gain_cut = -z_threshold(self.improvement_alpha * share)
            margin = 0 if kind == "reliable cases" else self.margin
            verdict.detail[scope] = (
                f"drop beyond {margin * 100:.0f} pts: z = {drop:+.2f} "
                f"(warn at {warn_cut:.2f}, block at {block_cut:.2f}), gain z = {gain:+.2f}"
            )
            if drop <= block_cut:
                verdict.regressed.append(scope)
            elif drop <= warn_cut:
                verdict.warned.append(scope)
            elif gain >= gain_cut:
                verdict.improved.append(scope)
        return verdict


def _binomial_below(passed: int, n: int, rate: float) -> float:
    """P(at most `passed` passes in `n` runs at pass rate `rate`)."""
    return sum(math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(passed + 1))
