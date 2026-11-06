"""Candidate gates under exploration, registered like any user gate.

- `higher_criticism`: Donoho-Jin higher criticism over the per-case one-sided
  p-values (Fisher exact). Designed for sparse changes: a few cases among many
  shift, too few to move a total, too weak to stand out alone.
- `min_p_cases`: the smallest per-case p-value against a Bonferroni bound over
  all cases. Flags when any single case collapses.
- `zscore_split_hc`: `zscore_split` with part of its budget moved to a higher
  criticism test, to cover sparse changes as well.
- `zscore_strata`: `zscore_split` plus a scope of the reliable cases, taking over
  the strength of testimate's p^3 weighting without its blind spot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

from qt_analyzer.gates import Verdict, ZScoreSplitGate, register_gate
from qt_analyzer.stats import fisher_drop

if TYPE_CHECKING:
    from qt_analyzer.analyse import CaseResult, Comparison

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
    return c.passed_b / c.judged_b >= 0.95


@register_gate("zscore_strata")
@dataclass(frozen=True)
class ZScoreStrataGate:
    """`zscore_split` plus a scope of the reliable cases (baseline >= 95 %).

    Budget for regressions: total 35 %, reliable cases 25 %, flaky cases 15 %,
    topics the rest; improvements get `improvement_alpha` split the same way.
    The reliable scope takes over the strength of testimate's p^3 weighting
    (reliable cases breaking) without its blindness to flaky cases.
    """

    alpha: float = 0.05
    improvement_alpha: float = 0.01
    shares: tuple[float, float, float] = (0.35, 0.25, 0.15)  # total, reliable, flaky

    def tuned(self, level: float) -> ZScoreStrataGate:
        return ZScoreStrataGate(level, level * self.improvement_alpha / self.alpha, self.shares)

    def judge(self, comparison: Comparison) -> Verdict:
        from qt_analyzer.gates import _flaky
        from qt_analyzer.stats import Pair, scope_test, z_threshold

        verdict = Verdict()
        if not comparison.overall.cases:
            return verdict
        total, reliable_share, flaky_share = self.shares
        topics = sorted((n, t) for n, t in comparison.families.items() if t.cases)
        topic_share = (1 - total - reliable_share - flaky_share) / max(len(topics), 1)

        def pairs(keep):
            return [Pair(c.passed_s, c.judged_s, c.passed_b, c.judged_b)
                    for c in _cases(comparison) if keep(c)]  # fmt: skip

        scopes = [("overall", comparison.overall.z, total)]
        for name, keep, share in (("reliable cases", _reliable, reliable_share),
                                  ("flaky cases", _flaky, flaky_share)):  # fmt: skip
            selected = pairs(keep)
            if selected:
                scopes.append((name, scope_test(selected).z, share))
        scopes += [(name, t.z, topic_share) for name, t in topics]
        for scope, z, share in scopes:
            drop_cut = z_threshold(self.alpha * share)
            gain_cut = -z_threshold(self.improvement_alpha * share)
            verdict.detail[scope] = f"z = {z:+.2f} (drop at {drop_cut:.2f})"
            if z <= drop_cut:
                verdict.regressed.append(scope)
            elif z >= gain_cut:
                verdict.improved.append(scope)
        return verdict
