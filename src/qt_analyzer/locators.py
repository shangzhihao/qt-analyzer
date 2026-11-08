"""Locators: once a gate has flagged a change, where is it?

A gate answers "did quality change?" for the whole suite; a locator ranks the
places that changed: scopes (topics, reliable cases, flaky cases), then single
test cases. Locators are registered like gates (`@register_locator`), selected
with `locator = "<name>"` in `analyzer.toml` and configured with
`[locators.<name>]`.

- `ranked`: scopes by their z in the direction of the change; cases by their
  one-sided Fisher exact p-value, restricted to the flagged topics when the gate
  named any (otherwise all cases), with Benjamini-Hochberg q-values within that
  region. A topic is usually found from one run; single cases need more runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from qt_analyzer.stats import Pair, benjamini_hochberg, fisher_drop, scope_test

if TYPE_CHECKING:
    from qt_analyzer.analyse import CaseResult, Comparison

STRATA = ("reliable cases", "flaky cases")


@dataclass(frozen=True)
class ScopeRank:
    scope: str
    z: float
    rate_b: float
    rate_s: float


@dataclass(frozen=True)
class CaseRank:
    case_key: str
    hrid: str
    family: str
    passed_s: int
    judged_s: int
    passed_b: int
    judged_b: int
    p: float
    q: float


@dataclass
class Location:
    direction: str  # "regression" | "improvement"
    scopes: list[ScopeRank] = field(default_factory=list)
    cases: list[CaseRank] = field(default_factory=list)


class Locator(Protocol):
    def locate(self, comparison: Comparison, direction: str, flagged: list[str]) -> Location: ...


_REGISTRY: dict[str, Callable[..., Locator]] = {}


def register_locator(name: str) -> Callable[[type], type]:
    def decorate(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"locator {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorate


def available_locators() -> list[str]:
    return sorted(_REGISTRY)


def make_locator(name: str, params: dict | None = None) -> Locator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown locator {name!r}; available: {', '.join(available_locators())}")
    return _REGISTRY[name](**(params or {}))


def _judged(comparison: Comparison) -> list[CaseResult]:
    return [c for c in comparison.cases if c.judged_s and c.judged_b]


def strata_of(case: CaseResult) -> str:
    rate = case.passed_b / case.judged_b
    if rate >= 0.95:
        return "reliable cases"
    if rate > 0.05:
        return "flaky cases"
    return ""


def case_p(case: CaseResult, direction: str) -> float:
    if direction == "regression":
        return case.p  # one-sided Fisher exact: fewer passes than the baseline
    failed_s, failed_b = case.judged_s - case.passed_s, case.judged_b - case.passed_b
    return fisher_drop(failed_s, case.passed_s, failed_b, case.passed_b)


@register_locator("ranked")
@dataclass(frozen=True)
class RankedLocator:
    top_scopes: int = 3
    top_cases: int = 10

    def locate(self, comparison: Comparison, direction: str, flagged: list[str]) -> Location:
        sign = -1 if direction == "regression" else 1
        cases = _judged(comparison)

        def scope_rank(name: str, members: list[CaseResult]) -> ScopeRank:
            pairs = [Pair(c.passed_s, c.judged_s, c.passed_b, c.judged_b) for c in members]
            t = scope_test(pairs)
            return ScopeRank(name, t.z, t.rate_b, t.rate_s)

        scopes = [ScopeRank(name, t.z, t.rate_b, t.rate_s)
                  for name, t in comparison.families.items() if t.cases]  # fmt: skip
        for stratum in STRATA:
            members = [c for c in cases if strata_of(c) == stratum]
            if members:
                scopes.append(scope_rank(stratum, members))
        scopes.sort(key=lambda s: sign * s.z, reverse=True)

        topics = [s for s in flagged if s in comparison.families]
        region = [c for c in cases if c.family in topics] if topics else cases
        ps = [case_p(c, direction) for c in region]
        qs = benjamini_hochberg(ps) if ps else []
        ranked = sorted(zip(region, ps, qs, strict=True), key=lambda t: t[1])
        return Location(
            direction,
            scopes[: self.top_scopes],
            [
                CaseRank(
                    c.case_key,
                    c.hrid,
                    c.family,
                    c.passed_s,
                    c.judged_s,
                    c.passed_b,
                    c.judged_b,
                    p,
                    q,
                )
                for c, p, q in ranked[: self.top_cases]
            ],  # fmt: skip
        )


def format_location(location: Location, limit: int = 5) -> list[str]:
    """Short human-readable lines: the leading scopes, then the most affected cases."""
    lines = []
    if location.scopes:
        lines.append(
            "where: "
            + " · ".join(
                f"{s.scope} {s.rate_b:.0%}→{s.rate_s:.0%} (z {s.z:+.1f})" for s in location.scopes
            )
        )
    if location.cases:
        lines.append(
            "most affected: "
            + " · ".join(
                f"{c.hrid} {c.passed_b}/{c.judged_b}→{c.passed_s}/{c.judged_s}"
                for c in location.cases[:limit]
            )
        )
    return lines


def location_from_json(data: dict) -> Location:
    return Location(
        data["direction"],
        [ScopeRank(**s) for s in data.get("scopes", [])],
        [CaseRank(**c) for c in data.get("cases", [])],
    )
