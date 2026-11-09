"""Check single test cases locally: regression, improvement or no evidence.

In CI each case runs once, too few to judge one case. Locally one case can run
many times (`pytest -k <case> --qt-runs 30 --report-log local.jsonl`); this
compares those runs with the same case's history on main.

Per case: one-sided Fisher exact tests in both directions against the case's
pooled counts from the newest measured main states. Several cases at once are
corrected with Holm, per direction. Without evidence, it estimates how many
local runs would detect a change of `delta`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

from qt_det.analyse import Agg, Counts, load, merge
from qt_det.contract import parse_shard_log
from qt_det.ingest import aggregate
from qt_det.stats import fisher_drop, wilson_interval
from qt_det.stores.results import ResultStore

_NORMAL = NormalDist()


@dataclass
class CaseVerdict:
    hrid: str
    case_key: str
    verdict: str  # "regression" | "improvement" | "no evidence" | "definition changed" | "new"
    local: tuple[int, int]  # passed, judged
    main: tuple[int, int] | None = None
    p_drop: float | None = None
    p_gain: float | None = None
    runs_needed: int | None = None  # local runs to detect a change of `delta`, if no evidence


def local_counts(paths: list[Path]) -> Counts:
    executions = []
    for path in paths:
        executions.extend(parse_shard_log(path).executions)
    return merge({}, aggregate(executions))


def main_counts(results: ResultStore, states: int) -> tuple[Counts, list[str]]:
    """Counts of the newest `states` measured main states, pooled."""
    timeline, _ = load(results)
    measured = [d for d in timeline if d.measured][-states:]
    pooled: Counts = {}
    for data in measured:
        merge(pooled, data.all_counts().items())
    return pooled, [d.state.key for d in measured]


def runs_needed(rate: float, n_main: int, delta: float, alpha: float, power: float = 0.8) -> int:
    """Local runs for a one-sided test to detect a drop of `delta` (normal approximation).

    Returns -1 when even unlimited local runs would not do, because main's
    history for this case is too short.
    """
    p = min(max(rate - delta / 2, 0.02), 0.98)  # variance between the two rates
    z = _NORMAL.inv_cdf(1 - alpha) + _NORMAL.inv_cdf(power)
    room = delta**2 / z**2 - p * (1 - p) / n_main
    if room <= 0:
        return -1
    return math.ceil(p * (1 - p) / room)


def _holm(p_values: list[float], alpha: float) -> list[bool]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    rejected = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] > alpha / (len(p_values) - rank):
            break
        rejected[i] = True
    return rejected


def check(
    local: Counts, main: Counts, *, alpha: float = 0.05, delta: float = 0.1
) -> list[CaseVerdict]:
    main_by_key: dict[str, list[tuple[str, Agg]]] = {}
    for (key, definition), agg in main.items():
        main_by_key.setdefault(key, []).append((definition, agg))

    verdicts: list[CaseVerdict] = []
    comparable: list[tuple[CaseVerdict, Agg, Agg]] = []
    for (key, definition), agg in sorted(local.items(), key=lambda kv: kv[1].hrid):
        v = CaseVerdict(agg.hrid, key, "no evidence", (agg.passed, agg.judged))
        base = dict(main_by_key.get(key, [])).get(definition)
        if base is None:
            v.verdict = "definition changed" if key in main_by_key else "new"
        elif base.judged == 0 or agg.judged == 0:
            v.verdict = "new"
        else:
            v.main = (base.passed, base.judged)
            comparable.append((v, agg, base))
        verdicts.append(v)

    for v, s, b in comparable:
        v.p_drop = fisher_drop(s.passed, s.failed, b.passed, b.failed)
        v.p_gain = fisher_drop(s.failed, s.passed, b.failed, b.passed)
    drops = _holm([v.p_drop or 1.0 for v, _, _ in comparable], alpha)
    gains = _holm([v.p_gain or 1.0 for v, _, _ in comparable], alpha)
    for (v, _, b), drop, gain in zip(comparable, drops, gains, strict=True):
        if drop:
            v.verdict = "regression"
        elif gain:
            v.verdict = "improvement"
        else:
            v.runs_needed = runs_needed(b.passed / b.judged, b.judged, delta, alpha)
    return verdicts


def _rate(passed: int, judged: int) -> str:
    low, high = wilson_interval(passed, judged)
    return f"{passed}/{judged} passed ({passed / judged:.1%}, 95% CI {low:.0%}-{high:.0%})"


def format_verdicts(verdicts: list[CaseVerdict], states: list[str], delta: float) -> str:
    lines = []
    for v in verdicts:
        lines.append(v.hrid)
        if v.main:
            lines.append(f"  main:   {_rate(*v.main)}  from the last {len(states)} main states")
        lines.append(f"  local:  {_rate(*v.local)}")
        if v.verdict == "regression":
            lines.append(f"  REGRESSION  (p = {v.p_drop:.2g}, one-sided Fisher exact)")
        elif v.verdict == "improvement":
            lines.append(f"  IMPROVEMENT  (p = {v.p_gain:.2g}, one-sided Fisher exact)")
        elif v.verdict == "definition changed":
            lines.append("  test definition changed: no comparable history on main")
        elif v.verdict == "new":
            lines.append("  new test: no history on main")
        else:
            extra = ""
            if v.runs_needed == -1:
                extra = (
                    f"; main's history ({v.main[1] if v.main else 0} runs) is too short to "
                    f"detect a {delta:.0%} drop however many local runs (more nightly runs "
                    "on main would help)"
                )
            elif v.runs_needed is not None:
                more = max(v.runs_needed - v.local[1], 0)
                extra = f"; about {v.runs_needed} local runs detect a {delta:.0%} drop" + (
                    f" ({more} more)" if more else ""
                )
            lines.append(f"  no evidence of change{extra}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
