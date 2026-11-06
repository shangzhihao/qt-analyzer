"""Analyse ingested runs: the main timeline, nightly detection, alerts and PRs.

Each pass recomputes everything from the per-run counts, so it is idempotent and
needs no state of its own beyond the ResultStore.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qt_analyzer.contract import Manifest, State, state_key
from qt_analyzer.stats import (
    Pair,
    ScopeTest,
    benjamini_hochberg,
    fisher_drop,
    scope_test,
    z_threshold,
)
from qt_analyzer.stores.base import CaseCount
from qt_analyzer.stores.results import ResultStore

CaseId = tuple[str, str]  # (case_key, definition_hash)


@dataclass(frozen=True)
class Config:
    pool_k: int = 5
    push_alpha_per_state: float = 0.05
    nightly_alpha_per_state: float = 0.01
    nightly_case_fdr: float = 0.05
    reference_alpha_per_state: float = 0.01
    # The reference moves to a state only if it shows no hint of a drop at this
    # looser level, so small steps cannot drag it down one by one.
    reference_move_alpha_per_state: float = 0.2
    pr_alpha: float = 0.05
    pr_case_p: float = 0.05

    @classmethod
    def load(cls, path: Path | None) -> Config:
        if path is None or not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        known = {k: v for k, v in data.get("analysis", {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# counts


@dataclass
class Agg:
    family: str
    hrid: str
    n: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0

    @property
    def judged(self) -> int:
        """Executions with a pass/fail verdict; errors are infrastructure noise."""
        return self.passed + self.failed


Counts = dict[CaseId, Agg]


def merge(target: Counts, rows: Iterable[CaseCount] | Iterable[tuple[CaseId, Agg]]) -> Counts:
    for row in rows:
        if isinstance(row, CaseCount):
            cid, src = (row.case_key, row.definition_hash), row
        else:
            cid, src = row
        agg = target.setdefault(cid, Agg(src.family, src.hrid))
        agg.n += src.n
        agg.passed += src.passed
        agg.failed += src.failed
        agg.error += src.error
    return target


def pass_rate(counts: Counts) -> float | None:
    judged = sum(a.judged for a in counts.values())
    return sum(a.passed for a in counts.values()) / judged if judged else None


# ---------------------------------------------------------------------------
# comparisons


@dataclass
class CaseResult:
    case_key: str
    hrid: str
    family: str
    passed_s: int
    judged_s: int
    passed_b: int
    judged_b: int
    p: float
    q: float | None = None


@dataclass
class Coverage:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)


@dataclass
class Comparison:
    overall: ScopeTest
    families: dict[str, ScopeTest]
    cases: list[CaseResult]
    coverage: Coverage

    def flagged_scopes(self, alpha_per_state: float) -> list[str]:
        """Scopes with a significant drop, Bonferroni-split over overall + families."""
        threshold = z_threshold(alpha_per_state / (1 + len(self.families)))
        scopes = [("overall", self.overall)] + sorted(self.families.items())
        return [name for name, t in scopes if t.cases and t.z <= threshold]

    def improved_scopes(self, alpha_per_state: float) -> list[str]:
        threshold = -z_threshold(alpha_per_state / (1 + len(self.families)))
        scopes = [("overall", self.overall)] + sorted(self.families.items())
        return [name for name, t in scopes if t.cases and t.z >= threshold]


def compare(new: Counts, base: Counts) -> Comparison:
    common = sorted(cid for cid in new if cid in base)
    pairs = {
        cid: Pair(new[cid].passed, new[cid].judged, base[cid].passed, base[cid].judged)
        for cid in common
    }
    by_family: dict[str, list[Pair]] = defaultdict(list)
    for cid, pair in pairs.items():
        by_family[new[cid].family].append(pair)
    cases = [
        CaseResult(
            cid[0],
            new[cid].hrid,
            new[cid].family,
            new[cid].passed,
            new[cid].judged,
            base[cid].passed,
            base[cid].judged,
            fisher_drop(new[cid].passed, new[cid].failed, base[cid].passed, base[cid].failed),
        )
        for cid in common
    ]
    new_keys = {cid[0] for cid in new}
    base_keys = {cid[0] for cid in base}
    common_keys = {cid[0] for cid in common}
    coverage = Coverage(
        added=sorted(new_keys - base_keys),
        removed=sorted(base_keys - new_keys),
        edited=sorted((new_keys & base_keys) - common_keys),
    )
    return Comparison(
        scope_test(pairs.values()),
        {family: scope_test(ps) for family, ps in sorted(by_family.items())},
        cases,
        coverage,
    )


# ---------------------------------------------------------------------------
# loaded data


@dataclass
class StateData:
    state: State
    counts: dict[str, Counts] = field(default_factory=lambda: {"main": {}, "nightly": {}})
    runs: dict[str, list[str]] = field(default_factory=lambda: {"main": [], "nightly": []})

    def all_counts(self) -> Counts:
        return merge(merge({}, self.counts["main"].items()), self.counts["nightly"].items())

    @property
    def measured(self) -> bool:
        return bool(self.runs["main"] or self.runs["nightly"])


@dataclass
class PrRun:
    manifest: Manifest
    counts: Counts


def load(results: ResultStore) -> tuple[list[StateData], list[PrRun]]:
    """Read complete runs; return the ordered main timeline and the PR runs."""
    states: dict[str, StateData] = {}
    prs: list[PrRun] = []
    for run_id, entry in sorted(results.load_index().items()):
        if entry.status != "complete":
            continue
        manifest = Manifest.from_json(entry.manifest)
        counts = merge({}, results.get_counts(run_id))
        if manifest.kind == "pr":
            prs.append(PrRun(manifest, counts))
            continue
        data = states.setdefault(manifest.state.key, StateData(manifest.state))
        merge(data.counts[manifest.kind], counts.items())
        data.runs[manifest.kind].append(run_id)
    timeline = sorted(states.values(), key=lambda d: (d.state.committed_at, d.state.key))
    return timeline, prs


# ---------------------------------------------------------------------------
# results


@dataclass
class PushResult:
    status: str  # "suspected" | "ok" | "no-baseline"
    baseline_states: list[str]
    suspect_range: list[str]
    flagged: list[str]
    improved: list[str]
    comparison: Comparison | None


@dataclass
class NightlyResult:
    status: str  # "confirmed" | "ongoing" | "cleared" | "ok" | "first"
    predecessor: str | None
    reference: str | None
    step_flagged: list[str]
    reference_flagged: list[str]
    cases_flagged: list[str]
    step: Comparison | None
    vs_reference: Comparison | None


@dataclass
class Alert:
    state: str
    suspect_range: list[str]
    scopes: list[str]
    cases: list[str]
    reference: str | None
    open: bool = True
    still_regressed_at: list[str] = field(default_factory=list)
    closed_by: str | None = None


@dataclass
class PrResult:
    repo: str
    number: int
    head_sha: str
    run_id: str
    status: str  # "baseline-pending" | "ok" | "regressed"
    baseline: str | None
    flagged: list[str]
    improved: list[str]
    suspected_cases: list[str]
    comparison: Comparison | None
    run_url: str = ""


@dataclass
class TimelineEntry:
    data: StateData
    push: PushResult | None = None
    nightly: NightlyResult | None = None


@dataclass
class Analysis:
    timeline: list[TimelineEntry]
    alerts: list[Alert]
    reference: str | None
    prs: list[PrResult]


def analyse(results: ResultStore, config: Config) -> Analysis:
    timeline_data, pr_runs = load(results)
    return analyse_data(timeline_data, pr_runs, config)


def analyse_data(timeline_data: list[StateData], pr_runs: list[PrRun], config: Config) -> Analysis:
    """Analyse an ordered main timeline and PR runs held in memory."""
    entries = [TimelineEntry(d) for d in timeline_data]
    alerts, reference = _nightly(entries, config)
    for i, entry in enumerate(entries):
        if entry.data.runs["main"]:
            entry.push = _push(entries, i, config)
        if entry.push and entry.push.status == "suspected" and entry.nightly:
            if entry.nightly.status == "ok":
                entry.nightly.status = "cleared"
    prs = _prs(entries, pr_runs, config)
    return Analysis(entries, alerts, reference, prs)


def _starts_regression(entry: TimelineEntry) -> bool:
    return entry.nightly is not None and entry.nightly.status == "confirmed"


def _baseline(entries: list[TimelineEntry], i: int, k: int) -> tuple[list[int], list[int]]:
    """Up to k measured states before i, and the unmeasured gap right before i.

    Pooling stops at a state that started a confirmed regression: states before
    it measured different quality and would make every later state look worse.
    """
    measured: list[int] = []
    gap: list[int] = []
    for j in range(i - 1, -1, -1):
        if entries[j].data.measured:
            measured.append(j)
            if len(measured) == k or _starts_regression(entries[j]):
                break
        elif not measured:
            gap.append(j)
    return measured, gap


def _push(entries: list[TimelineEntry], i: int, config: Config) -> PushResult:
    measured, gap = _baseline(entries, i, config.pool_k)
    suspect_range = [entries[j].data.state.key for j in reversed(gap)] + [entries[i].data.state.key]
    if not measured:
        return PushResult("no-baseline", [], suspect_range, [], [], None)
    base: Counts = {}
    for j in measured:
        merge(base, entries[j].data.all_counts().items())
    comparison = compare(entries[i].data.counts["main"], base)
    flagged = comparison.flagged_scopes(config.push_alpha_per_state)
    return PushResult(
        "suspected" if flagged else "ok",
        [entries[j].data.state.key for j in measured],
        suspect_range,
        flagged,
        comparison.improved_scopes(config.push_alpha_per_state),
        comparison,
    )


def _case_flags(comparison: Comparison, fdr: float) -> list[str]:
    qs = benjamini_hochberg([c.p for c in comparison.cases])
    for case, q in zip(comparison.cases, qs, strict=True):
        case.q = q
    return sorted(c.case_key for c in comparison.cases if c.q is not None and c.q <= fdr)


def _nightly(entries: list[TimelineEntry], config: Config) -> tuple[list[Alert], str | None]:
    alerts: list[Alert] = []
    open_alert: Alert | None = None
    reference: StateData | None = None
    previous: StateData | None = None
    previous_index = -1
    for i, entry in enumerate(entries):
        data = entry.data
        if not data.runs["nightly"]:
            continue
        nightly = data.counts["nightly"]
        if reference is None or previous is None:
            entry.nightly = NightlyResult("first", None, None, [], [], [], None, None)
            reference = previous = data
            previous_index = i
            continue

        step = compare(nightly, previous.counts["nightly"])
        step_flagged = step.flagged_scopes(config.nightly_alpha_per_state)
        cases_flagged = _case_flags(step, config.nightly_case_fdr)
        vs_ref = compare(nightly, reference.counts["nightly"])
        ref_flagged = vs_ref.flagged_scopes(config.reference_alpha_per_state)
        suspect_range = [entries[j].data.state.key for j in range(previous_index + 1, i + 1)]

        new_drop = bool(step_flagged or cases_flagged)
        if open_alert is None and ref_flagged:
            new_drop = True  # slow drift: no single step was large enough
        if new_drop:
            if open_alert is not None:
                open_alert.open = False
                open_alert.closed_by = f"superseded by {data.state.key}"
            open_alert = Alert(
                data.state.key,
                suspect_range,
                sorted(set(step_flagged) | set(ref_flagged)),
                cases_flagged,
                reference.state.key,
            )
            alerts.append(open_alert)
            status = "confirmed"
        elif open_alert is not None and ref_flagged:
            open_alert.still_regressed_at.append(data.state.key)
            status = "ongoing"
        else:
            if open_alert is not None:
                open_alert.open = False
                open_alert.closed_by = data.state.key
                open_alert = None
            if not vs_ref.flagged_scopes(config.reference_move_alpha_per_state):
                reference = data
            status = "ok"

        entry.nightly = NightlyResult(
            status,
            previous.state.key,
            reference.state.key,
            step_flagged,
            ref_flagged,
            cases_flagged,
            step,
            vs_ref,
        )
        previous, previous_index = data, i
    return alerts, reference.state.key if reference else None


def _prs(entries: list[TimelineEntry], runs: list[PrRun], config: Config) -> list[PrResult]:
    latest: dict[tuple[str, int], PrRun] = {}
    for run in runs:
        pr = run.manifest.pr
        assert pr is not None
        key = (pr.repo, pr.number)
        if key not in latest or run.manifest.created_at >= latest[key].manifest.created_at:
            latest[key] = run
    position = {e.data.state.key: i for i, e in enumerate(entries)}
    results = []
    for (repo, number), run in sorted(latest.items()):
        m, pr = run.manifest, run.manifest.pr
        assert pr is not None
        shas = m.state.shas | {pr.component: pr.base_sha}
        base_key = state_key(shas)
        i = position.get(base_key)
        if i is None or not entries[i].data.measured:
            results.append(
                PrResult(
                    repo,
                    number,
                    pr.head_sha,
                    m.run_id,
                    "baseline-pending",
                    base_key,
                    [],
                    [],
                    [],
                    None,
                    m.producer.get("run_url", ""),
                )
            )
            continue
        earlier: list[int] = []
        if not _starts_regression(entries[i]):
            earlier, _ = _baseline(entries, i, config.pool_k - 1)
        base = entries[i].data.all_counts()
        for j in earlier:
            merge(base, entries[j].data.all_counts().items())
        comparison = compare(run.counts, base)
        flagged = comparison.flagged_scopes(config.pr_alpha)
        suspected = sorted(
            c.case_key
            for c in comparison.cases
            if c.p <= config.pr_case_p and c.passed_s < c.judged_s
        )
        results.append(
            PrResult(
                repo,
                number,
                pr.head_sha,
                m.run_id,
                "regressed" if flagged else "ok",
                base_key,
                flagged,
                comparison.improved_scopes(config.pr_alpha),
                suspected,
                comparison,
                m.producer.get("run_url", ""),
            )
        )
    return results


# ---------------------------------------------------------------------------
# persisting


def _scope_json(t: ScopeTest) -> dict[str, Any]:
    return {"cases": t.cases, "rate_s": t.rate_s, "rate_b": t.rate_b, "z": t.z}


def comparison_json(c: Comparison | None) -> dict[str, Any] | None:
    if c is None:
        return None
    return {
        "overall": _scope_json(c.overall),
        "families": {f: _scope_json(t) for f, t in c.families.items()},
        "coverage": asdict(c.coverage),
    }


def _state_json(data: StateData) -> dict[str, Any]:
    s = data.state
    return {
        "state_key": s.key,
        "created_by": s.created_by,
        "committed_at": s.committed_at,
        "components": {n: c.sha for n, c in s.components.items()},
        "runs": data.runs,
        "main_rate": pass_rate(data.counts["main"]),
        "nightly_rate": pass_rate(data.counts["nightly"]),
    }


def _file_name(key: str) -> str:
    return key.removeprefix("sha256:")[:16] + ".json"


def save(results: ResultStore, analysis: Analysis) -> None:
    rows = []
    for e in analysis.timeline:
        row = _state_json(e.data)
        if e.push:
            row["push"] = {
                "status": e.push.status,
                "flagged": e.push.flagged,
                "improved": e.push.improved,
                "suspect_range": e.push.suspect_range,
                "comparison": comparison_json(e.push.comparison),
            }
        if e.nightly:
            row["nightly"] = {
                "status": e.nightly.status,
                "predecessor": e.nightly.predecessor,
                "reference": e.nightly.reference,
                "step_flagged": e.nightly.step_flagged,
                "reference_flagged": e.nightly.reference_flagged,
                "cases_flagged": e.nightly.cases_flagged,
                "step": comparison_json(e.nightly.step),
                "vs_reference": comparison_json(e.nightly.vs_reference),
            }
        rows.append(row)
    results.write_jsonl("states/timeline.jsonl", rows)
    results.write_json("reference.json", {"state_key": analysis.reference})
    results.replace_dir("alerts", {_file_name(a.state): asdict(a) for a in analysis.alerts})
    results.replace_dir(
        "pr",
        {
            f"{p.repo.replace('/', '__')}-{p.number}.json": {
                **{k: v for k, v in asdict(p).items() if k != "comparison"},
                "comparison": comparison_json(p.comparison),
            }
            for p in analysis.prs
        },
    )
