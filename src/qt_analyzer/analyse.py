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
from qt_analyzer.decisions import DecisionLog, alert_id, config_hash
from qt_analyzer.locators import Location, make_locator
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
    # Nightly step and drift tests confirm a drop when a scope's z is at or below
    # this. -2.5 gave about 4 false issues per month on the real suite's
    # distribution (60 nightly states a month, 10 runs).
    nightly_z: float = -2.5
    nightly_case_fdr: float = 0.05  # cases are listed as evidence only
    # The reference moves to a state only if it shows no hint of a drop at this
    # looser level, so small steps cannot drag it down one by one.
    reference_move_alpha_per_state: float = 0.2
    pr_alpha: float = 0.05
    pr_case_p: float = 0.05
    # Which gate algorithm decides (see gates.py). "zscore" uses the thresholds
    # above per stage; other gates use their own settings from [gates.<name>].
    gate: str = "zscore"
    gate_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Which locator says where a flagged change is (see locators.py).
    locator: str = "ranked"
    locator_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> Config:
        if path is None or not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        analysis = data.get("analysis", {})
        known = {k: v for k, v in analysis.items() if k in cls.__dataclass_fields__}
        from qt_analyzer.gates import load_gate_modules

        load_gate_modules(analysis.get("gate_modules", []))
        return cls(
            **known
            | {"gate_params": data.get("gates", {}), "locator_params": data.get("locators", {})}
        )


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

    @property
    def comparable(self) -> bool:
        """At least one case has the same identity and definition on both sides."""
        return self.overall.cases > 0

    def not_comparable_reason(self) -> str:
        c = self.coverage
        return (
            f"0 cases compared ({len(c.edited)} edited, {len(c.added)} added, "
            f"{len(c.removed)} removed)"
        )

    def scopes_below(self, z: float) -> list[str]:
        """Scopes whose z is at or below a fixed threshold."""
        scopes = [("overall", self.overall)] + sorted(self.families.items())
        return [name for name, t in scopes if t.cases and t.z <= z]

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


def judge(comparison: Comparison, stage: str, config: Config) -> tuple[list[str], list[str]]:
    """(regressed, improved) scopes by the configured gate, for a stage
    ("push", "nightly" or "pr")."""
    if config.gate == "zscore":
        if stage == "push":
            alpha = config.push_alpha_per_state
        elif stage == "nightly":
            return comparison.scopes_below(config.nightly_z), []
        else:
            alpha = config.pr_alpha
        return comparison.flagged_scopes(alpha), comparison.improved_scopes(alpha)
    from qt_analyzer.gates import make_gate

    verdict = make_gate(config.gate, stage_params(config, stage)).judge(comparison)
    return verdict.regressed, verdict.improved


def locate(comparison: Comparison, direction: str, flagged: list[str], config: Config) -> Location:
    """Where a flagged change is, by the configured locator."""
    locator = make_locator(config.locator, config.locator_params.get(config.locator))
    return locator.locate(comparison, direction, flagged)


def location_json(location: Location | None) -> dict[str, Any] | None:
    return asdict(location) if location else None


STAGES = ("push", "nightly", "pr")


def stage_params(config: Config, stage: str) -> dict[str, Any]:
    """The gate's settings for a stage: `[gates.<name>]`, overridden by
    `[gates.<name>.<stage>]`."""
    params = config.gate_params.get(config.gate, {})
    base = {k: v for k, v in params.items() if k not in STAGES}
    return base | params.get(stage, {})


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
    status: str  # "suspected" | "ok" | "no-baseline" | "not-comparable"
    baseline_states: list[str]
    suspect_range: list[str]
    flagged: list[str]
    improved: list[str]
    comparison: Comparison | None


@dataclass
class NightlyResult:
    status: str  # "confirmed" | "ongoing" | "cleared" | "ok" | "first" | "not-comparable"
    predecessor: str | None
    reference: str | None
    step_flagged: list[str]
    reference_flagged: list[str]
    cases_flagged: list[str]
    step: Comparison | None
    vs_reference: Comparison | None
    reason: str = ""  # why it was not comparable


@dataclass
class Alert:
    id: str
    state: str
    suspect_range: list[str]
    scopes: list[str]
    cases: list[str]
    reference: str | None
    open: bool = True
    still_regressed_at: list[str] = field(default_factory=list)
    closed_by: str | None = None
    where: dict[str, Any] | None = None  # the locator's ranking when the alert opened


@dataclass
class PrResult:
    repo: str
    number: int
    head_sha: str
    run_id: str
    status: str  # "baseline-pending" | "ok" | "regressed" | "not-comparable"
    baseline: str | None
    flagged: list[str]
    improved: list[str]
    suspected_cases: list[str]
    comparison: Comparison | None
    run_url: str = ""
    nearest_baseline: bool = False  # the stated baseline was not measured
    reason: str = ""  # why it was not comparable
    location: Location | None = None  # where the change is, when one was flagged


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
    log: DecisionLog = field(default_factory=DecisionLog)


def analyse(results: ResultStore, config: Config) -> Analysis:
    timeline_data, pr_runs = load(results)
    return analyse_data(timeline_data, pr_runs, config, DecisionLog.load(results))


def analyse_data(
    timeline_data: list[StateData],
    pr_runs: list[PrRun],
    config: Config,
    log: DecisionLog | None = None,
) -> Analysis:
    """Analyse an ordered main timeline and PR runs held in memory.

    Decided states and reported PR runs keep the verdicts in `log`; new ones are
    decided and appended to it.
    """
    log = log if log is not None else DecisionLog()
    entries = [TimelineEntry(d) for d in timeline_data]
    alerts, reference = _nightly(entries, config, log)
    for i, entry in enumerate(entries):
        if entry.data.runs["main"]:
            entry.push = _push(entries, i, config)
        if entry.push and entry.push.status == "suspected" and entry.nightly:
            if entry.nightly.status == "ok":
                entry.nightly.status = "cleared"
    prs = _prs(entries, pr_runs, config, log)
    return Analysis(entries, alerts, reference, prs, log)


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
    if not comparison.comparable:
        return PushResult("not-comparable", [entries[j].data.state.key for j in measured],
                          suspect_range, [], [], comparison)  # fmt: skip
    flagged, improved = judge(comparison, "push", config)
    return PushResult(
        "suspected" if flagged else "ok",
        [entries[j].data.state.key for j in measured],
        suspect_range,
        flagged,
        improved,
        comparison,
    )


def _case_flags(comparison: Comparison, fdr: float) -> list[str]:
    qs = benjamini_hochberg([c.p for c in comparison.cases])
    for case, q in zip(comparison.cases, qs, strict=True):
        case.q = q
    return sorted(c.case_key for c in comparison.cases if c.q is not None and c.q <= fdr)


def _apply(effects: list[dict[str, Any]], alerts: dict[str, Alert], open_ids: list[str]) -> None:
    for e in effects:
        if e["action"] == "opened":
            alerts[e["id"]] = Alert(e["id"], e["state"], e["suspect_range"], e["scopes"],
                                    e["cases"], e["reference"], where=e.get("where"))  # fmt: skip
            open_ids.append(e["id"])
        elif e["action"] == "extended":
            alerts[e["id"]].still_regressed_at.append(e["state"])
        elif e["action"] == "closed":
            alerts[e["id"]].open = False
            alerts[e["id"]].closed_by = e["state"]
            open_ids.remove(e["id"])


def _nightly(
    entries: list[TimelineEntry], config: Config, log: DecisionLog
) -> tuple[list[Alert], str | None]:
    """Walk the nightly-measured states in timeline order.

    A state with a `state.decided` record replays it: verdict, alert effects and
    the reference afterwards come from the log. A new state is decided against
    the reference and open alerts in force at its position, and recorded.
    """
    decided = log.decided()
    by_key = {e.data.state.key: e.data for e in entries}
    alerts: dict[str, Alert] = {}
    open_ids: list[str] = []
    reference: StateData | None = None
    previous: StateData | None = None
    previous_index = -1
    for i, entry in enumerate(entries):
        data = entry.data
        if not data.runs["nightly"]:
            continue
        key = data.state.key
        nightly = data.counts["nightly"]
        step = compare(nightly, previous.counts["nightly"]) if previous else None
        vs_ref = compare(nightly, reference.counts["nightly"]) if reference else None
        record = decided.get(key)
        if record is None:
            record = _decide(entries, i, previous_index, step, vs_ref, reference, open_ids,
                             alerts, config, log)  # fmt: skip
        _apply(record["effects"], alerts, open_ids)
        entry.nightly = NightlyResult(
            record["verdict"],
            previous.state.key if previous else None,
            record["reference"],
            record["step_flagged"],
            record["reference_flagged"],
            record["cases_flagged"],
            step,
            vs_ref,
            record.get("reason", ""),
        )
        reference = by_key.get(record["reference"], data)
        previous, previous_index = data, i
    return list(alerts.values()), reference.state.key if reference else None


def _decide(
    entries: list[TimelineEntry],
    i: int,
    previous_index: int,
    step: Comparison | None,
    vs_ref: Comparison | None,
    reference: StateData | None,
    open_ids: list[str],
    alerts: dict[str, Alert],
    config: Config,
    log: DecisionLog,
) -> dict[str, Any]:
    data = entries[i].data
    key = data.state.key
    step_ok = step is not None and step.comparable
    step_flagged = judge(step, "nightly", config)[0] if step_ok else []
    cases_flagged = _case_flags(step, config.nightly_case_fdr) if step_ok else []
    ref_ok = vs_ref is not None and vs_ref.comparable
    ref_flagged = judge(vs_ref, "nightly", config)[0] if ref_ok else []
    effects: list[dict[str, Any]] = []
    reason = ""
    if reference is None or step is None or vs_ref is None:
        verdict, reference_after = "first", data
    elif not ref_ok:
        # The reference has other test definitions (e.g. the suite was edited):
        # nothing can be checked. Restart the reference here; alerts about the
        # old definitions can no longer be checked and are closed.
        reason = vs_ref.not_comparable_reason()
        effects += [{"action": "closed", "id": a, "state": key} for a in open_ids]
        verdict, reference_after = "not-comparable", data
    elif step_flagged or (ref_flagged and not open_ids):
        # a new drop (a step, or slow drift while no alert is open)
        new_id = alert_id(key, log.alert_ids() | set(alerts))
        suspect = [entries[j].data.state.key for j in range(previous_index + 1, i + 1)]
        source = step if step_flagged else vs_ref
        flagged = step_flagged or ref_flagged
        where = location_json(locate(source, "regression", flagged, config))
        effects.append({"action": "opened", "id": new_id, "state": key, "suspect_range": suspect,
                        "scopes": sorted(set(step_flagged) | set(ref_flagged)),
                        "cases": cases_flagged, "reference": reference.state.key,
                        "where": where})  # fmt: skip
        verdict, reference_after = "confirmed", reference
    elif ref_flagged and open_ids:
        effects.append({"action": "extended", "id": open_ids[-1], "state": key})
        verdict, reference_after = "ongoing", reference
    else:
        effects += [{"action": "closed", "id": a, "state": key} for a in open_ids]
        moves = not vs_ref.flagged_scopes(config.reference_move_alpha_per_state)
        verdict, reference_after = "ok", data if moves else reference
    record = log.append(
        "state.decided",
        state=key,
        verdict=verdict,
        reference=reference_after.state.key,
        step_flagged=step_flagged,
        reference_flagged=ref_flagged,
        cases_flagged=cases_flagged,
        effects=effects,
        reason=reason,
        gate=config.gate,
        nightly_z=config.nightly_z,
        config=config_hash(config),
    )
    for e in effects:
        log.append(f"alert.{e['action']}", alert=e["id"], state=key)
    if reference is None or reference_after is not reference:
        moved_because = "definitions changed" if verdict == "not-comparable" else verdict
        log.append("reference.moved", state=reference_after.state.key, by=key,
                   reason=moved_because)  # fmt: skip
    return record


def _nearest_measured(entries: list[TimelineEntry], before: str) -> int | None:
    candidates = [
        i for i, e in enumerate(entries) if e.data.measured and e.data.state.committed_at <= before
    ]
    return candidates[-1] if candidates else None


def _prs(
    entries: list[TimelineEntry], runs: list[PrRun], config: Config, log: DecisionLog
) -> list[PrResult]:
    latest: dict[tuple[str, int], PrRun] = {}
    for run in runs:
        pr = run.manifest.pr
        assert pr is not None
        key = (pr.repo, pr.number)
        if key not in latest or run.manifest.created_at >= latest[key].manifest.created_at:
            latest[key] = run
    position = {e.data.state.key: i for i, e in enumerate(entries)}
    reports = log.pr_reports()
    results = []
    for (repo, number), run in sorted(latest.items()):
        m, pr = run.manifest, run.manifest.pr
        assert pr is not None
        base_key = state_key(pr.baseline)
        i = position.get(base_key)
        nearest = False
        if i is None or not entries[i].data.measured:
            # The stated baseline was never measured: use the newest measured
            # main state from before this PR run, and say so.
            i = _nearest_measured(entries, m.created_at)
            nearest = i is not None
        if i is None:
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
        reported = reports.get(m.run_id)
        reason = "" if comparison.comparable else comparison.not_comparable_reason()
        if reported is not None:  # decided once; later data cannot flip it
            flagged, improved = reported["flagged"], reported["improved"]
            suspected = reported["suspected_cases"]
        elif reason:
            flagged, improved, suspected = [], [], []
            if not nearest:
                log.append("pr.reported", run_id=m.run_id, repo=repo, number=number,
                           head_sha=pr.head_sha, baseline=base_key, status="not-comparable",
                           flagged=[], improved=[], suspected_cases=[], reason=reason)  # fmt: skip
        else:
            flagged, improved = judge(comparison, "pr", config)
            suspected = sorted(
                c.case_key
                for c in comparison.cases
                if c.p <= config.pr_case_p and c.passed_s < c.judged_s
            )
            if not nearest:
                log.append("pr.reported", run_id=m.run_id, repo=repo, number=number,
                           head_sha=pr.head_sha, baseline=base_key,
                           status="regressed" if flagged else "ok", flagged=flagged,
                           improved=improved, suspected_cases=suspected)  # fmt: skip
        location = None
        if flagged:
            location = locate(comparison, "regression", flagged, config)
        elif improved:
            location = locate(comparison, "improvement", improved, config)
        results.append(
            PrResult(
                repo,
                number,
                pr.head_sha,
                m.run_id,
                "not-comparable" if reason else "regressed" if flagged else "ok",
                entries[i].data.state.key,
                flagged,
                improved,
                suspected,
                comparison,
                m.producer.get("run_url", ""),
                nearest,
                reason,
                location,
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


def save(results: ResultStore, analysis: Analysis) -> None:
    rows = []
    for e in analysis.timeline:
        row = _state_json(e.data)
        if e.push:
            row["push"] = {
                "status": e.push.status,
                "reason": (
                    e.push.comparison.not_comparable_reason()
                    if e.push.status == "not-comparable" and e.push.comparison
                    else ""
                ),
                "flagged": e.push.flagged,
                "improved": e.push.improved,
                "suspect_range": e.push.suspect_range,
                "comparison": comparison_json(e.push.comparison),
            }
        if e.nightly:
            row["nightly"] = {
                "status": e.nightly.status,
                "reason": e.nightly.reason,
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
    results.replace_dir("alerts", {f"{a.id}.json": asdict(a) for a in analysis.alerts})
    analysis.log.save(results)
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
