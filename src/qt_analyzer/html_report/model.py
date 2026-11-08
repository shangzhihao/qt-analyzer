"""The report model: everything the report shows, computed in Python.

The HTML front end only draws, sorts and searches this data; a later
interactive front end (e.g. Streamlit) can reuse `build_model` unchanged.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qt_analyzer.analyse import Analysis, Comparison, Config, Counts, analyse, pass_rate
from qt_analyzer.gate_compare import build_gates
from qt_analyzer.locators import location_from_json
from qt_analyzer.stores.results import ResultStore

FAILURES_PER_CASE = 3
FAILURE_LENGTH = 300


def _short(key: str | None) -> str:
    return key.removeprefix("sha256:")[:10] if key else ""


def _num(x: float | None) -> float | None:
    """JSON-safe number: NaN and infinities become null."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 4)


def _commit_url(repo: str, sha: str) -> str:
    if not repo or "/" not in repo or "+" in sha:
        return ""
    return f"https://github.com/{repo}/commit/{sha}"


def _rates_by_family(counts: Counts) -> dict[str, float]:
    passed: dict[str, int] = defaultdict(int)
    judged: dict[str, int] = defaultdict(int)
    for agg in counts.values():
        passed[agg.family] += agg.passed
        judged[agg.family] += agg.judged
    return {f: passed[f] / judged[f] for f in judged if judged[f]}


def _overall_z(comparison: Comparison | None) -> float | None:
    if comparison is None or not comparison.comparable:
        return None
    return _num(comparison.overall.z)


def _coverage(comparison: Comparison | None) -> dict[str, int]:
    if comparison is None:
        return {"added": 0, "removed": 0, "edited": 0}
    c = comparison.coverage
    return {"added": len(c.added), "removed": len(c.removed), "edited": len(c.edited)}


def _state_rows(analysis: Analysis) -> list[dict[str, Any]]:
    rows = []
    for e in analysis.timeline:
        d = e.data
        created = d.state.created_by or ""
        version = d.state.components.get(created) if created else None
        push = e.push
        nightly = e.nightly
        rows.append({
            "key": d.state.key,
            "short": _short(d.state.key),
            "created_by": created,
            "sha": version.sha[:7] if version else "",
            "commit_url": _commit_url(version.repo, version.sha) if version else "",
            "committed_at": d.state.committed_at,
            "components": {n: c.sha[:7] for n, c in d.state.components.items()},
            "main_rate": _num(pass_rate(d.counts["main"])),
            "nightly_rate": _num(pass_rate(d.counts["nightly"])),
            "main_runs": len(d.runs["main"]),
            "nightly_runs": len(d.runs["nightly"]),
            "push": None if push is None else {
                "status": push.status, "flagged": push.flagged, "improved": push.improved,
                "z": _overall_z(push.comparison),
                "reason": push.comparison.not_comparable_reason()
                if push.status == "not-comparable" and push.comparison else "",
            },
            "nightly": None if nightly is None else {
                "status": nightly.status, "step_flagged": nightly.step_flagged,
                "reference_flagged": nightly.reference_flagged, "reason": nightly.reason,
                "z": _overall_z(nightly.step),
            },
            "coverage": _coverage((nightly.step if nightly else None)
                                  or (push.comparison if push else None)),
        })  # fmt: skip
    return rows


def _heat(analysis: Analysis) -> dict[str, Any]:
    """Topic x state pass rates (nightly runs preferred), change vs the previous
    measured state, and which cells a gate flagged."""
    per_state = []
    for e in analysis.timeline:
        counts = e.data.counts["nightly"] or e.data.counts["main"]
        per_state.append(_rates_by_family(counts))
    topics = sorted({t for rates in per_state for t in rates})
    rate, delta, flag = [], [], []
    previous: dict[str, float] = {}
    for e, rates in zip(analysis.timeline, per_state, strict=True):
        regressed = set()
        improved = set()
        if e.nightly:
            regressed |= set(e.nightly.step_flagged) | set(e.nightly.reference_flagged)
        if e.push:
            regressed |= set(e.push.flagged)
            improved |= set(e.push.improved)
        rate.append([_num(rates.get(t)) for t in topics])
        delta.append([_num(rates[t] - previous[t]) if t in rates and t in previous else None
                      for t in topics])  # fmt: skip
        flag.append(["regressed" if t in regressed else "improved" if t in improved else None
                     for t in topics])  # fmt: skip
        previous = {**previous, **rates}
    # Colour range: from the lowest observed topic rate (rounded down to 5 pts) to
    # 100 %, so the usual 85-100 % band is not squeezed into one colour step.
    observed = [v for row in rate for v in row if v is not None]
    low = math.floor(min(observed) * 20) / 20 if observed else 0.0
    return {"topics": topics, "rate": rate, "delta": delta, "flag": flag,
            "rate_domain": [min(low, 0.95), 1.0]}  # fmt: skip


def _closed(closed_by: str | None) -> str:
    if closed_by and closed_by.startswith("sha256:"):
        return _short(closed_by)
    return closed_by or ""


def _alerts(analysis: Analysis) -> list[dict[str, Any]]:
    history: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in analysis.log.records:
        if r["kind"].startswith("alert."):
            history[r["alert"]].append(
                {"at": r["at"], "kind": r["kind"], "state": _short(r["state"])}
            )
    out = []
    for a in analysis.alerts:
        where = location_from_json(a.where) if a.where else None
        out.append({
            "id": a.id, "state": _short(a.state), "open": a.open, "scopes": a.scopes,
            "suspect_range": [_short(k) for k in a.suspect_range],
            "reference": _short(a.reference),
            "still_regressed_at": [_short(k) for k in a.still_regressed_at],
            "closed_by": _closed(a.closed_by),
            "where": _location(where), "history": history.get(a.id, []),
        })  # fmt: skip
    return out


def _location(location) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "direction": location.direction,
        "scopes": [{"scope": s.scope, "z": _num(s.z), "rate_b": _num(s.rate_b),
                    "rate_s": _num(s.rate_s)} for s in location.scopes],
        "cases": [{"hrid": c.hrid, "family": c.family, "before": f"{c.passed_b}/{c.judged_b}",
                   "after": f"{c.passed_s}/{c.judged_s}", "p": _num(c.p), "q": _num(c.q)}
                  for c in location.cases],
    }  # fmt: skip


def _prs(analysis: Analysis) -> list[dict[str, Any]]:
    out = []
    for p in analysis.prs:
        c = p.comparison
        out.append({
            "repo": p.repo, "number": p.number, "head": p.head_sha[:7], "run_id": p.run_id,
            "status": p.status, "flagged": p.flagged, "improved": p.improved,
            "nearest_baseline": p.nearest_baseline, "reason": p.reason, "run_url": p.run_url,
            "url": f"https://github.com/{p.repo}/pull/{p.number}" if "/" in p.repo else "",
            "overall": None if c is None or not c.comparable else {
                "rate_b": _num(c.overall.rate_b), "rate_s": _num(c.overall.rate_s),
                "z": _num(c.overall.z), "cases": c.overall.cases},
            "where": _location(p.location),
        })  # fmt: skip
    return out


def _gate_rows(analysis: Analysis, config: Config) -> dict[str, Any]:
    """What every registered gate would have said on each comparison."""
    gates = build_gates(None, config.gate_params)
    rows = []
    for e in analysis.timeline:
        if e.nightly and e.nightly.step is not None and e.nightly.step.comparable:
            comparison, label = e.nightly.step, "nightly vs previous"
        elif e.push and e.push.comparison is not None and e.push.comparison.comparable:
            comparison, label = e.push.comparison, "1 run vs baseline"
        else:
            continue
        rows.append(_gate_row(f"{e.data.state.created_by}@{_short(e.data.state.key)}", label,
                              comparison, gates))  # fmt: skip
    for p in analysis.prs:
        if p.comparison is not None and p.comparison.comparable:
            rows.append(_gate_row(f"PR {p.repo}#{p.number}", "1 run vs main", p.comparison, gates))
    return {"names": list(gates), "decides": config.gate, "rows": rows}


def _gate_row(label: str, comparison_kind: str, comparison: Comparison, gates) -> dict[str, Any]:
    verdicts = {}
    for name, gate in gates.items():
        v = gate.judge(comparison)
        verdicts[name] = {"regressed": v.regressed, "improved": v.improved}
    return {"label": label, "comparison": comparison_kind, "verdicts": verdicts}


def _cases(analysis: Analysis, failures: dict[str, list[str]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    n_states = len(analysis.timeline)
    for i, e in enumerate(analysis.timeline):
        for (key, definition), agg in e.data.all_counts().items():
            case = by_case.setdefault(key, {
                "key": key, "hrid": agg.hrid, "family": agg.family,
                "rates": [None] * n_states, "runs": [0] * n_states,
                "definitions": [None] * n_states,
            })  # fmt: skip
            if agg.judged:
                case["rates"][i] = _num(agg.passed / agg.judged)
            case["runs"][i] = agg.judged
            case["definitions"][i] = definition
    out = []
    for case in by_case.values():
        seen = [d for d in case["definitions"] if d is not None]
        changed = [i for i in range(1, n_states)
                   if case["definitions"][i] is not None
                   and case["definitions"][i - 1] is not None
                   and case["definitions"][i] != case["definitions"][i - 1]]  # fmt: skip
        latest = next((r for r in reversed(case["rates"]) if r is not None), None)
        kind = ("reliable" if latest is not None and latest >= 0.95
                else "failing" if latest is not None and latest <= 0.05 else "flaky")  # fmt: skip
        out.append({
            "key": case["key"], "hrid": case["hrid"], "family": case["family"], "kind": kind,
            "rates": case["rates"], "runs": case["runs"], "definition_changed_at": changed,
            "definitions": len(set(seen)), "failures": failures.get(case["key"], []),
        })  # fmt: skip
    return sorted(out, key=lambda c: (c["family"], c["hrid"]))


def _runs(results: ResultStore) -> list[dict[str, Any]]:
    rows = []
    for run_id, entry in sorted(results.load_index().items()):
        m = entry.manifest or {}
        errors = judged = 0
        if entry.status == "complete":
            try:
                counts = results.get_counts(run_id)
                errors = sum(c.error for c in counts)
                judged = sum(c.passed + c.failed for c in counts)
            except FileNotFoundError:
                pass
        rows.append({"run_id": run_id, "status": entry.status, "detail": entry.detail,
                     "kind": m.get("kind", ""), "created_at": m.get("created_at", ""),
                     "shards": m.get("shards"), "runs_per_case": m.get("runs_per_case"),
                     "executions": judged, "errors": errors,
                     "url": (m.get("producer") or {}).get("run_url", "")})  # fmt: skip
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


def failure_samples(log_paths: list[Path]) -> dict[str, list[str]]:
    """Up to a few distinct failure messages per case from raw report logs."""
    samples: dict[str, list[str]] = defaultdict(list)
    for path in log_paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if '"TestReport"' not in line:
                    continue
                entry = json.loads(line)
                if entry.get("when") != "call" or entry.get("outcome") != "failed":
                    continue
                key = dict(entry.get("user_properties") or []).get("qt_case_key")
                message = _message(entry.get("longrepr"))
                if key and message and message not in samples[key]:
                    if len(samples[key]) < FAILURES_PER_CASE:
                        samples[key].append(message)
    return dict(samples)


def _message(longrepr: Any) -> str:
    if isinstance(longrepr, dict):
        crash = longrepr.get("reprcrash") or {}
        text = crash.get("message") or ""
    else:
        text = str(longrepr or "")
        text = text.strip().splitlines()[-1] if text.strip() else ""
    return text[:FAILURE_LENGTH]


def build_model(
    results: ResultStore,
    config: Config,
    *,
    failures: dict[str, list[str]] | None = None,
    source: str = "",
) -> dict[str, Any]:
    analysis = analyse(results, config)
    states = _state_rows(analysis)
    open_alerts = [a for a in analysis.alerts if a.open]
    latest = states[-1] if states else None
    runs = _runs(results)
    return {
        "meta": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "source": source or str(results.root),
            "gate": config.gate,
            "locator": config.locator,
            "failure_messages": failures is not None,
        },
        "overview": {
            "latest": latest,
            "reference": _short(analysis.reference),
            "states": len(states),
            "open_alerts": len(open_alerts),
            "alerts": len(analysis.alerts),
            "prs": len(analysis.prs),
            "prs_regressed": sum(p.status == "regressed" for p in analysis.prs),
            "last_run_at": max((r["created_at"] for r in runs if r["created_at"]), default=""),
            "runs": {
                status: sum(r["status"] == status for r in runs)
                for status in ("complete", "incomplete", "rejected")
            },  # fmt: skip
        },
        "states": states,
        "heat": _heat(analysis),
        "alerts": _alerts(analysis),
        "prs": _prs(analysis),
        "gates": _gate_rows(analysis, config),
        "cases": _cases(analysis, failures or {}),
        "runs": runs,
        "decisions": analysis.log.records,
    }
