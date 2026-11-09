"""Turn raw runs into per-case counts in the ResultStore.

Every call ingests everything outstanding: runs missing from the index and runs
recorded as incomplete (a re-run of failed shards can complete them). Ingesting
a complete or rejected run again is a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from qt_det.contract import ContractError, Execution, Manifest, parse_shard_log
from qt_det.stores.base import CaseCount, IndexEntry, LogStore
from qt_det.stores.results import ResultStore


@dataclass
class IngestReport:
    complete: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def ingest_pending(logs: LogStore, results: ResultStore) -> IngestReport:
    index = results.load_index()
    report = IngestReport()
    for run_id in logs.list_runs():
        known = index.get(run_id)
        if known is not None and known.status != "incomplete":
            continue
        entry, counts = _ingest_run(logs, run_id)
        if counts is not None:
            results.put_counts(run_id, counts)
        index[run_id] = entry
        getattr(report, entry.status).append(run_id)
    results.save_index(index)
    return report


def _ingest_run(logs: LogStore, run_id: str) -> tuple[IndexEntry, list[CaseCount] | None]:
    try:
        manifest = logs.get_manifest(run_id)
    except ContractError as error:
        return IndexEntry(run_id, "rejected", str(error)), None
    if manifest.run_id != run_id:
        return IndexEntry(run_id, "rejected", f"manifest says run_id {manifest.run_id}"), None

    paths = logs.get_shard_logs(run_id)
    problem = _shard_problem(manifest, paths)
    if problem:
        return IndexEntry(run_id, "incomplete", problem, manifest.raw), None
    executions: list[Execution] = []
    for path in paths:
        try:
            shard = parse_shard_log(path)
        except ContractError as error:
            return IndexEntry(run_id, "rejected", str(error), manifest.raw), None
        if not shard.finished:
            detail = f"{path.name} has no SessionFinish"
            return IndexEntry(run_id, "incomplete", detail, manifest.raw), None
        executions.extend(shard.executions)
    return IndexEntry(run_id, "complete", "", manifest.raw), aggregate(executions)


def _shard_problem(manifest: Manifest, paths: list[Path]) -> str:
    expected = {f"shard-{i}-of-{manifest.shards}" for i in range(1, manifest.shards + 1)}
    found = {p.name.split(".")[0] for p in paths}
    if missing := sorted(expected - found):
        return f"missing {', '.join(missing)}"
    if unexpected := sorted(found - expected):
        return f"unexpected {', '.join(unexpected)}"
    return ""


def aggregate(executions: list[Execution]) -> list[CaseCount]:
    """Count outcomes per case version (case_key, definition_hash)."""
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "error": 0}
    )
    labels: dict[tuple[str, str], tuple[str, str]] = {}
    for e in executions:
        key = (e.case_key, e.definition_hash)
        totals[key][e.outcome] += 1
        labels[key] = (e.family, e.hrid)
    counts = []
    for key, t in sorted(totals.items()):
        family, hrid = labels[key]
        n = t["passed"] + t["failed"] + t["error"]
        counts.append(CaseCount(key[0], key[1], family, hrid, n, **t))
    return counts
