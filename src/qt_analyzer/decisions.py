"""The decision log: verdicts are taken once and never rewritten.

Each pass may recompute evidence, but a state that has a `state.decided`
record keeps its verdict, reference and alert effects. New states are decided
against the reference and alerts in force at their position, so a threshold
change or late evidence never moves an alert that was already reported.

Records in `decisions.jsonl` (append-only):
- state.decided: state, verdict, reference after this state, alert effect
- alert.opened / alert.extended / alert.closed
- reference.moved
- pr.reported: a PR run's verdict, once its stated baseline is measured
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from qt_analyzer.stores.results import ResultStore

PATH = "decisions.jsonl"


def config_hash(config: Any) -> str:
    text = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def alert_id(state_key: str, existing: set[str]) -> str:
    base = state_key.removeprefix("sha256:")[:16]
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


@dataclass
class DecisionLog:
    records: list[dict[str, Any]] = field(default_factory=list)
    new: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, results: ResultStore) -> DecisionLog:
        return cls(results.read_jsonl(PATH))

    def save(self, results: ResultStore) -> None:
        if self.new:
            results.write_jsonl(PATH, self.records)

    def append(self, kind: str, **data: Any) -> dict[str, Any]:
        record = {"kind": kind, "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), **data}
        self.records.append(record)
        self.new.append(record)
        return record

    def decided(self) -> dict[str, dict[str, Any]]:
        return {r["state"]: r for r in self.records if r["kind"] == "state.decided"}

    def pr_reports(self) -> dict[str, dict[str, Any]]:
        return {r["run_id"]: r for r in self.records if r["kind"] == "pr.reported"}

    def alert_ids(self) -> set[str]:
        return {r["alert"] for r in self.records if r["kind"] == "alert.opened"}
