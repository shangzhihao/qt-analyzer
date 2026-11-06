"""Write synthetic runs in the contract format, for tests."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qt_analyzer.contract import SCHEMA_VERSION, state_key


@dataclass(frozen=True)
class Case:
    key: str
    family: str
    p: float
    definition: str = "d1"


def make_cases(families: int = 4, per_family: int = 25, p: float = 0.9) -> list[Case]:
    return [
        Case(f"case{f}x{i:02d}", f"family{f}", p)
        for f in range(families)
        for i in range(per_family)
    ]


def manifest(
    run_id: str,
    kind: str,
    components: dict[str, tuple[str, str]],
    created_by: str | None,
    runs: int,
    shards: int,
    pr: dict | None = None,
    created_at: str = "2026-11-06T22:00:00Z",
) -> dict:
    shas = {name: sha for name, (sha, _) in components.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "created_at": created_at,
        "state": {
            "key": state_key(shas),
            "created_by": created_by,
            "components": {
                name: {"repo": f"me/qt-{name}", "sha": sha, "committed_at": at}
                for name, (sha, at) in components.items()
            },
        },
        "pr": pr,
        "runs_per_case": runs,
        "shards": shards,
        "seed": 1,
        "producer": {"repo": "me/qt-graph", "run_url": "https://example.invalid/run"},
        "tools": {},
    }


def report_entries(case: Case, repeat: int, outcome: str) -> list[dict]:
    props = [
        ["qt_case_key", case.key],
        ["qt_definition_hash", case.definition],
        ["qt_family", case.family],
        ["qt_repeat", repeat],
        ["qt_hrid", case.key],
    ]
    nodeid = f"tests/test_x.py::test_case[{case.key}-{repeat}]"

    def entry(when: str, result: str) -> dict:
        return {
            "$report_type": "TestReport",
            "nodeid": nodeid,
            "when": when,
            "outcome": result,
            "user_properties": props,
        }

    if outcome == "error":
        return [entry("setup", "failed"), entry("teardown", "passed")]
    return [entry("setup", "passed"), entry("call", outcome), entry("teardown", "passed")]


def write_run(
    root: Path,
    data: dict,
    cases: list[Case],
    rng: random.Random,
    *,
    drop_shards: tuple[int, ...] = (),
    unfinished_shards: tuple[int, ...] = (),
) -> Path:
    run_dir = root / data["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(data))
    shards = data["shards"]
    for shard in range(1, shards + 1):
        if shard in drop_shards:
            continue
        lines: list[dict[str, Any]] = [{"$report_type": "SessionStart", "pytest_version": "8.3.3"}]
        for index, case in enumerate(cases):
            if index % shards != shard - 1:
                continue
            for repeat in range(data["runs_per_case"]):
                outcome = "passed" if rng.random() < case.p else "failed"
                lines.extend(report_entries(case, repeat, outcome))
        if shard not in unfinished_shards:
            lines.append({"$report_type": "SessionFinish", "exitstatus": 0})
        path = run_dir / f"shard-{shard}-of-{shards}.jsonl.gz"
        with gzip.open(path, "wt") as f:
            f.writelines(json.dumps(line) + "\n" for line in lines)
    return run_dir
