"""The contract between the test producer (qt-app) and the analyzer.

A run is a manifest plus one pytest report log per shard. This module is the
only place that knows their format; everything else works on the dataclasses.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# 2: the analyzer computes state keys itself; PR manifests name their baseline.
SCHEMA_VERSION = 2

Kind = Literal["pr", "main", "nightly"]
Outcome = Literal["passed", "failed", "error"]
KINDS: tuple[Kind, ...] = ("pr", "main", "nightly")

# user_properties written by the qt pytest plugin
CASE_KEY = "qt_case_key"
DEFINITION_HASH = "qt_definition_hash"
FAMILY = "qt_family"
REPEAT = "qt_repeat"
HRID = "qt_hrid"


class ContractError(ValueError):
    """A manifest or log does not follow the contract."""


def state_key(components: Mapping[str, str]) -> str:
    """Identify a state by the sorted `name=sha` pairs of its components.

    Only the analyzer computes this; producers send component versions only.
    """
    text = "\n".join(f"{name}={sha}" for name, sha in sorted(components.items()))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ComponentVersion:
    repo: str
    sha: str
    committed_at: str


@dataclass(frozen=True)
class State:
    key: str
    created_by: str | None
    components: dict[str, ComponentVersion]

    @property
    def shas(self) -> dict[str, str]:
        return {name: c.sha for name, c in self.components.items()}

    @property
    def committed_at(self) -> str:
        """Commit time of the version that created this state (its timeline position)."""
        if self.created_by is None:
            return max(c.committed_at for c in self.components.values())
        return self.components[self.created_by].committed_at


@dataclass(frozen=True)
class PrInfo:
    component: str
    repo: str
    number: int
    head_sha: str
    merge_sha: str
    baseline: dict[str, str]  # the main state to compare with: component -> sha


@dataclass(frozen=True)
class Manifest:
    run_id: str
    kind: Kind
    created_at: str
    state: State
    pr: PrInfo | None
    runs_per_case: int
    shards: int
    seed: int
    producer: dict[str, str]
    tools: dict[str, str]
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Manifest:
        try:
            version = data["schema_version"]
            if version != SCHEMA_VERSION:
                raise ContractError(f"unsupported schema_version {version!r}")
            kind = data["kind"]
            if kind not in KINDS:
                raise ContractError(f"unknown kind {kind!r}")
            raw_state = data["state"]
            components = {
                name: ComponentVersion(c["repo"], c["sha"], c["committed_at"])
                for name, c in raw_state["components"].items()
            }
            shas = {name: c.sha for name, c in components.items()}
            state = State(state_key(shas), raw_state.get("created_by"), components)
            raw_pr = data.get("pr")
            pr = None
            if raw_pr is not None:
                pr = PrInfo(
                    raw_pr["component"],
                    raw_pr["repo"],
                    int(raw_pr["number"]),
                    raw_pr["head_sha"],
                    raw_pr["merge_sha"],
                    {str(k): str(v) for k, v in raw_pr["baseline"].items()},
                )
                if set(pr.baseline) != set(components):
                    raise ContractError("pr.baseline must name the same components as the state")
            if (kind == "pr") != (pr is not None):
                raise ContractError("pr must be set exactly when kind is 'pr'")
            if kind != "pr" and state.created_by not in components:
                raise ContractError("a main or nightly state needs created_by naming a component")
            manifest = cls(
                run_id=data["run_id"],
                kind=kind,
                created_at=data["created_at"],
                state=state,
                pr=pr,
                runs_per_case=int(data["runs_per_case"]),
                shards=int(data["shards"]),
                seed=int(data["seed"]),
                producer=dict(data.get("producer", {})),
                tools=dict(data.get("tools", {})),
                raw=dict(data),
            )
        except (KeyError, TypeError) as error:
            raise ContractError(f"malformed manifest: {error!r}") from error
        if manifest.runs_per_case < 1 or manifest.shards < 1:
            raise ContractError("runs_per_case and shards must be at least 1")
        return manifest

    @classmethod
    def load(cls, path: Path) -> Manifest:
        try:
            return cls.from_json(json.loads(path.read_text()))
        except json.JSONDecodeError as error:
            raise ContractError(f"manifest is not JSON: {error}") from error


@dataclass(frozen=True)
class Execution:
    """One execution of one case (one repeat), as recorded in a shard log."""

    case_key: str
    definition_hash: str
    family: str
    hrid: str
    repeat: int
    outcome: Outcome


@dataclass(frozen=True)
class ShardLog:
    executions: list[Execution]
    finished: bool


def _read_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from f


def parse_shard_log(path: Path) -> ShardLog:
    """Turn a pytest-reportlog file into one execution per test item.

    Per item: setup or teardown failure -> error; call passed/failed -> passed/failed;
    skipped items are dropped; entries with outcome "rerun" are ignored.
    """
    finished = False
    phases: dict[str, dict[str, str]] = {}
    props: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(_read_lines(path), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"{path.name}:{number} is not JSON") from error
        report_type = entry.get("$report_type")
        if report_type == "SessionFinish":
            finished = True
            continue
        if report_type != "TestReport" or entry.get("outcome") == "rerun":
            continue
        nodeid = entry["nodeid"]
        phases.setdefault(nodeid, {})[entry["when"]] = entry["outcome"]
        user_properties = dict(entry.get("user_properties") or [])
        if CASE_KEY in user_properties:
            props[nodeid] = user_properties

    executions = []
    for nodeid, by_phase in phases.items():
        if nodeid not in props:
            raise ContractError(f"{path.name}: {nodeid} has no {CASE_KEY} property")
        outcome = _outcome(by_phase)
        if outcome is None:
            continue
        p = props[nodeid]
        executions.append(
            Execution(
                case_key=str(p[CASE_KEY]),
                definition_hash=str(p[DEFINITION_HASH]),
                family=str(p.get(FAMILY, "")),
                hrid=str(p.get(HRID, nodeid)),
                repeat=int(p.get(REPEAT, 0)),
                outcome=outcome,
            )
        )
    return ShardLog(executions, finished)


def _outcome(by_phase: Mapping[str, str]) -> Outcome | None:
    setup, call, teardown = (by_phase.get(w) for w in ("setup", "call", "teardown"))
    if setup == "failed" or teardown == "failed":
        return "error"
    if setup == "skipped" or call == "skipped" or call is None:
        return None
    return "passed" if call == "passed" else "failed"
