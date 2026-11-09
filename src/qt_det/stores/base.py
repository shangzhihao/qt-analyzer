from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from qt_det.contract import Manifest

RunStatus = Literal["complete", "incomplete", "rejected"]


class LogStore(Protocol):
    """Raw runs: a manifest and one gzipped report log per shard."""

    def list_runs(self) -> list[str]: ...

    def get_manifest(self, run_id: str) -> Manifest: ...

    def get_shard_logs(self, run_id: str) -> list[Path]: ...


@dataclass(frozen=True)
class CaseCount:
    """Executions of one case version in one run."""

    case_key: str
    definition_hash: str
    family: str
    hrid: str
    n: int
    passed: int
    failed: int
    error: int


@dataclass
class IndexEntry:
    run_id: str
    status: RunStatus
    detail: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
