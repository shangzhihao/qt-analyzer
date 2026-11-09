from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from qt_det.stores.base import CaseCount, IndexEntry

_SCHEMA = pa.schema(
    [
        ("case_key", pa.string()),
        ("definition_hash", pa.string()),
        ("family", pa.string()),
        ("hrid", pa.string()),
        ("n", pa.int32()),
        ("passed", pa.int32()),
        ("failed", pa.int32()),
        ("error", pa.int32()),
    ]
)


def _safe(run_id: str) -> str:
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError(f"unsafe run id {run_id!r}")
    return run_id


class ResultStore:
    """Analysis results in a directory with the layout of the `qt-results` branch."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # index ---------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.root / "index" / "runs.jsonl"

    def load_index(self) -> dict[str, IndexEntry]:
        if not self._index_path.exists():
            return {}
        entries = {}
        for line in self._index_path.read_text().splitlines():
            if line.strip():
                entry = IndexEntry(**json.loads(line))
                entries[entry.run_id] = entry
        return entries

    def save_index(self, entries: dict[str, IndexEntry]) -> None:
        lines = [json.dumps(asdict(e), sort_keys=True) for _, e in sorted(entries.items())]
        self._write(self._index_path, "\n".join(lines) + ("\n" if lines else ""))

    # per-run counts ------------------------------------------------------
    def put_counts(self, run_id: str, counts: list[CaseCount]) -> None:
        path = self.root / "runs" / f"{_safe(run_id)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {f.name: [getattr(c, f.name) for c in counts] for f in fields(CaseCount)}
        pq.write_table(pa.table(columns, schema=_SCHEMA), path)

    def get_counts(self, run_id: str) -> list[CaseCount]:
        table = pq.read_table(self.root / "runs" / f"{_safe(run_id)}.parquet")
        return [CaseCount(**row) for row in table.to_pylist()]

    # derived JSON files ---------------------------------------------------
    def write_json(self, relative: str, data: Any) -> None:
        self._write(self.root / relative, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def read_json(self, relative: str) -> Any:
        path = self.root / relative
        return json.loads(path.read_text()) if path.exists() else None

    def read_jsonl(self, relative: str) -> list[dict[str, Any]]:
        path = self.root / relative
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def write_jsonl(self, relative: str, rows: list[dict[str, Any]]) -> None:
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        self._write(self.root / relative, text)

    def replace_dir(self, relative: str, files: dict[str, Any]) -> None:
        """Make `relative` hold exactly `files` (name -> JSON data)."""
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.json"):
            if stale.name not in files:
                stale.unlink()
        for name, data in files.items():
            self.write_json(f"{relative}/{name}", data)

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
