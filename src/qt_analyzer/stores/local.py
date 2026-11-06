from __future__ import annotations

from pathlib import Path

from qt_analyzer.contract import Manifest


class LocalDirStore:
    """Runs on disk: `<root>/<run_id>/manifest.json` and `shard-<i>-of-<n>.jsonl.gz`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list_runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/manifest.json"))

    def get_manifest(self, run_id: str) -> Manifest:
        return Manifest.load(self.root / run_id / "manifest.json")

    def get_shard_logs(self, run_id: str) -> list[Path]:
        return sorted((self.root / run_id).glob("shard-*.jsonl*"))
