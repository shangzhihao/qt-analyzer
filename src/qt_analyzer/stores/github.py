from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from qt_analyzer.contract import Manifest

API = "https://api.github.com"
_MANIFEST = "-manifest"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class GitHubArtifactsStore:
    """Runs stored as Actions artifacts of the producer repo.

    `qt-<run_id>-manifest` holds manifest.json and `qt-<run_id>-shard-<i>` one
    gzipped report log. A re-run that overwrites an artifact leaves the newest
    one, so the newest artifact of each name wins. Downloads are cached by
    artifact id.
    """

    def __init__(self, repo: str, token: str | None, cache: Path) -> None:
        self.repo = repo
        self.token = token
        self.cache = cache
        self._artifacts: dict[str, dict[str, Any]] | None = None

    # HTTP ------------------------------------------------------------------
    def _request(self, url: str, auth: bool = True) -> urllib.request.Request:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(url, headers=headers)

    def get_json(self, url: str) -> Any:
        with urllib.request.urlopen(self._request(url), timeout=60) as response:
            return json.load(response)

    def get_bytes(self, url: str) -> bytes:
        """Download through GitHub's redirect without sending the token to blob storage."""
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(self._request(url), timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers["Location"]
        with urllib.request.urlopen(self._request(location, auth=False), timeout=300) as response:
            return response.read()

    # artifacts -------------------------------------------------------------
    def artifacts(self) -> dict[str, dict[str, Any]]:
        if self._artifacts is None:
            newest: dict[str, dict[str, Any]] = {}
            page = 1
            while True:
                url = f"{API}/repos/{self.repo}/actions/artifacts?per_page=100&page={page}"
                batch = self.get_json(url)["artifacts"]
                for artifact in batch:
                    name = artifact["name"]
                    if not name.startswith("qt-") or artifact.get("expired"):
                        continue
                    if name not in newest or artifact["created_at"] > newest[name]["created_at"]:
                        newest[name] = artifact
                if len(batch) < 100:
                    break
                page += 1
            self._artifacts = newest
        return self._artifacts

    def _files(self, artifact: dict[str, Any]) -> Path:
        target = self.cache / str(artifact["id"])
        if not target.exists():
            data = self.get_bytes(artifact["archive_download_url"])
            tmp = target.with_suffix(".tmp")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                archive.extractall(tmp)
            tmp.rename(target)
        return target

    # LogStore --------------------------------------------------------------
    def list_runs(self) -> list[str]:
        return sorted(
            name[len("qt-") : -len(_MANIFEST)]
            for name in self.artifacts()
            if name.endswith(_MANIFEST)
        )

    def get_manifest(self, run_id: str) -> Manifest:
        return Manifest.load(
            self._files(self.artifacts()[f"qt-{run_id}{_MANIFEST}"]) / "manifest.json"
        )

    def get_shard_logs(self, run_id: str) -> list[Path]:
        prefix = f"qt-{run_id}-shard-"
        paths: list[Path] = []
        for name, artifact in sorted(self.artifacts().items()):
            if name.startswith(prefix) and name[len(prefix) :].isdigit():
                paths.extend(sorted(self._files(artifact).glob("shard-*.jsonl*")))
        return paths
