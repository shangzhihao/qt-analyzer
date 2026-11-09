"""Keep a local copy of the published results (the `qt-results` branch)."""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_REPO = "shangzhihao/qt-det"
BRANCH = "qt-results"


def fetch_results(cache: Path, repo: str = DEFAULT_REPO, url: str | None = None) -> Path:
    """Clone or update `<cache>/qt-results` and return its path.

    The folder belongs to the analyzer's cache: an update replaces its content
    with the published branch.
    """
    dest = cache / BRANCH
    source = url or f"https://github.com/{repo}.git"
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", BRANCH],
                       check=True)  # fmt: skip
        subprocess.run(["git", "-C", str(dest), "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                        "--single-branch", source, str(dest)], check=True)  # fmt: skip
    return dest
