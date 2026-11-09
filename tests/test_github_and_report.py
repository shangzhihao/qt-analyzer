import io
import random
import zipfile
from urllib.parse import parse_qs, urlsplit

from synthetic import make_cases, manifest, write_run
from test_analyse import Scenario, with_family

from qt_det.ingest import ingest_pending
from qt_det.report import MARKER, pr_comment, sync_pr_comments
from qt_det.stores import ResultStore
from qt_det.stores.github import GitHubArtifactsStore


def _artifact(artifact_id, name, created_at, url):
    return {"id": artifact_id, "name": name, "created_at": created_at, "expired": False,
            "archive_download_url": url}  # fmt: skip


class FakeGitHub(GitHubArtifactsStore):
    """Artifacts served from a local run directory instead of the API."""

    def __init__(self, run_dirs, cache, stale=()):
        super().__init__("me/qt-app", "token", cache)
        self.blobs, self.listing = {}, []
        next_id = 1
        for run_dir in run_dirs:
            run_id = run_dir.name
            files = {"manifest": [run_dir / "manifest.json"]}
            for shard in sorted(run_dir.glob("shard-*")):
                files[f"shard-{shard.name.split('-')[1]}"] = [shard]
            for suffix, paths in files.items():
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    for path in paths:
                        archive.write(path, path.name)
                self.blobs[f"blob/{next_id}"] = buffer.getvalue()
                self.listing.append(
                    _artifact(
                        next_id, f"qt-{run_id}-{suffix}", "2026-11-06T10:00:00Z", f"blob/{next_id}"
                    )
                )
                next_id += 1
        for name in stale:  # an older artifact of the same name, replaced by a re-run
            self.listing.append(_artifact(999, name, "2026-11-01T00:00:00Z", "blob/broken"))
        self.listing.append(_artifact(1000, "unrelated", "2026-11-06T00:00:00Z", "blob/x"))
        self.downloads = 0

    def get_json(self, url):
        return {"artifacts": self.listing}

    def get_bytes(self, url):
        self.downloads += 1
        return self.blobs[url]


def test_ingest_from_artifacts_uses_the_newest_artifact_and_caches(tmp_path):
    components = {"graph": ("g1", "2026-11-01T10:00:00Z")}
    run_dir = write_run(tmp_path / "logs", manifest("graph-1", "main", components, "graph", 2, 2),
                        make_cases(2, 5), random.Random(0))  # fmt: skip
    store = FakeGitHub([run_dir], tmp_path / "cache", stale=["qt-app-1-shard-1"])
    results = ResultStore(tmp_path / "results")

    assert store.list_runs() == ["graph-1"]
    report = ingest_pending(store, results)

    assert report.complete == ["graph-1"]
    assert sum(c.n for c in results.get_counts("graph-1")) == 20
    downloads = store.downloads
    store.get_shard_logs("graph-1")
    assert store.downloads == downloads  # cached by artifact id


class FakeApi:
    """GitHub issue comments in memory: list, create, update."""

    def __init__(self):
        self.comments = {}  # id -> (issue url, body)
        self.calls = []

    def __call__(self, method, url, body):
        self.calls.append((method, url.split("?")[0], body["body"] if body else None))
        if method == "GET":
            base = url.split("?")[0]
            page = int(parse_qs(urlsplit(url).query).get("page", ["1"])[0])
            items = [{"id": i, "body": b} for i, (u, b) in self.comments.items() if u == base]
            return items if page == 1 else []
        if method == "POST":
            new_id = len(self.comments) + 1
            self.comments[new_id] = (url, body["body"])
            return {"id": new_id}
        comment_id = int(url.rsplit("/", 1)[1])
        issue_url = self.comments[comment_id][0]
        self.comments[comment_id] = (issue_url, body["body"])
        return {"id": comment_id}

    def writes(self):
        return [(m, u) for m, u, _ in self.calls if m != "GET"]


def test_pr_comment_is_created_once_and_updated_on_a_new_result(tmp_path):
    cases = make_cases(families=4, per_family=25, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("data", "d1")
    s.pr("graph", 5, "g1", "h1", with_family(cases, "family2", 0.1), "2026-11-03T01:00:00Z")
    analysis = s.analyse()
    api = FakeApi()

    assert sync_pr_comments(analysis.prs, api) == ["me/qt-app#5"]
    assert sync_pr_comments(analysis.prs, api) == []  # repeated pass: nothing written
    assert api.writes() == [("POST", "https://api.github.com/repos/me/qt-app/issues/5/comments")]
    (body,) = [b for _, b in api.comments.values()]
    assert MARKER in body and "Possible regression" in body and "family2" in body

    s.pr("graph", 5, "g1", "h2", cases, "2026-11-03T02:00:00Z")
    analysis = s.analyse()
    assert sync_pr_comments(analysis.prs, api) == ["me/qt-app#5"]
    assert api.writes()[-1] == (
        "PATCH",
        "https://api.github.com/repos/me/qt-app/issues/comments/1",
    )
    assert len(api.comments) == 1
    assert "No significant drop" in api.comments[1][1]


def test_an_existing_comment_is_found_by_its_marker(tmp_path):
    # e.g. results were reset or a pass crashed after commenting: no duplicate
    cases = make_cases(families=2, per_family=10, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.pr("graph", 8, "g1", "h1", cases, "2026-11-03T01:00:00Z")
    api = FakeApi()
    api.comments[7] = ("https://api.github.com/repos/me/qt-app/issues/8/comments",
                       MARKER + "\nold report")  # fmt: skip

    sync_pr_comments(s.analyse().prs, api)

    assert api.writes() == [("PATCH", "https://api.github.com/repos/me/qt-app/issues/comments/7")]
    assert len(api.comments) == 1


def test_pending_baseline_comment_when_nothing_is_measured_yet(tmp_path):
    s = Scenario(tmp_path, make_cases(2, 5))
    s.pr("graph", 6, "g-unmeasured", "h1", make_cases(2, 5), "2026-11-03T01:00:00Z")
    (pr,) = s.analyse().prs
    assert "Baseline pending" in pr_comment(pr)


def test_nearest_baseline_is_mentioned(tmp_path):
    s = Scenario(tmp_path, make_cases(2, 5))
    s.commit("graph", "g1")
    s.pr("graph", 6, "g-unmeasured", "h1", make_cases(2, 5), "2026-11-03T01:00:00Z")
    (pr,) = s.analyse().prs
    assert pr.nearest_baseline
    assert "nearest measured `main` state" in pr_comment(pr)
