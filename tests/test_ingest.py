import random

from synthetic import make_cases, manifest, write_run

from qt_det.ingest import ingest_pending
from qt_det.stores import LocalDirStore, ResultStore

COMPONENTS = {"graph": ("g1", "2026-11-01T10:00:00Z"), "data": ("d1", "2026-11-01T09:00:00Z")}


def test_counts_cover_every_shard(tmp_path):
    cases = make_cases(families=2, per_family=5, p=1.0)
    write_run(
        tmp_path / "logs",
        manifest("r1", "nightly", COMPONENTS, "graph", 3, 2),
        cases,
        random.Random(0),
    )
    results = ResultStore(tmp_path / "results")

    report = ingest_pending(LocalDirStore(tmp_path / "logs"), results)

    assert report.complete == ["r1"]
    counts = results.get_counts("r1")
    assert len(counts) == 10
    assert all(c.n == 3 and c.passed == 3 for c in counts)


def test_missing_shard_is_incomplete_until_rerun(tmp_path):
    cases = make_cases(families=1, per_family=4)
    data = manifest("r1", "main", COMPONENTS, "graph", 1, 2)
    logs, results = LocalDirStore(tmp_path / "logs"), ResultStore(tmp_path / "results")
    write_run(tmp_path / "logs", data, cases, random.Random(0), drop_shards=(2,))

    first = ingest_pending(logs, results)
    assert first.incomplete == ["r1"]
    assert "missing shard-2-of-2" in results.load_index()["r1"].detail

    write_run(tmp_path / "logs", data, cases, random.Random(0))  # re-run of failed shards
    second = ingest_pending(logs, results)
    assert second.complete == ["r1"]

    third = ingest_pending(logs, results)
    assert third.complete == third.incomplete == third.rejected == []


def test_unfinished_shard_is_incomplete(tmp_path):
    write_run(
        tmp_path / "logs",
        manifest("r1", "main", COMPONENTS, "graph", 1, 2),
        make_cases(1, 4),
        random.Random(0),
        unfinished_shards=(1,),
    )
    report = ingest_pending(LocalDirStore(tmp_path / "logs"), ResultStore(tmp_path / "results"))
    assert report.incomplete == ["r1"]


def test_unknown_schema_is_rejected(tmp_path):
    data = manifest("r1", "main", COMPONENTS, "graph", 1, 1) | {"schema_version": 9}
    write_run(tmp_path / "logs", data, make_cases(1, 2), random.Random(0))
    results = ResultStore(tmp_path / "results")
    report = ingest_pending(LocalDirStore(tmp_path / "logs"), results)
    assert report.rejected == ["r1"]
    assert "schema_version" in results.load_index()["r1"].detail
