import gzip
import json

import pytest
from synthetic import Case, manifest, report_entries

from qt_det.contract import ContractError, Manifest, parse_shard_log, state_key


def _components():
    return {"graph": ("g1", "2026-11-01T10:00:00Z"), "data": ("d1", "2026-11-01T09:00:00Z")}


def test_state_key_ignores_component_order():
    assert state_key({"graph": "g1", "data": "d1"}) == state_key({"data": "d1", "graph": "g1"})
    assert state_key({"graph": "g1", "data": "d1"}) != state_key({"graph": "g1", "data": "d2"})


def test_manifest_round_trip():
    m = Manifest.from_json(manifest("r1", "main", _components(), "graph", 1, 2))
    assert m.state.shas == {"graph": "g1", "data": "d1"}
    assert m.state.committed_at == "2026-11-01T10:00:00Z"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 1}, "schema_version"),
        ({"kind": "weekly"}, "kind"),
        ({"pr": {"component": "graph"}}, "malformed"),
        ({"shards": 0}, "at least 1"),
    ],
)
def test_manifest_rejects_contract_violations(change, message):
    data = manifest("r1", "main", _components(), "graph", 1, 2) | change
    with pytest.raises(ContractError, match=message):
        Manifest.from_json(data)


def test_the_analyzer_computes_the_state_key():
    data = manifest("r1", "main", _components(), "graph", 1, 2)
    assert "key" not in data["state"]
    assert Manifest.from_json(data).state.key == state_key({"graph": "g1", "data": "d1"})


def _pr(baseline):
    return {"component": "graph", "repo": "me/qt-app", "number": 3, "head_sha": "h",
            "merge_sha": "m", "baseline": baseline}  # fmt: skip


def test_pr_manifest_names_its_baseline():
    data = manifest("r1", "pr", _components(), None, 1, 2, pr=_pr({"graph": "g0", "data": "d1"}))
    assert Manifest.from_json(data).pr.baseline == {"graph": "g0", "data": "d1"}


def test_pr_baseline_must_cover_the_same_components():
    data = manifest("r1", "pr", _components(), None, 1, 2, pr=_pr({"graph": "g0"}))
    with pytest.raises(ContractError, match="same components"):
        Manifest.from_json(data)


def test_pr_kind_requires_pr_info():
    with pytest.raises(ContractError, match="exactly when"):
        Manifest.from_json(manifest("r1", "pr", _components(), None, 1, 1))


def _write(path, entries):
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)


def test_outcomes_from_report_log(tmp_path):
    ok, bad, broken = Case("a", "f", 1), Case("b", "f", 1), Case("c", "f", 1)
    rerun = report_entries(bad, 0, "failed")[1] | {"outcome": "rerun"}
    teardown_failure = report_entries(ok, 1, "passed")
    teardown_failure[-1]["outcome"] = "failed"
    skipped = report_entries(Case("d", "f", 1), 0, "passed")[:1]
    skipped[0]["outcome"] = "skipped"
    entries = (
        report_entries(ok, 0, "passed")
        + [rerun]
        + report_entries(bad, 0, "failed")
        + report_entries(broken, 0, "error")
        + teardown_failure
        + skipped
        + [{"$report_type": "SessionFinish", "exitstatus": 1}]
    )
    path = tmp_path / "shard-1-of-1.jsonl.gz"
    _write(path, entries)

    log = parse_shard_log(path)

    assert log.finished
    outcomes = {(e.case_key, e.repeat): e.outcome for e in log.executions}
    assert outcomes == {
        ("a", 0): "passed",
        ("b", 0): "failed",
        ("c", 0): "error",
        ("a", 1): "error",
    }


def test_log_without_identity_properties_is_rejected(tmp_path):
    entry = report_entries(Case("a", "f", 1), 0, "passed")[1]
    entry["user_properties"] = []
    path = tmp_path / "shard-1-of-1.jsonl.gz"
    _write(path, [entry])
    with pytest.raises(ContractError, match="qt_case_key"):
        parse_shard_log(path)
