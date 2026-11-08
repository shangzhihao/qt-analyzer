import json
import re

from synthetic import Case, make_cases
from test_analyse import Scenario, with_family

from qt_analyzer.analyse import Config
from qt_analyzer.cli import main
from qt_analyzer.html_report.html import render
from qt_analyzer.html_report.model import build_model, failure_samples


def _history(tmp_path):
    cases = make_cases(4, 20, 0.9)
    s = Scenario(tmp_path, cases)
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    s.commit("data", "d1", with_family(cases, "family1", 0.3))
    s.pr("graph", 7, "g3", "abc", with_family(cases, "family2", 0.1), "2026-11-03T01:00:00Z")
    s.analyse()
    return s


def test_model_has_every_section(tmp_path):
    s = _history(tmp_path)
    model = build_model(s.results, Config())
    assert set(model) >= {"meta", "overview", "states", "heat", "alerts", "prs", "gates", "cases",
                          "runs", "decisions"}  # fmt: skip
    assert len(model["states"]) == 4 and model["overview"]["open_alerts"] == 1
    assert model["alerts"][0]["where"]["scopes"][0]["scope"] == "family1"
    assert model["heat"]["topics"] == ["family0", "family1", "family2", "family3"]
    assert "regressed" in model["heat"]["flag"][-1]
    assert {"testimate_threshold", "zscore_strata"} <= set(model["gates"]["names"])
    assert model["prs"][0]["status"] == "regressed"
    assert len(model["cases"]) == 80
    json.dumps(model, allow_nan=False)  # strict JSON: no NaN


def test_html_is_self_contained_and_embeds_the_model(tmp_path):
    s = _history(tmp_path)
    html = render(build_model(s.results, Config()))
    assert "<script src" not in html and "<link" not in html  # works offline
    data = re.search(r'<script id="model" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(data.group(1))["overview"]["states"] == 4


def test_text_cannot_break_out_of_the_data_block():
    model = {"meta": {"note": "</script><script>alert(1)</script>"}}
    html = render(model)
    block = re.search(r'<script id="model" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(block.group(1)) == model


def test_failure_samples_from_raw_logs(tmp_path):
    log = tmp_path / "shard-1-of-1.jsonl"
    entry = {"$report_type": "TestReport", "when": "call", "outcome": "failed",
             "user_properties": [["qt_case_key", "k1"]],
             "longrepr": {"reprcrash": {"message": "AssertionError: wrong answer"}}}  # fmt: skip
    log.write_text("\n".join(json.dumps(entry) for _ in range(5)) + "\n")
    assert failure_samples([log]) == {"k1": ["AssertionError: wrong answer"]}


def test_report_command_writes_the_file(tmp_path, capsys):
    s = _history(tmp_path)
    out = tmp_path / "report.html"
    assert main(["report", "--results", str(s.results.root), "--logs", str(s.logs),
                 "--out", str(out), "--config", "missing.toml"]) == 0  # fmt: skip
    assert out.exists() and "Answer quality report" in out.read_text()
    assert "4 states, 1 alerts, 1 PRs" in capsys.readouterr().out


def test_definition_changes_are_marked_per_case(tmp_path):
    cases = [Case("a", "t", 0.9), Case("b", "t", 0.9)]
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2", [Case("a", "t", 0.9, "d2"), Case("b", "t", 0.9)])
    s.analyse()
    model = build_model(s.results, Config())
    by_key = {c["key"]: c for c in model["cases"]}
    assert (
        by_key["a"]["definition_changed_at"] == [1] and by_key["b"]["definition_changed_at"] == []
    )
