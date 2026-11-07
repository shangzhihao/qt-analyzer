"""When no case has the same definition on both sides, nothing can be checked."""

import dataclasses

from synthetic import make_cases
from test_analyse import Scenario, with_family

from qt_analyzer.decisions import DecisionLog
from qt_analyzer.report import pr_comment


def edited(cases, definition="d2"):
    return [dataclasses.replace(c, definition=definition) for c in cases]


def test_an_edited_suite_is_not_comparable_and_restarts_the_reference(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    changed = s.commit("graph", "g3", edited(cases))
    after = s.commit("graph", "g4", edited(cases))

    analysis = s.analyse()
    by_key = {e.data.state.key: e for e in analysis.timeline}

    assert by_key[changed].push.status == "not-comparable"
    assert by_key[changed].nightly.status == "not-comparable"
    assert "100 edited" in by_key[changed].nightly.reason
    assert analysis.alerts == []
    # the next state is compared with the restarted reference, normally
    assert by_key[after].nightly.status == "ok"
    assert by_key[after].nightly.vs_reference.overall.cases == 100
    moves = [r for r in DecisionLog.load(s.results).records if r["kind"] == "reference.moved"]
    restart = next(r for r in moves if r["state"] == changed)
    assert restart["reason"] == "definitions changed"


def test_alerts_about_old_definitions_are_closed(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    bad = s.commit("graph", "g3", with_family(cases, "family1", 0.3))
    changed = s.commit("graph", "g4", edited(cases))

    (alert,) = s.analyse().alerts

    assert alert.state == bad and not alert.open and alert.closed_by == changed


def test_a_pr_that_edits_every_test_is_not_comparable(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("data", "d1")
    s.pr("graph", 9, "g1", "h1", edited(cases), "2026-11-03T01:00:00Z")

    (pr,) = s.analyse().prs
    body = pr_comment(pr)

    assert pr.status == "not-comparable" and "100 edited" in pr.reason
    assert "Not comparable" in body
    assert "No significant drop" not in body and "nan" not in body
    reported = DecisionLog.load(s.results).pr_reports()["pr-9-h1"]
    assert reported["status"] == "not-comparable"


def test_partly_edited_states_are_still_compared(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    half = [dataclasses.replace(c, definition="d2") if c.family == "family0" else c for c in cases]
    key = s.commit("graph", "g3", half)

    entry = next(e for e in s.analyse().timeline if e.data.state.key == key)

    assert entry.push.status == "ok" and entry.nightly.status == "ok"
