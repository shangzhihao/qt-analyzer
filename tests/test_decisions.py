"""The decision log: verdicts are taken once and never rewritten."""

import dataclasses

from synthetic import make_cases, manifest, write_run
from test_analyse import Scenario, with_family

from qt_det.analyse import Config
from qt_det.decisions import DecisionLog


def _verdicts(analysis):
    return {e.data.state.key: e.nightly.status for e in analysis.timeline if e.nightly}


def test_a_second_pass_appends_nothing(tmp_path):
    s = Scenario(tmp_path, make_cases(4, 25, 0.9))
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    first = s.analyse()
    count = len(DecisionLog.load(s.results).records)
    second = s.analyse()
    assert len(DecisionLog.load(s.results).records) == count
    assert _verdicts(second) == _verdicts(first)
    assert not second.log.new


def test_a_looser_threshold_does_not_re_decide_old_states(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    for i in range(1, 5):
        s.commit("graph", f"g{i}")
    before = s.analyse()
    assert before.alerts == []

    loose = dataclasses.replace(Config(), nightly_z=5.0)  # everything would be a "drop"
    after = s.analyse(loose)
    assert _verdicts(after) == _verdicts(before)
    assert after.alerts == []

    s.commit("graph", "g5")  # only the new state is decided with the new threshold
    newest = s.analyse(loose)
    assert newest.timeline[-1].nightly.status == "confirmed"
    assert {k: v for k, v in _verdicts(newest).items() if k != s.keys[-1]} == _verdicts(before)


def test_late_evidence_for_an_older_state_does_not_rewrite_later_verdicts(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2", nightly=False)  # its nightly arrives late
    s.commit("graph", "g3")
    s.commit("graph", "g4")
    before = s.analyse()
    late_key = s.keys[1]

    # the late nightly for g2 shows a drop
    components = {"graph": ("g2", "2026-11-02T02:00:00Z"), "data": ("d0", "2026-11-01T00:00:00Z")}
    write_run(s.logs, manifest("nightly-late", "nightly", components, "graph", 10, 2),
              with_family(cases, "family0", 0.2), s.rng)  # fmt: skip
    after = s.analyse()

    verdicts = _verdicts(after)
    assert verdicts[late_key] == "confirmed"  # decided late, at its own position
    for key, verdict in _verdicts(before).items():
        assert verdicts[key] == verdict  # later states keep their verdicts
    assert [a.state for a in after.alerts] == [late_key]


def test_a_pr_verdict_is_decided_once(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("data", "d1")
    s.pr("graph", 5, "g1", "h1", with_family(cases, "family2", 0.1), "2026-11-03T01:00:00Z")
    (first,) = s.analyse().prs
    assert first.status == "regressed"

    # later main data that would make the baseline look just as bad
    s.commit("data", "d2", with_family(cases, "family2", 0.1))
    (again,) = s.analyse().prs
    assert again.status == first.status and again.flagged == first.flagged
    reports = DecisionLog.load(s.results).pr_reports()
    assert list(reports) == ["pr-5-h1"]


def test_pr_against_an_unmeasured_baseline_is_not_reported_yet(tmp_path):
    s = Scenario(tmp_path, make_cases(2, 5))
    s.commit("graph", "g1")
    s.pr("graph", 6, "g-unmeasured", "h1", make_cases(2, 5), "2026-11-03T01:00:00Z")
    (pr,) = s.analyse().prs
    assert pr.nearest_baseline
    assert DecisionLog.load(s.results).pr_reports() == {}
