"""End-to-end: synthetic runs -> ingest -> analyse, checking the decisions."""

import dataclasses
import random

from synthetic import REPOS, Case, make_cases, manifest, write_run

from qt_det.analyse import Config, analyse, save
from qt_det.contract import state_key
from qt_det.ingest import ingest_pending
from qt_det.stores import LocalDirStore, ResultStore


class Scenario:
    """A main timeline built from commits; each state gets a 1-run and a 10-run nightly run."""

    def __init__(self, tmp_path, cases):
        self.logs = tmp_path / "logs"
        self.results = ResultStore(tmp_path / "results")
        self.rng = random.Random(1234)
        self.cases = cases
        self.components = {
            "graph": ("g0", "2026-11-01T00:00:00Z"),
            "data": ("d0", "2026-11-01T00:00:00Z"),
        }
        self.keys: list[str] = []
        self.counter = 0

    def commit(self, component, sha, cases=None, main=True, nightly=True):
        self.counter += 1
        at = f"2026-11-02T{self.counter:02d}:00:00Z"
        self.components = self.components | {component: (sha, at)}
        cases = cases or self.cases
        if main:
            data = manifest(f"main-{self.counter}", "main", self.components, component, 1, 2)
            write_run(self.logs, data, cases, self.rng)
        if nightly:
            data = manifest(f"nightly-{self.counter}", "nightly", self.components, component, 10, 2)
            write_run(self.logs, data, cases, self.rng)
        self.keys.append(state_key({n: s for n, (s, _) in self.components.items()}))
        return self.keys[-1]

    def pr(self, component, number, base_sha, head_sha, cases, created_at):
        components = self.components | {component: (f"merge-{head_sha}", "2026-11-03T00:00:00Z")}
        pr = {
            "component": component,
            "repo": REPOS[component],
            "number": number,
            "head_sha": head_sha,
            "merge_sha": f"merge-{head_sha}",
            "baseline": {n: s for n, (s, _) in self.components.items()} | {component: base_sha},
        }
        run_id = f"pr-{number}-{head_sha}"
        data = manifest(run_id, "pr", components, None, 1, 2, pr=pr, created_at=created_at)
        write_run(self.logs, data, cases, self.rng)

    def analyse(self, config=None):
        ingest_pending(LocalDirStore(self.logs), self.results)
        analysis = analyse(self.results, config or Config())
        save(self.results, analysis)
        return analysis


def with_family(cases, family, p):
    return [dataclasses.replace(c, p=p) if c.family == family else c for c in cases]


def test_regression_in_data_is_suspected_confirmed_and_attributed(tmp_path):
    cases = make_cases(families=4, per_family=25, p=0.9)
    s = Scenario(tmp_path, cases)
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    bad = s.commit("data", "d1", with_family(cases, "family1", 0.4))
    s.commit("graph", "g4", with_family(cases, "family1", 0.4))

    analysis = s.analyse()

    by_key = {e.data.state.key: e for e in analysis.timeline}
    assert by_key[bad].push.status == "suspected"
    assert "family1" in by_key[bad].push.flagged
    assert by_key[bad].nightly.status == "confirmed"
    assert [a.state for a in analysis.alerts] == [bad]
    alert = analysis.alerts[0]
    assert alert.open and alert.suspect_range == [bad]
    assert by_key[bad].data.state.created_by == "data"
    # The next state is still below the reference: same alert, no new one, and
    # it is not blamed, because baselines do not pool across the regression.
    assert alert.still_regressed_at == [s.keys[-1]]
    after = by_key[s.keys[-1]]
    assert after.nightly.status == "ongoing"
    assert after.push.status == "ok"
    assert list((tmp_path / "results/alerts").glob("*.json"))


def test_pr_on_top_of_a_regression_is_not_blamed_for_it(tmp_path):
    cases = make_cases(families=4, per_family=25, p=0.9)
    regressed = with_family(cases, "family1", 0.4)
    s = Scenario(tmp_path, cases)
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    s.commit("data", "d1", regressed)
    s.pr("graph", 3, "g3", "abc", regressed, "2026-11-03T01:00:00Z")

    (pr,) = s.analyse().prs

    assert pr.status == "ok"


def test_unchanged_quality_raises_nothing(tmp_path):
    s = Scenario(tmp_path, make_cases(families=4, per_family=25, p=0.85))
    for i in range(1, 9):
        s.commit("graph" if i % 2 else "data", f"x{i}")

    analysis = s.analyse()

    assert analysis.alerts == []
    assert all(e.push.status in ("ok", "no-baseline") for e in analysis.timeline)
    assert analysis.reference == s.keys[-1]


def test_recovery_closes_the_alert(tmp_path):
    cases = make_cases(families=4, per_family=25, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    bad = s.commit("graph", "g3", with_family(cases, "family2", 0.3))
    fixed = s.commit("graph", "g4")

    analysis = s.analyse()

    assert len(analysis.alerts) == 1
    assert analysis.alerts[0].state == bad
    assert not analysis.alerts[0].open and analysis.alerts[0].closed_by == fixed
    assert analysis.reference == fixed


def test_slow_drift_is_caught_by_the_reference(tmp_path):
    cases = make_cases(families=2, per_family=50, p=0.95)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g0x")
    for i, p in enumerate([0.92, 0.89, 0.86, 0.83, 0.80, 0.77, 0.74], start=1):
        s.commit("graph", f"g{i}", [dataclasses.replace(c, p=p) for c in cases])

    analysis = s.analyse()

    assert analysis.alerts, "a drift of 21 points must be flagged at some state"
    assert analysis.alerts[0].reference == s.keys[0]


def test_edited_case_is_coverage_not_regression(tmp_path):
    cases = make_cases(families=2, per_family=20, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    edited = [Case(c.key, c.family, 0.0, "d2") if c.family == "family0" else c for c in cases]
    key = s.commit("graph", "g3", edited)

    analysis = s.analyse()

    entry = next(e for e in analysis.timeline if e.data.state.key == key)
    assert entry.push.status == "ok"
    assert entry.nightly.status == "ok"
    assert len(entry.push.comparison.coverage.edited) == 20


def test_unmeasured_predecessor_widens_the_suspect_range(tmp_path):
    cases = make_cases(families=2, per_family=25, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("graph", "g2")
    # g3 was never measured (lost trigger) and has not been run by a nightly yet
    s.commit("graph", "g3", main=False, nightly=False)
    s.keys.pop()  # unmeasured states are unknown to the analyzer
    bad = s.commit("graph", "g4", with_family(cases, "family0", 0.3), nightly=False)

    analysis = s.analyse()

    entry = next(e for e in analysis.timeline if e.data.state.key == bad)
    assert entry.push.status == "suspected"


def test_pr_against_its_base_state(tmp_path):
    cases = make_cases(families=4, per_family=25, p=0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("data", "d1")
    s.pr("data", 7, "d1", "h1", with_family(cases, "family3", 0.2), "2026-11-03T01:00:00Z")
    s.pr("data", 7, "d1", "h2", cases, "2026-11-03T02:00:00Z")  # fixed in a later push
    s.pr("graph", 8, "g1", "h3", with_family(cases, "family0", 0.1), "2026-11-03T01:00:00Z")
    s.pr("graph", 9, "g-unknown", "h4", cases, "2026-11-03T01:00:00Z")

    prs = {p.number: p for p in s.analyse().prs}

    assert prs[7].head_sha == "h2" and prs[7].status == "ok"
    assert prs[8].status == "regressed" and "family0" in prs[8].flagged
    # One run against a 22-run baseline at 90 %: only cases whose baseline never
    # failed can be called suspected, so a few family0 cases, not most of them.
    suspected_families = [
        c.family for c in prs[8].comparison.cases if c.case_key in prs[8].suspected_cases
    ]
    assert suspected_families.count("family0") >= 1
    assert len(suspected_families) - suspected_families.count("family0") <= 2
    # baseline never measured: compared with the nearest measured main state
    assert prs[9].status == "ok" and prs[9].nearest_baseline
    assert prs[9].baseline == s.keys[-1]


def test_alerts_close_one_by_one_and_the_nightly_sees_improvements(tmp_path):
    cases = make_cases(10, 20, 0.97)
    s = Scenario(tmp_path, cases)
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    first = with_family(cases, "family0", 0.5)
    drop0 = s.commit("data", "d1", first)
    drop1 = s.commit("data", "d2", with_family(first, "family1", 0.8))
    back0 = s.commit("data", "d3", with_family(cases, "family1", 0.8))
    analysis = s.analyse()
    alerts = {a.state: a for a in analysis.alerts}
    assert "family0" in alerts[drop0].scopes
    assert "family1" in alerts[drop1].scopes and "family0" not in alerts[drop1].scopes
    entry = next(e for e in analysis.timeline if e.data.state.key == back0)
    assert "family0" in entry.nightly.step_improved
    assert not alerts[drop0].open and alerts[drop0].closed_by == back0
    assert alerts[drop1].open and back0 in alerts[drop1].still_regressed_at
