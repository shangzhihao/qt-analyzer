import pytest
from synthetic import make_cases
from test_analyse import Scenario, with_family

from qt_analyzer.analyse import Config
from qt_analyzer.gate_compare import EXAMPLES, build_gates, locator_benchmark
from qt_analyzer.locators import (
    available_locators,
    format_location,
    location_from_json,
    make_locator,
    register_locator,
)
from qt_analyzer.report import pr_comment


def test_ranked_locator_is_registered():
    assert "ranked" in available_locators()


def test_ranked_finds_the_topic_and_its_cases():
    comparison = EXAMPLES["regression"][1]()  # topic0 80 % -> 40 %, topics 1-4 up
    location = make_locator("ranked").locate(comparison, "regression", ["topic0"])
    assert location.scopes[0].scope == "topic0"
    assert len(location.cases) == 10
    assert {c.family for c in location.cases} == {"topic0"}


def test_improvements_are_located_in_the_right_direction():
    comparison = EXAMPLES["improvement"][1]()
    location = make_locator("ranked").locate(comparison, "improvement", ["topic0"])
    assert location.scopes[0].scope == "topic0" and location.scopes[0].z > 0


def test_location_round_trips_through_json():
    from dataclasses import asdict

    location = make_locator("ranked").locate(EXAMPLES["regression"][1](), "regression", [])
    assert location_from_json(asdict(location)) == location
    assert format_location(location)[0].startswith("where: ")


def test_alerts_and_pr_comments_say_where(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    for i in range(1, 4):
        s.commit("graph", f"g{i}")
    s.commit("data", "d1", with_family(cases, "family1", 0.3))
    s.pr("graph", 5, "g3", "h1", with_family(cases, "family2", 0.1), "2026-11-03T01:00:00Z")
    analysis = s.analyse(Config())

    (alert,) = analysis.alerts
    assert alert.where["scopes"][0]["scope"] == "family1"
    (pr,) = analysis.prs
    assert pr.location.scopes[0].scope == "family2"
    assert "**Where:** family2" in pr_comment(pr)


def test_locator_benchmark_finds_the_topic():
    gate = build_gates(["zscore_strata"], {})["zscore_strata"]
    table = locator_benchmark(gate, make_locator("ranked"), trials=30, runs=10, baseline_runs=10)
    row = table["one topic -30 pts"]
    assert row["fired"] >= 0.8 and row["scope@1"] >= 0.9 and row["precision@10"] >= 0.8


def test_a_locator_from_your_own_code():
    with pytest.raises(ValueError, match="already registered"):

        @register_locator("ranked")
        class Other:
            def locate(self, comparison, direction, flagged):
                return None
