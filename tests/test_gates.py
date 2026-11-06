import sys
import textwrap

import pytest
from synthetic import make_cases
from test_analyse import Scenario, with_family

from qt_analyzer.analyse import Config
from qt_analyzer.decisions import DecisionLog
from qt_analyzer.gate_compare import (
    EXAMPLES,
    benchmark,
    build_gates,
    judge_all,
)
from qt_analyzer.gates import (
    Observation,
    Verdict,
    available_gates,
    make_gate,
    poisson_binomial,
    register_gate,
)
from qt_analyzer.gates import testimate_threshold as threshold_gate

TESTIMATE = ("testimate_threshold", "poisson_binomial")
OURS = ("zscore", "bayes")


def test_all_four_gates_are_registered():
    assert set(available_gates()) >= {*TESTIMATE, *OURS}


def test_regression_example_masked_by_gains_elsewhere():
    _, make = EXAMPLES["regression"]
    verdicts = judge_all(make(), build_gates([*TESTIMATE, *OURS], {}))
    for name in TESTIMATE:
        assert verdicts[name].regressed == [], name
    for name in OURS:
        assert verdicts[name].regressed == ["topic0"], name


def test_improvement_example_only_two_sided_gates_see():
    _, make = EXAMPLES["improvement"]
    verdicts = judge_all(make(), build_gates([*TESTIMATE, *OURS], {}))
    for name in TESTIMATE:
        assert verdicts[name].improved == [] and verdicts[name].regressed == [], name
    for name in OURS:
        assert "topic0" in verdicts[name].improved, name


def test_poisson_binomial_port_on_a_hand_computed_case():
    # two runs at p = 0.5: P(0) = 0.25, P(<=1) = 0.75 -> at alpha 0.3 at least 1 pass needed
    r = poisson_binomial([Observation(0.5, 0, 1), Observation(0.5, 0, 1)], 0.3)
    assert r.minimum_accepted_passes == 1 and not r.passed
    assert r.score == pytest.approx(0.25)
    assert poisson_binomial([Observation(0.5, 1, 1), Observation(0.5, 0, 1)], 0.3).passed


def test_testimate_threshold_port():
    reference = [0.9] * 50
    assert threshold_gate(reference, reference).passed
    assert not threshold_gate(reference, [0.0] * 50).passed
    r = threshold_gate(reference, reference)
    assert r.threshold == pytest.approx(r.y_reference + 10 * (0.729 * 0.3 / (0.729 * 50**0.5)))


def test_benchmark_keeps_false_alarms_low_for_the_alternatives():
    table = benchmark(build_gates(list(OURS), {}), trials=40, seed=3)
    assert table["unchanged"]["zscore"] <= 0.05
    assert table["unchanged"]["bayes"] <= 0.05
    assert table["one topic -30 pts"]["zscore"] >= 0.5


def test_a_gate_from_your_own_module_joins_everything(tmp_path, monkeypatch):
    (tmp_path / "my_gates.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass
            from qt_analyzer.gates import Verdict, register_gate

            @register_gate("always_alarm")
            @dataclass(frozen=True)
            class AlwaysAlarm:
                scope: str = "overall"

                def judge(self, comparison):
                    return Verdict(regressed=[self.scope])
            """
        )
    )
    (tmp_path / "analyzer.toml").write_text(
        '[analysis]\ngate = "always_alarm"\ngate_modules = ["my_gates"]\n'
        '[gates.always_alarm]\nscope = "overall"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        config = Config.load(tmp_path / "analyzer.toml")
        assert "always_alarm" in available_gates()
        assert make_gate("always_alarm", {"scope": "x"}).judge(None).regressed == ["x"]

        s = Scenario(tmp_path / "run", make_cases(2, 10, 0.9))
        s.commit("graph", "g1")
        s.commit("graph", "g2")
        analysis = s.analyse(config)
        assert analysis.timeline[-1].nightly.status == "confirmed"
        decided = DecisionLog.load(s.results).decided()
        assert {r["gate"] for r in decided.values()} == {"always_alarm"}
    finally:
        sys.modules.pop("my_gates", None)


def test_registering_a_different_gate_under_an_existing_name_fails():
    with pytest.raises(ValueError, match="already registered"):

        @register_gate("zscore")
        class Other:
            def judge(self, comparison):
                return Verdict()


def test_the_configured_gate_decides_pr_results(tmp_path):
    cases = make_cases(4, 25, 0.9)
    s = Scenario(tmp_path, cases)
    s.commit("graph", "g1")
    s.commit("data", "d1")
    s.pr("graph", 5, "g1", "h1", with_family(cases, "family2", 0.1), "2026-11-03T01:00:00Z")
    (pr,) = s.analyse(Config(gate="bayes")).prs
    assert pr.status == "regressed" and "family2" in pr.flagged


def test_zscore_split_catches_broad_local_and_flaky_drops():
    table = benchmark(build_gates(["zscore_split"], {}), trials=60, seed=5)
    row = {name: t["zscore_split"] for name, t in table.items()}
    assert row["unchanged"] <= 0.1
    for scenario in ("all topics -10 pts", "one topic -30 pts", "flaky cases -30 pts"):
        assert row[scenario] >= 0.4, scenario


def test_zscore_split_has_a_stricter_budget_for_improvements():
    gate = make_gate("zscore_split", {})
    (scope, _, share), *_ = gate._scopes(EXAMPLES["improvement"][1]())
    from qt_analyzer.stats import z_threshold

    assert scope == "overall"
    assert -z_threshold(gate.improvement_alpha * share) > -z_threshold(gate.alpha * share)


def test_equal_false_alarm_tuning_reaches_the_target_without_exceeding_it():
    from qt_analyzer.gate_compare import equalise_false_alarms

    gates, notes = equalise_false_alarms(build_gates(["poisson_binomial", "zscore_split"], {}),
                                         0.1, trials=80, seed=7)  # fmt: skip
    table = benchmark(gates, trials=80, seed=8)
    for name in gates:
        assert table["unchanged"][name] <= 0.2, name
        assert notes[name].startswith("level")
