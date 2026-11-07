import gzip
import json
import subprocess

from synthetic import Case, report_entries
from test_analyse import Scenario

from qt_analyzer import case_check
from qt_analyzer.analyse import Agg
from qt_analyzer.cli import main
from qt_analyzer.fetch import fetch_results

STEADY = Case("steady", "topicA", 0.9)
COIN = Case("coin", "topicB", 0.5)


def _history(tmp_path, states=3):
    s = Scenario(tmp_path / "history", [STEADY, COIN])
    for i in range(states):
        s.commit("graph", f"g{i}")
    s.analyse()
    return s.results


def _local_log(path, outcomes):
    """outcomes: list of (case, definition, passes, runs)."""
    entries = []
    for case, definition, passes, runs in outcomes:
        c = Case(case.key, case.family, case.p, definition)
        for repeat in range(runs):
            entries += report_entries(c, repeat, "passed" if repeat < passes else "failed")
    entries.append({"$report_type": "SessionFinish", "exitstatus": 1})
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    return path


def _verdicts(tmp_path, outcomes, **kwargs):
    results = _history(tmp_path)
    log = _local_log(tmp_path / "local.jsonl.gz", outcomes)
    main_counts, states = case_check.main_counts(results, 5)
    return {v.case_key: v for v in case_check.check(case_check.local_counts([log]), main_counts,
                                                     **kwargs)}  # fmt: skip


def test_regression_of_one_case(tmp_path):
    v = _verdicts(tmp_path, [(STEADY, "d1", 12, 30)])["steady"]
    assert v.verdict == "regression" and v.p_drop < 0.001
    assert v.main[1] == 33  # 3 states x (1 main + 10 nightly) runs


def test_improvement_of_one_case(tmp_path):
    assert _verdicts(tmp_path, [(COIN, "d1", 30, 30)])["coin"].verdict == "improvement"


def test_no_evidence_says_how_many_runs_would_do(tmp_path):
    v = _verdicts(tmp_path, [(STEADY, "d1", 3, 3)])["steady"]
    assert v.verdict == "no evidence"
    assert v.runs_needed is not None and v.runs_needed != 0


def test_definition_changed_and_new_tests_are_not_judged(tmp_path):
    verdicts = _verdicts(tmp_path, [(STEADY, "d2", 0, 30), (Case("brand-new", "t", 1), "d1", 0, 5)])
    assert verdicts["steady"].verdict == "definition changed"
    assert verdicts["brand-new"].verdict == "new"


def test_several_cases_are_corrected_for_multiple_testing():
    # One borderline drop (p about 0.03) among many cases is not a regression after Holm.
    local, main = {}, {}
    for i in range(20):
        local[(f"c{i}", "d")] = Agg("f", f"c{i}", 10, 10, 0, 0)
        main[(f"c{i}", "d")] = Agg("f", f"c{i}", 30, 30, 0, 0)
    local[("c0", "d")] = Agg("f", "c0", 10, 7, 3, 0)
    alone = case_check.check({("c0", "d"): local[("c0", "d")]}, main)
    together = case_check.check(local, main)
    assert alone[0].verdict == "regression"
    assert next(v for v in together if v.case_key == "c0").verdict == "no evidence"


def test_runs_needed_grows_for_smaller_changes_and_short_histories():
    assert case_check.runs_needed(0.9, 200, 0.2, 0.05) < case_check.runs_needed(0.9, 200, 0.1, 0.05)
    assert case_check.runs_needed(0.9, 5, 0.05, 0.05) == -1


def test_command_exit_code_and_output(tmp_path, capsys):
    results = _history(tmp_path)
    bad = _local_log(tmp_path / "bad.jsonl.gz", [(STEADY, "d1", 10, 30)])
    good = _local_log(tmp_path / "good.jsonl.gz", [(STEADY, "d1", 27, 30)])
    assert main(["case", str(bad), "--results", str(results.root)]) == 1
    assert "REGRESSION" in capsys.readouterr().out
    assert main(["case", str(good), "--results", str(results.root)]) == 0
    assert "no evidence of change" in capsys.readouterr().out


def test_fetch_clones_then_updates(tmp_path):
    origin = tmp_path / "origin"

    def git(*args, cwd=origin):
        subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)

    origin.mkdir()
    git("init", "-q", "-b", "qt-results")
    (origin / "index").mkdir()
    (origin / "index" / "runs.jsonl").write_text("one\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "1")
    dest = fetch_results(tmp_path / "cache", url=str(origin))
    assert (dest / "index" / "runs.jsonl").read_text() == "one\n"

    (origin / "index" / "runs.jsonl").write_text("two\n")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "2")
    fetch_results(tmp_path / "cache", url=str(origin))
    assert (dest / "index" / "runs.jsonl").read_text() == "two\n"
