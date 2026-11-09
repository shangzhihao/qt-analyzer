import random

from synthetic import make_cases, manifest, write_run

from qt_det.cli import main
from qt_det.ingest import ingest_pending
from qt_det.simulate import population_from_results, sweep, synthetic_population
from qt_det.stores import LocalDirStore, ResultStore


def test_large_family_drop_is_caught_and_no_change_stays_quiet():
    no_change, large = sweep(
        synthetic_population(), [0.0, 0.4], [1], [10], trials=10, states=6, seed=3
    )
    assert no_change.false_alerts_per_state <= 0.05
    assert large.push_detected >= 0.8
    assert large.nightly_confirmed == 1.0
    assert large.false_alerts_per_state == 0.0


def test_population_from_an_ingested_run(tmp_path):
    components = {"graph": ("g1", "2026-11-01T10:00:00Z")}
    cases = make_cases(families=2, per_family=5, p=1.0)
    write_run(
        tmp_path / "logs",
        manifest("r1", "nightly", components, "graph", 4, 2),
        cases,
        random.Random(0),
    )
    results = ResultStore(tmp_path / "results")
    ingest_pending(LocalDirStore(tmp_path / "logs"), results)

    population = population_from_results(results)

    assert len(population) == 10
    assert {c.p for c in population} == {1.0}
    assert {c.family for c in population} == {"family0", "family1"}


def test_simulate_command_prints_a_table(capsys):
    main(
        [
            "simulate",
            "--shifts",
            "0.3",
            "--nightly-runs",
            "10",
            "--trials",
            "3",
            "--states",
            "4",
            "--config",
            "missing.toml",
        ]
    )
    out = capsys.readouterr().out
    assert "push detect" in out and "0.30" in out
