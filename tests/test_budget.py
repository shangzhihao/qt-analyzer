import numpy as np
import pytest

from qt_analyzer import budget
from qt_analyzer.stats import Pair, scope_test


def test_vectorised_scope_z_matches_scope_test():
    rng = np.random.default_rng(0)
    population = budget.population_from(list(rng.uniform(0.3, 1.0, 40)), ["a", "b"] * 20)
    passed_s = rng.integers(0, 11, 40)
    passed_b = rng.integers(0, 21, 40)

    overall, families = budget.scope_z(passed_s, 10, passed_b, 20, population)

    pairs = [Pair(int(s), 10, int(b), 20) for s, b in zip(passed_s, passed_b, strict=True)]
    assert overall == pytest.approx(scope_test(pairs).z)
    for index, name in enumerate(["a", "b"]):
        members = [pair for pair, fam in zip(pairs, ["a", "b"] * 20, strict=True) if fam == name]
        assert families[index] == pytest.approx(scope_test(members).z)


def test_dispersion_inflates_z():
    population = budget.population_from([0.5] * 10, ["a"] * 10)
    passes = np.full(10, 2)
    baseline = np.full(10, 10)
    plain, _ = budget.scope_z(passes, 10, baseline, 20, population)
    inflated, _ = budget.scope_z(passes, 10, baseline, 20, population, phi_overall=4.0)
    assert inflated == pytest.approx(plain / 2)


def test_dispersion_estimate_is_near_one_without_night_effects_and_large_with_them():
    population = budget.realistic_population(cases=600, families=30, seed=2)
    rng = np.random.default_rng(3)
    quiet = np.stack([budget.night(population, 10, 0.0, rng) for _ in range(40)])
    noisy = np.stack([budget.night(population, 10, 0.4, rng) for _ in range(40)])

    quiet_phi = budget.estimate_dispersion(quiet, 10, population)
    noisy_phi = budget.estimate_dispersion(noisy, 10, population)

    assert quiet_phi[0] < 1.6 and quiet_phi[1] < 1.3
    assert noisy_phi[0] > 5 and noisy_phi[1] > quiet_phi[1]


def test_deterministic_cases_stay_deterministic_under_night_effects():
    population = budget.population_from([0.0, 1.0, 0.5], ["a", "a", "a"])
    rng = np.random.default_rng(4)
    for _ in range(20):
        passes = budget.night(population, 10, 1.0, rng)
        assert passes[0] == 0 and passes[1] == 10


def test_threshold_tightens_with_more_scopes_and_smaller_budget():
    assert budget.threshold(0.01, 10) < budget.threshold(0.01, 1) < 0
    assert budget.threshold(0.001, 10) < budget.threshold(0.01, 10)


def test_false_issues_follow_the_budget():
    population = budget.realistic_population(cases=400, families=20, seed=5)
    loose = budget.null_months(population, budget.Scenario(20, budget=4.0), 150, seed=6)
    tight = budget.null_months(population, budget.Scenario(20, budget=0.25), 150, seed=6)
    assert tight.per_month < loose.per_month
    assert loose.per_month <= 4.0 * 1.5


def test_a_large_drop_is_detected():
    population = budget.realistic_population(cases=400, families=20, seed=7)
    on_time, within_one = budget.detection(population, budget.Scenario(20), 0.3, "overall", 20, 8)
    assert on_time >= 0.9 and within_one >= on_time


def test_tiny_families_do_not_break_the_budget():
    # 150 families of 1-3 uncertain cases: with a baseline-only variance, a short
    # baseline that happens to look perfect made these raise many false issues.
    rng = np.random.default_rng(9)
    sizes = rng.integers(1, 4, 150)
    p = list(rng.uniform(0.6, 0.95, sizes.sum()))
    family = [f"f{i}" for i, n in enumerate(sizes) for _ in range(n)]
    population = budget.population_from(p, family)
    result = budget.null_months(population, budget.Scenario(30, budget=1.0), 150, seed=10)
    assert result.per_month <= 1.0 * 1.6
