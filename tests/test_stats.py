import math

import pytest

from qt_det.stats import (
    Pair,
    benjamini_hochberg,
    fisher_drop,
    jeffreys,
    scope_test,
    wilson_interval,
    z_threshold,
)


def test_jeffreys_never_reaches_bounds():
    assert 0 < jeffreys(0, 20) < 0.05
    assert 0.95 < jeffreys(20, 20) < 1


def test_one_run_against_a_perfect_baseline_is_finite():
    t = scope_test([Pair(0, 1, 20, 20)])
    assert math.isfinite(t.z) and t.z < 0


def test_no_change_gives_zero_z():
    pairs = [Pair(9, 10, 18, 20)] * 50
    assert scope_test(pairs).z == pytest.approx(0)


def test_broad_drop_is_significant():
    pairs = [Pair(6, 10, 18, 20)] * 50
    t = scope_test(pairs)
    assert t.z < z_threshold(0.001)
    assert t.delta == pytest.approx(0.6 - 0.9)


def test_fisher_matches_a_known_value():
    # 1 of 10 passes vs 10 of 10: P = C(11,1)C(9,9)/C(20,10)... computed directly
    expected = math.comb(11, 1) * math.comb(9, 9) / math.comb(20, 10)
    assert fisher_drop(1, 9, 10, 0) == pytest.approx(expected)
    assert fisher_drop(10, 0, 10, 0) == 1.0


def test_fisher_one_failure_vs_twenty_passes():
    assert fisher_drop(0, 1, 20, 0) == pytest.approx(1 / 21)


def test_benjamini_hochberg():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_wilson_interval_contains_rate():
    low, high = wilson_interval(18, 20)
    assert low < 0.9 < high


def test_a_short_perfect_baseline_does_not_make_normal_noise_look_extreme():
    # A case that really passes ~80 %: a 10-run baseline came out 10/10 and the
    # nightly 6/10. With a baseline-only variance this was z ≈ -4 (a false
    # alarm); the pooled variance and the continuity correction keep it modest.
    assert scope_test([Pair(6, 10, 10, 10)]).z > -2.5


def test_continuity_correction_keeps_no_change_at_zero():
    assert scope_test([Pair(5, 10, 10, 20)]).z == 0
    assert scope_test([Pair(5, 10, 11, 20)]).z == 0  # within half a pass of expected
