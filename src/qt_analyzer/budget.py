"""Measure the nightly decision rule at realistic scale before fixing its budget.

The rule (design revision 2, §3): a nightly-measured state is confirmed when a
scope test rejects at `α_state = false_issues_per_month / states_per_month`,
split by Bonferroni over the step test (against the predecessor's nightly) and
the drift test (against the reference), each over the overall scope and every
family. Variances are inflated by the dispersion `φ` estimated from A/A nights
(the reference state re-run every night), because LLM suites vary together from
night to night, which plain binomial noise ignores.

This module simulates that rule with numpy: a population of per-case pass rates,
a per-night random effect on the logit scale (global and per family), A/A nights
to estimate `φ` exactly as production would, and months of nightly decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from qt_analyzer.stats import CONTINUITY

# Shape of the real vws-graph suite (testimate reference, 1,934 cases):
# p = 1: 62.4 %, [0.95, 1): 15.8 %, (0.05, 0.95): 17.3 %, (0, 0.05]: 1.6 %, p = 0: 2.9 %.
_SHAPE = ((1.0, 1.0, 0.624), (0.95, 0.999, 0.158), (0.05, 0.95, 0.173), (0.001, 0.05, 0.016),
          (0.0, 0.0, 0.029))  # fmt: skip


@dataclass(frozen=True)
class Population:
    p: np.ndarray  # pass probability per case at full quality
    family: np.ndarray  # family index per case
    families: int


def realistic_population(cases: int = 1900, families: int = 100, seed: int = 1) -> Population:
    rng = np.random.default_rng(seed)
    weights = np.array([w for _, _, w in _SHAPE])
    band = rng.choice(len(_SHAPE), size=cases, p=weights / weights.sum())
    low = np.array([lo for lo, _, _ in _SHAPE])[band]
    high = np.array([hi for _, hi, _ in _SHAPE])[band]
    p = np.where(low == high, low, rng.uniform(low, high))
    return Population(p, rng.integers(0, families, size=cases), families)


def population_from(p: list[float], family: list[str]) -> Population:
    names = {name: i for i, name in enumerate(sorted(set(family)))}
    return Population(np.array(p), np.array([names[f] for f in family]), len(names))


# ---------------------------------------------------------------------------
# simulation of one night


def night(
    population: Population,
    runs: int,
    night_sd: float,
    rng: np.random.Generator,
    drop: np.ndarray | None = None,
) -> np.ndarray:
    """Passes per case for one nightly run of `runs` repeats.

    Uncertain cases share a random logit shift for the night: one global and one
    per family, each N(0, night_sd). Cases at exactly 0 or 1 stay deterministic.
    """
    p = population.p if drop is None else np.clip(population.p - drop, 0.0, 1.0)
    if night_sd > 0:
        shift = rng.normal(0, night_sd) + rng.normal(0, night_sd, population.families)
        uncertain = (p > 0) & (p < 1)
        logit = np.log(p[uncertain] / (1 - p[uncertain])) + shift[population.family[uncertain]]
        p = p.copy()
        p[uncertain] = 1 / (1 + np.exp(-logit))
    return rng.binomial(runs, p)


# ---------------------------------------------------------------------------
# statistics (vectorised; same formula as stats.scope_test)


def scope_z(
    passed_s: np.ndarray,
    n_s: np.ndarray | int,
    passed_b: np.ndarray,
    n_b: np.ndarray | int,
    population: Population,
    phi_overall: float = 1.0,
    phi_family: float = 1.0,
) -> tuple[float, np.ndarray]:
    """z for the overall scope and per family, variance inflated by φ."""
    n_s = np.broadcast_to(n_s, passed_s.shape).astype(float)
    n_b = np.broadcast_to(n_b, passed_b.shape).astype(float)
    raw = passed_b / n_b
    pooled = (passed_s + passed_b + 1) / (n_s + n_b + 2)
    num = passed_s - n_s * raw
    var = n_s * pooled * (1 - pooled) * (1 + n_s / n_b)
    z_overall = _continuity(num.sum()) / np.sqrt(var.sum() * phi_overall)
    fam_num = np.bincount(population.family, num, population.families)
    fam_var = np.bincount(population.family, var, population.families)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_family = np.where(fam_var > 0, _continuity(fam_num) / np.sqrt(fam_var * phi_family), 0.0)
    return float(z_overall), z_family


def _continuity(difference):  # same correction as stats.continuity, vectorised
    return np.sign(difference) * np.maximum(np.abs(difference) - CONTINUITY, 0.0)


def estimate_dispersion(aa: np.ndarray, runs: int, population: Population) -> tuple[float, float]:
    """φ for the overall scope and (pooled) for families from A/A nights.

    `aa` has one row of passes per case for each night of an unchanged state.
    φ = observed variance of the nightly pass totals / binomial variance, at
    least 1.
    """
    nights = aa.shape[0]
    if nights < 3:
        return 1.0, 1.0
    p_hat = aa.sum(axis=0) / (nights * runs)
    binom = runs * p_hat * (1 - p_hat)
    totals = aa.sum(axis=1)
    phi_overall = totals.var(ddof=1) / max(binom.sum(), 1e-12)
    fam_totals = np.stack([np.bincount(population.family, row, population.families) for row in aa])
    fam_binom = np.bincount(population.family, binom, population.families)
    phi_family = fam_totals.var(axis=0, ddof=1).sum() / max(fam_binom.sum(), 1e-12)
    return max(1.0, float(phi_overall)), max(1.0, float(phi_family))


def threshold(alpha_state: float, families: int) -> float:
    """z at which a scope test rejects: α_state split over (step, drift) x (overall, families)."""
    return NormalDist().inv_cdf(alpha_state / (2 * (1 + families)))


# ---------------------------------------------------------------------------
# the decision for one state


def confirmed(
    state: np.ndarray,
    predecessor: np.ndarray,
    reference: np.ndarray,
    reference_nights: int,
    runs: int,
    population: Population,
    phi: tuple[float, float],
    z_cut: float,
) -> tuple[bool, bool]:
    """(step rejects, drift rejects) for one nightly-measured state."""
    so, sf = scope_z(state, runs, predecessor, runs, population, *phi)
    do, df = scope_z(state, runs, reference, runs * reference_nights, population, *phi)
    return bool(so <= z_cut or (sf <= z_cut).any()), bool(do <= z_cut or (df <= z_cut).any())


@dataclass
class Scenario:
    states_per_month: int = 60
    runs: int = 10
    night_sd: float = 0.0
    aa_nights: int = 14
    budget: float = 1.0  # false issues per month


@dataclass
class NullResult:
    months: int
    false_issues: list[int] = field(default_factory=list)
    phi: list[tuple[float, float]] = field(default_factory=list)

    @property
    def per_month(self) -> float:
        return float(np.mean(self.false_issues))

    @property
    def months_with_any(self) -> float:
        return float(np.mean([n > 0 for n in self.false_issues]))


def null_months(population: Population, s: Scenario, months: int, seed: int) -> NullResult:
    """Months without any change: every alert is a false issue.

    A step rejection always opens an alert; a drift-only rejection opens one only
    when no alert is open (an open alert is extended instead). An alert closes
    when a later state shows no drift.
    """
    rng = np.random.default_rng(seed)
    z_cut = threshold(s.budget / s.states_per_month, population.families)
    result = NullResult(months)
    for _ in range(months):
        aa = np.stack([night(population, s.runs, s.night_sd, rng) for _ in range(s.aa_nights)])
        phi = estimate_dispersion(aa, s.runs, population)
        reference = aa.sum(axis=0)
        previous = night(population, s.runs, s.night_sd, rng)
        issues, open_alert = 0, False
        for _ in range(s.states_per_month):
            current = night(population, s.runs, s.night_sd, rng)
            step, drift = confirmed(current, previous, reference, s.aa_nights, s.runs,
                                    population, phi, z_cut)  # fmt: skip
            if step or (drift and not open_alert):
                issues += 1
                open_alert = True
            elif not drift:
                open_alert = False
            previous = current
        result.false_issues.append(issues)
        result.phi.append(phi)
    return result


def detection(
    population: Population, s: Scenario, drop: float, scope: str, trials: int, seed: int
) -> tuple[float, float]:
    """(on time, within one more state) for a drop at one state."""
    rng = np.random.default_rng(seed)
    z_cut = threshold(s.budget / s.states_per_month, population.families)
    on_time = late = 0
    for _ in range(trials):
        if scope == "overall":
            shift = np.full(population.p.shape, drop)
        else:
            shift = np.where(population.family == rng.integers(population.families), drop, 0.0)
        aa = np.stack([night(population, s.runs, s.night_sd, rng) for _ in range(s.aa_nights)])
        phi = estimate_dispersion(aa, s.runs, population)
        reference = aa.sum(axis=0)
        before = night(population, s.runs, s.night_sd, rng)
        at = night(population, s.runs, s.night_sd, rng, shift)
        after = night(population, s.runs, s.night_sd, rng, shift)
        hit = any(confirmed(at, before, reference, s.aa_nights, s.runs, population, phi, z_cut))
        on_time += hit
        if not hit:
            late += any(confirmed(after, at, reference, s.aa_nights, s.runs, population, phi,
                                  z_cut))  # fmt: skip
    return on_time / trials, (on_time + late) / trials


@dataclass
class Row:
    night_sd: float
    budget: float
    phi: tuple[float, float]
    false_per_month: float
    months_with_any: float
    power: dict[str, tuple[float, float]]


DROPS = (("family", 0.05), ("family", 0.10), ("family", 0.20), ("overall", 0.02),
         ("overall", 0.05))  # fmt: skip


def table(
    population: Population,
    night_sds: list[float],
    budgets: list[float],
    *,
    months: int = 1000,
    trials: int = 300,
    states_per_month: int = 60,
    runs: int = 10,
    aa_nights: int = 14,
    seed: int = 1,
) -> list[Row]:
    rows = []
    for night_sd in night_sds:
        for budget in budgets:
            s = Scenario(states_per_month, runs, night_sd, aa_nights, budget)
            null = null_months(population, s, months, seed)
            power = {
                f"{scope} -{drop * 100:.0f} pts": detection(
                    population, s, drop, scope, trials, seed
                )
                for scope, drop in DROPS
            }
            phi = tuple(float(np.mean([p[i] for p in null.phi])) for i in (0, 1))
            rows.append(Row(night_sd, budget, phi, null.per_month, null.months_with_any, power))  # type: ignore[arg-type]
    return rows


def format_rows(rows: list[Row]) -> str:
    drops = list(rows[0].power) if rows else []
    header = (
        f"{'night sd':>8} {'budget':>6} {'phi ov/fam':>11} {'false/month':>11} {'months>=1':>9} "
        + " ".join(f"{d:>18}" for d in drops)
    )
    lines = [header]
    for r in rows:
        cells = " ".join(f"{on:>8.0%} ({any_:>4.0%})    " for on, any_ in r.power.values())
        lines.append(
            f"{r.night_sd:8.2f} {r.budget:6.1f} {r.phi[0]:5.2f}/{r.phi[1]:<5.2f} "
            f"{r.false_per_month:11.2f} {r.months_with_any:9.0%} {cells}"
        )
    return "\n".join(lines)
