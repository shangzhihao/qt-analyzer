"""Measure detection power offline: how many runs are needed to catch a regression?

A population of cases with pass rates (from a real ingested run, or synthetic)
is replayed over a timeline of states. At one state a shift lowers the pass
rates of one family (or of every case). The analysis then runs exactly as on
real data, and the outcome is counted:

- push detection: the 1-run signal marks the regressed state "suspected"
- nightly confirmation: an alert on the regressed state (on time), or on a
  later state (late, e.g. caught by the drift check)
- false alarms: states before the regression (or any state when nothing
  changed) marked "suspected" or alerted

No product code runs: the analyzer never depends on the code under test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

from qt_analyzer.analyse import Agg, Config, StateData, analyse_data
from qt_analyzer.contract import ComponentVersion, State, state_key
from qt_analyzer.stores.results import ResultStore


@dataclass(frozen=True)
class SimCase:
    key: str
    family: str
    p: float


def synthetic_population(seed: int = 20261106) -> list[SimCase]:
    """10 families x 20 cases, shaped like a real LLM suite."""
    rng = random.Random(seed)
    cases = []
    for f in range(10):
        for i in range(20):
            r = rng.random()
            if r < 0.60:
                p = 1.0
            elif r < 0.80:
                p = rng.uniform(0.95, 0.99)
            elif r < 0.95:
                p = rng.uniform(0.05, 0.95)
            else:
                p = rng.uniform(0.0, 0.02)
            cases.append(SimCase(f"f{f}c{i:02d}", f"family{f}", p))
    return cases


def population_from_results(results: ResultStore, run_id: str | None = None) -> list[SimCase]:
    """Per-case pass rates of one complete run (default: the run with the most executions)."""
    index = {k: e for k, e in results.load_index().items() if e.status == "complete"}
    if not index:
        raise ValueError(f"no complete runs in {results.root}")
    if run_id is None:
        run_id = max(index, key=lambda r: sum(c.n for c in results.get_counts(r)))
    cases = []
    for c in results.get_counts(run_id):
        judged = c.passed + c.failed
        if judged:
            cases.append(SimCase(c.case_key, c.family, c.passed / judged))
    return cases


@dataclass(frozen=True)
class Setting:
    shift: float
    push_runs: int
    nightly_runs: int


@dataclass
class Outcome:
    setting: Setting
    trials: int
    effective_drop: float  # mean pass-rate drop of the shifted cases
    push_detected: float
    nightly_confirmed: float
    nightly_late: float
    false_suspected_per_state: float
    false_alerts_per_state: float
    executions_per_state: int


def _shifted(cases: list[SimCase], family: str | None, shift: float) -> list[SimCase]:
    return [
        SimCase(c.key, c.family, max(0.0, c.p - shift))
        if family is None or c.family == family
        else c
        for c in cases
    ]


def _counts(cases: list[SimCase], runs: int, rng: random.Random) -> dict:
    counts = {}
    for c in cases:
        passed = rng.binomialvariate(runs, c.p) if runs else 0
        counts[(c.key, "d")] = Agg(c.family, c.key, runs, passed, runs - passed, 0)
    return counts


def _timeline(
    base: list[SimCase], regressed: list[SimCase], states: int, at: int, s: Setting,
    rng: random.Random,
) -> list[StateData]:  # fmt: skip
    timeline = []
    for i in range(states):
        sha = f"g{i}"
        state = State(
            state_key({"graph": sha}),
            "graph",
            {"graph": ComponentVersion("sim/graph", sha, f"2026-01-01T{i:02d}:00:00Z")},
        )
        cases = regressed if i >= at else base
        data = StateData(state)
        data.counts["main"] = _counts(cases, s.push_runs, rng)
        data.counts["nightly"] = _counts(cases, s.nightly_runs, rng)
        data.runs = {"main": [f"m{i}"], "nightly": [f"n{i}"]}
        timeline.append(data)
    return timeline


def run_setting(
    population: list[SimCase],
    setting: Setting,
    *,
    trials: int,
    states: int,
    scope: str,
    config: Config,
    rng: random.Random,
) -> Outcome:
    families = sorted({c.family for c in population})
    at = states // 2
    detected = confirmed = late = 0
    false_suspected = false_alerts = 0
    drops = []
    for _ in range(trials):
        family = None if scope == "overall" else rng.choice(families)
        regressed = _shifted(population, family, setting.shift)
        changed = [(a.p, b.p) for a, b in zip(population, regressed, strict=True) if a.p != b.p]
        drops.append(sum(a - b for a, b in changed) / len(changed) if changed else 0.0)
        analysis = analyse_data(
            _timeline(population, regressed, states, at, setting, rng), [], config
        )
        entries = analysis.timeline
        target = entries[at].data.state.key
        after = {e.data.state.key for e in entries[at + 1 :]}
        if setting.shift > 0:
            detected += entries[at].push is not None and entries[at].push.status == "suspected"
            on_time = any(a.state == target for a in analysis.alerts)
            confirmed += on_time
            late += not on_time and any(a.state in after for a in analysis.alerts)
        clean = entries if setting.shift == 0 else entries[:at]
        false_suspected += sum(e.push is not None and e.push.status == "suspected" for e in clean)
        clean_keys = {e.data.state.key for e in clean}
        false_alerts += sum(a.state in clean_keys for a in analysis.alerts)
    clean_states = trials * (states if setting.shift == 0 else at)
    return Outcome(
        setting,
        trials,
        sum(drops) / len(drops),
        detected / trials if setting.shift > 0 else float("nan"),
        confirmed / trials if setting.shift > 0 else float("nan"),
        late / trials if setting.shift > 0 else float("nan"),
        false_suspected / clean_states,
        false_alerts / clean_states,
        len(population) * (setting.push_runs + setting.nightly_runs),
    )


def sweep(
    population: list[SimCase],
    shifts: list[float],
    push_runs: list[int],
    nightly_runs: list[int],
    *,
    trials: int = 40,
    states: int = 8,
    scope: str = "family",
    config: Config | None = None,
    seed: int = 1,
) -> list[Outcome]:
    rng = random.Random(seed)
    config = config or Config()
    return [
        run_setting(population, Setting(shift, p, n), trials=trials, states=states,
                    scope=scope, config=config, rng=rng)
        for shift, p, n in product(shifts, push_runs, nightly_runs)
    ]  # fmt: skip


def format_table(outcomes: list[Outcome]) -> str:
    header = (
        f"{'shift':>5} {'eff.drop':>8} {'push':>4} {'nightly':>7} "
        f"{'push detect':>11} {'confirmed':>9} {'late':>5} {'false susp/state':>16} "
        f"{'false alert/state':>17} {'exec/state':>10}"
    )
    lines = [header]
    for o in outcomes:
        s = o.setting

        def pct(x: float) -> str:
            return "    -" if x != x else f"{x:5.0%}"

        lines.append(
            f"{s.shift:5.2f} {o.effective_drop:8.3f} {s.push_runs:4d} {s.nightly_runs:7d} "
            f"{pct(o.push_detected):>11} {pct(o.nightly_confirmed):>9} {pct(o.nightly_late):>5} "
            f"{o.false_suspected_per_state:16.3f} {o.false_alerts_per_state:17.3f} "
            f"{o.executions_per_state:10d}"
        )
    return "\n".join(lines)
