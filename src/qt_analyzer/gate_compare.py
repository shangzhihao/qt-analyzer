"""Compare gate algorithms: two worked examples and a fair benchmark.

Every registered gate (see `gates.py`) is judged on exactly the same data.

Examples (fixed counts, same result every time):
- `regression`: one topic drops sharply while others improve, so the total is
  unchanged. A gate that only looks at the total cannot see it.
- `improvement`: one topic improves sharply. A one-sided gate cannot see it.

Benchmark (random draws): a population of cases with pass rates, a 20-run
baseline (testimate's reference size) and new runs (1 by default, as testimate's
`compare`), over several scenarios — including the broad drop that a
total-based gate handles well. Each cell is the share of trials in which the gate
flagged what the scenario contains; for "unchanged" it is the false-alarm rate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qt_analyzer.analyse import Agg, Comparison, compare
from qt_analyzer.gates import Gate, Verdict, available_gates, make_gate
from qt_analyzer.simulate import SimCase, synthetic_population

Counts = dict[tuple[str, str], Agg]


def _counts(rows: Sequence[tuple[str, str, int, int]]) -> Counts:
    """(case, family, passed, runs) -> counts."""
    return {(case, "d"): Agg(family, case, runs, passed, runs - passed, 0)
            for case, family, passed, runs in rows}  # fmt: skip


# ---------------------------------------------------------------------------
# examples


def _example_cases() -> list[tuple[str, str]]:
    return [(f"topic{f}-{i:02d}", f"topic{f}") for f in range(10) for i in range(20)]


def example_regression() -> Comparison:
    """topic0 drops from 80 % to 40 %; topic1-4 rise from 80 % to 90 %; total unchanged."""
    cases = _example_cases()
    base = _counts([(c, f, 16, 20) for c, f in cases])
    new_passes = {"topic0": 4, "topic1": 9, "topic2": 9, "topic3": 9, "topic4": 9}
    new = _counts([(c, f, new_passes.get(f, 8), 10) for c, f in cases])
    return compare(new, base)


def example_improvement() -> Comparison:
    """topic0 rises from 50 % to 90 %; every other topic stays at 80 %."""
    cases = _example_cases()
    base = _counts([(c, f, 10 if f == "topic0" else 16, 20) for c, f in cases])
    new = _counts([(c, f, 9 if f == "topic0" else 8, 10) for c, f in cases])
    return compare(new, base)


EXAMPLES: dict[str, tuple[str, Callable[[], Comparison]]] = {
    "regression": (
        "topic0 drops 80% -> 40% while topic1-4 rise 80% -> 90%; the total is unchanged "
        "(200 cases in 10 topics, baseline 20 runs, new 10 runs)",
        example_regression,
    ),
    "improvement": (
        "topic0 rises 50% -> 90%; the other topics stay at 80% "
        "(200 cases in 10 topics, baseline 20 runs, new 10 runs)",
        example_improvement,
    ),
}


def judge_all(comparison: Comparison, gates: dict[str, Gate]) -> dict[str, Verdict]:
    return {name: gate.judge(comparison) for name, gate in gates.items()}


def format_verdicts(verdicts: dict[str, Verdict]) -> str:
    lines = []
    for name, v in verdicts.items():
        found = []
        if v.regressed:
            found.append(f"REGRESSION in {', '.join(v.regressed)}")
        if v.improved:
            found.append(f"improvement in {', '.join(v.improved)}")
        lines.append(f"{name:22} {'; '.join(found) or 'nothing detected'}")
        for scope in dict.fromkeys(("overall", *v.regressed, *v.improved)):
            if scope in v.detail:
                lines.append(f"{'':22}   {scope}: {v.detail[scope]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# benchmark


@dataclass(frozen=True)
class Choice:
    """What a trial picked: three topics (with >= 10 cases where possible) and
    five cases that usually pass."""

    topics: list[str]
    cases: set[str]


@dataclass(frozen=True)
class Scenario:
    name: str
    expect: str  # "regression" | "improvement" | "false regression" | "false improvement"
    shift: Callable[[SimCase, Choice], float]  # new pass rate


def _clip(p: float) -> float:
    return min(1.0, max(0.0, p))


def _topic(c: SimCase, choice: Choice, delta: float, which: slice = slice(0, 1)) -> float:
    return _clip(c.p + delta) if c.family in choice.topics[which] else c.p


SCENARIOS: list[Scenario] = [
    Scenario("unchanged", "false regression", lambda c, ch: c.p),
    Scenario("unchanged ", "false improvement", lambda c, ch: c.p),
    Scenario("all topics -2 pts", "regression", lambda c, ch: _clip(c.p - 0.02)),
    Scenario("all topics -5 pts", "regression", lambda c, ch: _clip(c.p - 0.05)),
    Scenario("one topic -15 pts", "regression", lambda c, ch: _topic(c, ch, -0.15)),
    Scenario("one topic -30 pts", "regression", lambda c, ch: _topic(c, ch, -0.30)),
    Scenario("one topic -30, two +15 (masked)", "regression",
             lambda c, ch: _topic(c, ch, -0.30) if c.family == ch.topics[0]
             else _topic(c, ch, 0.15, slice(1, 3))),
    Scenario("flaky cases -30 pts", "regression",
             lambda c, ch: _clip(c.p - 0.30) if 0.05 < c.p < 0.95 else c.p),
    Scenario("always-pass cases -5 pts", "regression",
             lambda c, ch: 0.95 if c.p == 1.0 else c.p),
    Scenario("five cases collapse to 0%", "regression",
             lambda c, ch: 0.0 if c.key in ch.cases else c.p),
    Scenario("one topic +30 pts", "improvement", lambda c, ch: _topic(c, ch, 0.30)),
]  # fmt: skip


def _choose(cases: list[SimCase], rng: random.Random) -> Choice:
    sizes: dict[str, int] = {}
    for c in cases:
        sizes[c.family] = sizes.get(c.family, 0) + 1
    big = sorted(f for f, n in sizes.items() if n >= 10) or sorted(sizes)
    topics = rng.sample(big, min(3, len(big)))
    likely = [c.key for c in cases if c.p >= 0.5]
    return Choice(topics, set(rng.sample(likely, min(5, len(likely)))))


def _draw(cases: list[SimCase], rates: list[float], runs: int, rng: random.Random) -> Counts:
    rows = []
    for case, p in zip(cases, rates, strict=True):
        passed = sum(rng.random() < p for _ in range(runs))
        rows.append((case.key, case.family, passed, runs))
    return _counts(rows)


def benchmark(
    gates: dict[str, Gate],
    *,
    population: list[SimCase] | None = None,
    trials: int = 300,
    baseline_runs: int = 20,
    runs: int = 1,
    seed: int = 1,
) -> dict[str, dict[str, float]]:
    """scenario -> gate -> share of trials where the gate flagged what is expected."""
    cases = population or synthetic_population()
    rng = random.Random(seed)
    table: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        hits = dict.fromkeys(gates, 0)
        for _ in range(trials):
            choice = _choose(cases, rng)
            base = _draw(cases, [c.p for c in cases], baseline_runs, rng)
            new = _draw(cases, [scenario.shift(c, choice) for c in cases], runs, rng)
            comparison = compare(new, base)
            for name, gate in gates.items():
                v = gate.judge(comparison)
                if scenario.expect.endswith("improvement"):
                    hits[name] += bool(v.improved)
                else:
                    hits[name] += bool(v.regressed)
        table[scenario.name] = {name: hits[name] / trials for name in gates}
    return table


def format_benchmark(table: dict[str, dict[str, float]]) -> str:
    gates = list(next(iter(table.values())))
    width = max(len(g) for g in gates) + 2
    lines = [f"{'scenario':34}" + "".join(f"{g:>{width}}" for g in gates)]
    for scenario in SCENARIOS:
        row = table[scenario.name]
        label = scenario.name.strip()
        if scenario.expect.startswith("false"):
            label += f" ({scenario.expect}s)"
        lines.append(f"{label:34}" + "".join(f"{row[g]:>{width}.0%}" for g in gates))
    return "\n".join(lines)


def equalise_false_alarms(
    gates: dict[str, Gate],
    target: float,
    *,
    population: list[SimCase] | None = None,
    trials: int = 400,
    baseline_runs: int = 20,
    runs: int = 1,
    seed: int = 1000,
) -> tuple[dict[str, Gate], dict[str, str]]:
    """Tune every gate that has `tuned(level)` to `target` false regressions on
    unchanged data; return the tuned gates and a note per gate.

    The unchanged comparisons are drawn once (with a seed different from the
    benchmark's) and reused for every level, so the search is exact and cheap.
    """
    cases = population or synthetic_population()
    rng = random.Random(seed)
    nulls = []
    for _ in range(trials):
        base = _draw(cases, [c.p for c in cases], baseline_runs, rng)
        nulls.append(compare(_draw(cases, [c.p for c in cases], runs, rng), base))

    def rate(gate: Gate) -> float:
        return sum(bool(gate.judge(c).regressed) for c in nulls) / trials

    tuned: dict[str, Gate] = {}
    notes: dict[str, str] = {}
    for name, gate in gates.items():
        if not hasattr(gate, "tuned"):
            tuned[name], notes[name] = gate, "not tunable; as configured"
            continue
        lo, hi = -7.0, math.log10(0.99)  # log10(level): lo raises no more than target
        for _ in range(20):
            mid = (lo + hi) / 2
            if rate(gate.tuned(10**mid)) <= target:
                lo = mid
            else:
                hi = mid
        tuned[name] = gate.tuned(10**lo)
        notes[name] = f"level {10**lo:.2e}"
    return tuned, notes


def build_gates(names: Sequence[str] | None, params: dict[str, dict]) -> dict[str, Gate]:
    names = list(names) if names else available_gates()
    stages = ("push", "nightly", "pr")  # per-stage sub-tables are for the analysis

    def base(name: str) -> dict:
        return {k: v for k, v in params.get(name, {}).items() if k not in stages}

    return {name: make_gate(name, base(name)) for name in names}
