"""Command line: `qt-analyzer ingest|analyse|status|comment|simulate`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qt_analyzer import budget
from qt_analyzer.analyse import Analysis, Config, analyse, pass_rate, save
from qt_analyzer.ingest import ingest_pending
from qt_analyzer.report import github_request, sync_pr_comments
from qt_analyzer.simulate import (
    format_table,
    population_from_results,
    sweep,
    synthetic_population,
)
from qt_analyzer.stores import LocalDirStore, ResultStore
from qt_analyzer.stores.github import GitHubArtifactsStore


def _short(key: str | None) -> str:
    return key.removeprefix("sha256:")[:10] if key else "-"


def _rate(value: float | None) -> str:
    return f"{value:6.1%}" if value is not None else "     -"


def print_status(analysis: Analysis, out=sys.stdout) -> None:
    print(
        f"{'state':10}  {'created by':10}  {'main':>6}  {'nightly':>7}  push        nightly",
        file=out,
    )
    for e in analysis.timeline:
        d = e.data
        created = f"{d.state.created_by}@{d.state.components[d.state.created_by].sha[:7]}"
        push = e.push.status if e.push else "-"
        if e.push and e.push.flagged:
            push += f" ({', '.join(e.push.flagged)})"
        nightly = e.nightly.status if e.nightly else "-"
        if e.nightly and (e.nightly.step_flagged or e.nightly.reference_flagged):
            scopes = sorted(set(e.nightly.step_flagged) | set(e.nightly.reference_flagged))
            nightly += f" ({', '.join(scopes)})"
        print(
            f"{_short(d.state.key):10}  {created:10}  "
            f"{_rate(pass_rate(d.counts['main']))}  "
            f"{_rate(pass_rate(d.counts['nightly'])):>7}  {push:10}  {nightly}",
            file=out,
        )
    open_alerts = [a for a in analysis.alerts if a.open]
    print(f"\nreference: {_short(analysis.reference)}   open alerts: {len(open_alerts)}", file=out)
    for a in open_alerts:
        print(f"  {_short(a.state)}: {', '.join(a.scopes) or '-'}  cases: {len(a.cases)}", file=out)
    for p in analysis.prs:
        extra = f" ({', '.join(p.flagged)})" if p.flagged else ""
        print(
            f"PR {p.repo}#{p.number}: {p.status}{extra}  suspected cases: {len(p.suspected_cases)}",
            file=out,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qt-analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest every run not yet complete")
    source = p_ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--logs", type=Path, help="LocalDirStore root")
    source.add_argument("--github", metavar="OWNER/REPO", help="read runs from Actions artifacts")
    p_ingest.add_argument("--cache", type=Path, default=Path(".qt-cache"))
    p_ingest.add_argument("--results", type=Path, required=True, help="results directory")

    for name, text in (
        ("analyse", "analyse and write results"),
        ("status", "print a summary"),
        ("comment", "create or update PR comments (needs QT_PAT)"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--results", type=Path, required=True, help="results directory")
        p.add_argument("--config", type=Path, default=Path("analyzer.toml"))

    p_sim = sub.add_parser("simulate", help="measure detection power offline")
    p_sim.add_argument(
        "--population",
        type=Path,
        help="results directory whose largest complete run gives the case pass rates "
        "(default: a synthetic population)",
    )
    p_sim.add_argument("--run-id", help="use this run as the population")
    p_sim.add_argument("--shifts", default="0,0.05,0.1,0.2,0.3")
    p_sim.add_argument("--push-runs", default="1")
    p_sim.add_argument("--nightly-runs", default="5,10,20")
    p_sim.add_argument("--scope", choices=["family", "overall"], default="family")
    p_sim.add_argument("--trials", type=int, default=40)
    p_sim.add_argument("--states", type=int, default=8)
    p_sim.add_argument("--seed", type=int, default=1)
    p_sim.add_argument("--config", type=Path, default=Path("analyzer.toml"))

    p_budget = sub.add_parser(
        "budget", help="measure the nightly rule: false issues per month and power"
    )
    source = p_budget.add_mutually_exclusive_group()
    source.add_argument("--population", type=Path, help="results directory (largest run)")
    source.add_argument("--population-json", type=Path, help='JSON list of {"p", "family"}')
    p_budget.add_argument("--night-sd", default="0,0.1,0.2,0.3",
                          help="night-to-night logit variation to simulate")  # fmt: skip
    p_budget.add_argument("--budgets", default="0.5,1,2", help="false issues per month")
    p_budget.add_argument("--months", type=int, default=1000)
    p_budget.add_argument("--trials", type=int, default=300)
    p_budget.add_argument("--states", type=int, default=60, help="nightly states per month")
    p_budget.add_argument("--runs", type=int, default=10, help="nightly runs per case")
    p_budget.add_argument("--aa-nights", type=int, default=14)
    p_budget.add_argument("--seed", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "budget":
        if args.population_json:
            rows = json.loads(args.population_json.read_text())
            population = budget.population_from([r["p"] for r in rows],
                                                [r["family"] for r in rows])  # fmt: skip
        elif args.population:
            cases = population_from_results(ResultStore(args.population))
            population = budget.population_from([c.p for c in cases], [c.family for c in cases])
        else:
            population = budget.realistic_population()
        print(
            f"population: {len(population.p)} cases in {population.families} families; "
            f"{args.states} nightly states/month, {args.runs} runs, {args.aa_nights} A/A nights; "
            f"{args.months} null months, {args.trials} trials per drop"
        )
        rows = budget.table(
            population,
            [float(x) for x in args.night_sd.split(",")],
            [float(x) for x in args.budgets.split(",")],
            months=args.months,
            trials=args.trials,
            states_per_month=args.states,
            runs=args.runs,
            aa_nights=args.aa_nights,
            seed=args.seed,
        )
        print(budget.format_rows(rows))
        print("power: on time (within one more state)")
        return 0
    if args.command == "simulate":
        population = (
            population_from_results(ResultStore(args.population), args.run_id)
            if args.population
            else synthetic_population()
        )
        outcomes = sweep(
            population,
            [float(x) for x in args.shifts.split(",")],
            [int(x) for x in args.push_runs.split(",")],
            [int(x) for x in args.nightly_runs.split(",")],
            trials=args.trials,
            states=args.states,
            scope=args.scope,
            config=Config.load(args.config),
            seed=args.seed,
        )
        print(
            f"population: {len(population)} cases; scope: {args.scope}; "
            f"{args.trials} trials x {args.states} states per setting"
        )
        print(format_table(outcomes))
        return 0
    results = ResultStore(args.results)
    if args.command == "ingest":
        token = os.environ.get("QT_PAT") or os.environ.get("GITHUB_TOKEN")
        logs = (
            GitHubArtifactsStore(args.github, token, args.cache)
            if args.github
            else LocalDirStore(args.logs)
        )
        report = ingest_pending(logs, results)
        print(
            f"complete: {len(report.complete)}  incomplete: {len(report.incomplete)}  "
            f"rejected: {len(report.rejected)}"
        )
        for run_id in report.incomplete + report.rejected:
            print(f"  {run_id}: {results.load_index()[run_id].detail}")
        return 0
    analysis = analyse(results, Config.load(args.config))
    if args.command == "comment":
        token = os.environ.get("QT_PAT")
        if not token:
            print("QT_PAT is not set", file=sys.stderr)
            return 1
        for key in sync_pr_comments(results, analysis.prs, github_request(token)):
            print(f"commented on {key}")
        return 0
    if args.command == "analyse":
        save(results, analysis)
    print_status(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
