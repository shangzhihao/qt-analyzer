"""Command line: `qt-analyzer ingest|analyse|status|comment|simulate|gates|case|report`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qt_analyzer import case_check, gate_compare
from qt_analyzer.analyse import Analysis, Config, analyse, pass_rate, save
from qt_analyzer.fetch import DEFAULT_REPO, fetch_results
from qt_analyzer.gates import available_gates
from qt_analyzer.ingest import ingest_pending
from qt_analyzer.locators import format_location, location_from_json
from qt_analyzer.report import github_request, sync_pr_comments
from qt_analyzer.simulate import (
    SimCase,
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
        if e.push and e.push.status == "not-comparable":
            push += " (tests changed)"
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
        print(f"  {_short(a.state)}: {', '.join(a.scopes) or '-'}", file=out)
        if a.where:
            for line in format_location(location_from_json(a.where)):
                print(f"    {line}", file=out)
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

    p_gates = sub.add_parser("gates", help="compare gate algorithms")
    what = p_gates.add_mutually_exclusive_group(required=True)
    what.add_argument("--list", action="store_true", help="list registered gates")
    what.add_argument("--example", choices=[*gate_compare.EXAMPLES, "all"])
    what.add_argument("--benchmark", action="store_true", help="all scenarios, random trials")
    what.add_argument(
        "--locate", action="store_true", help="benchmark the configured locator and gate"
    )
    p_gates.add_argument("--gates", help="comma-separated gate names (default: all)")
    p_gates.add_argument("--trials", type=int, default=300)
    p_gates.add_argument("--runs", type=int, default=1, help="new runs per case")
    p_gates.add_argument("--baseline-runs", type=int, default=20)
    population = p_gates.add_mutually_exclusive_group()
    population.add_argument("--population", type=Path, help="results directory (largest run)")
    population.add_argument("--population-json", type=Path, help='JSON list of {"p", "family"}')
    p_gates.add_argument("--seed", type=int, default=1)
    p_gates.add_argument(
        "--equal-false-alarms",
        type=float,
        metavar="RATE",
        help="tune every gate to this false-regression rate before comparing",
    )
    p_gates.add_argument("--tune-trials", type=int, default=400, help="unchanged draws for tuning")
    p_gates.add_argument("--config", type=Path, default=Path("analyzer.toml"))

    p_case = sub.add_parser(
        "case", help="regression or improvement of single test cases, from local runs"
    )
    p_case.add_argument("logs", nargs="+", type=Path, help="pytest --report-log files")
    source = p_case.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="local copy of the results")
    source.add_argument("--fetch", action="store_true",
                        help="download the published results (the default)")  # fmt: skip
    p_case.add_argument("--repo", default=DEFAULT_REPO, help="repository that publishes results")
    p_case.add_argument("--cache", type=Path, default=Path(".qt-cache"))
    p_case.add_argument("--states", type=int, default=5, help="newest main states to pool")
    p_case.add_argument("--alpha", type=float, default=0.05)
    p_case.add_argument("--delta", type=float, default=0.10,
                        help="change size for the runs-needed hint")  # fmt: skip

    p_report = sub.add_parser("report", help="write a comprehensive HTML report")
    source = p_report.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="local copy of the results")
    source.add_argument("--fetch", action="store_true", help="download the published results")
    p_report.add_argument("--repo", default=DEFAULT_REPO, help="repository that publishes results")
    p_report.add_argument("--cache", type=Path, default=Path(".qt-cache"))
    logs = p_report.add_mutually_exclusive_group()
    logs.add_argument("--logs", type=Path, help="runs folder (or log files) for failure messages")
    logs.add_argument("--logs-github", metavar="OWNER/REPO",
                      help="failure messages from that repo's Actions artifacts")  # fmt: skip
    p_report.add_argument("--logs-runs", type=int, default=20, help="newest runs to read logs from")
    p_report.add_argument("--out", type=Path, default=Path("report.html"))
    p_report.add_argument("--config", type=Path, default=Path("analyzer.toml"))

    args = parser.parse_args(argv)
    if args.command == "report":
        from qt_analyzer.html_report.html import render
        from qt_analyzer.html_report.model import build_model, failure_samples

        results_dir = args.results or fetch_results(args.cache, args.repo)
        failures = None
        if args.logs:
            paths = (
                [args.logs] if args.logs.is_file() else sorted(args.logs.rglob("shard-*.jsonl*"))
            )
            failures = failure_samples(paths)
        elif args.logs_github:
            token = os.environ.get("QT_PAT") or os.environ.get("GITHUB_TOKEN")
            store = GitHubArtifactsStore(args.logs_github, token, args.cache / "artifacts")
            newest = store.list_runs()[-args.logs_runs :]
            failures = failure_samples([p for run in newest for p in store.get_shard_logs(run)])
        model = build_model(ResultStore(results_dir), Config.load(args.config), failures=failures,
                            source=str(results_dir))  # fmt: skip
        args.out.write_text(render(model), encoding="utf-8")
        print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB): {len(model['states'])} "
              f"states, {len(model['alerts'])} alerts, {len(model['prs'])} PRs, "
              f"{len(model['cases'])} test cases")  # fmt: skip
        return 0
    if args.command == "case":
        results_dir = args.results or fetch_results(args.cache, args.repo)
        main_counts, states = case_check.main_counts(ResultStore(results_dir), args.states)
        if not states:
            print(f"no measured main states in {results_dir}", file=sys.stderr)
            return 2
        verdicts = case_check.check(case_check.local_counts(args.logs), main_counts,
                                    alpha=args.alpha, delta=args.delta)  # fmt: skip
        print(case_check.format_verdicts(verdicts, states, args.delta))
        return 1 if any(v.verdict == "regression" for v in verdicts) else 0
    if args.command == "gates":
        config = Config.load(args.config)  # also imports gate_modules
        if args.list:
            print("\n".join(available_gates()))
            return 0
        names = args.gates.split(",") if args.gates else None
        gates = gate_compare.build_gates(names, config.gate_params)
        if args.example:
            chosen = (
                gate_compare.EXAMPLES
                if args.example == "all"
                else {args.example: gate_compare.EXAMPLES[args.example]}
            )
            for name, (text, make) in chosen.items():
                print(f"=== {name}: {text}")
                print(gate_compare.format_verdicts(gate_compare.judge_all(make(), gates)))
                print()
            return 0
        population = None
        if args.population:
            population = population_from_results(ResultStore(args.population))
        elif args.population_json:
            rows = json.loads(args.population_json.read_text())
            population = [SimCase(f"case{i}", r["family"], r["p"]) for i, r in enumerate(rows)]
        if args.locate:
            from qt_analyzer.analyse import stage_params
            from qt_analyzer.gates import make_gate
            from qt_analyzer.locators import make_locator

            stage = "nightly" if args.runs > 1 else "pr"
            gate = make_gate(config.gate, stage_params(config, stage))
            locator = make_locator(config.locator, config.locator_params.get(config.locator))
            table = gate_compare.locator_benchmark(gate, locator, population=population,
                                                   trials=args.trials,
                                                   baseline_runs=args.baseline_runs,
                                                   runs=args.runs, seed=args.seed)  # fmt: skip
            print(f"gate {config.gate} ({stage} settings), locator {config.locator}; "
                  f"{args.trials} trials, baseline {args.baseline_runs} runs, "
                  f"new {args.runs}")  # fmt: skip
            print(gate_compare.format_locator_benchmark(table))
            return 0
        if args.equal_false_alarms is not None:
            gates, notes = gate_compare.equalise_false_alarms(
                gates, args.equal_false_alarms, population=population,
                baseline_runs=args.baseline_runs, runs=args.runs, trials=args.tune_trials,
            )  # fmt: skip
            print(f"tuned to {args.equal_false_alarms:.0%} false regressions on unchanged data:")
            for name, note in notes.items():
                print(f"  {name}: {note}")
        table = gate_compare.benchmark(gates, population=population, trials=args.trials,
                                       baseline_runs=args.baseline_runs, runs=args.runs,
                                       seed=args.seed)  # fmt: skip
        print(f"{args.trials} trials per scenario; baseline {args.baseline_runs} runs, "
              f"new {args.runs} run(s) per case")  # fmt: skip
        print(gate_compare.format_benchmark(table))
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
        for key in sync_pr_comments(analysis.prs, github_request(token)):
            print(f"commented on {key}")
        return 0
    if args.command == "analyse":
        save(results, analysis)
    print_status(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
