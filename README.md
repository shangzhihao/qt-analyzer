# qt-analyzer

Analyses the pytest report logs of a non-deterministic test suite, state by
state: which combination of component versions (e.g. `qt-graph` + `qt-data`)
was tested, how its pass rates compare with earlier states, and whether a drop
is noise, suspected or confirmed.

It never imports the code under test. Its only input is the run contract: a
`manifest.json` plus one pytest-reportlog file per shard (see
[the design](#contract)).

## Usage

```sh
uv sync
uv run qt-analyzer ingest --logs runs/ --results results/
uv run qt-analyzer analyse --results results/
uv run qt-analyzer status --results results/
```

- `ingest` reads every run in `runs/<run_id>/` that is not yet complete and
  stores per-case counts in `results/`.
- `analyse` rebuilds the timeline, nightly detection, alerts and PR results
  and writes them to `results/` (the layout of the `qt-results` branch).
  Verdicts are taken once and recorded in `decisions.jsonl`: a state or PR run
  that was decided keeps its verdict on every later pass, so a threshold change
  or late evidence never rewrites an alert that was already reported.
- `comment` creates or updates one comment per PR, found by its marker. CI runs
  it only after the results are pushed.
- `status` prints the same summary without writing.
- `simulate` measures detection power offline: it replays a population of
  per-case pass rates (from `--population results/`, or synthetic) over a
  timeline, injects a drop at one state and reports how often it is suspected,
  confirmed on time or late, and how often unchanged states raise false alarms.

```sh
uv run qt-analyzer simulate --shifts 0,0.05,0.1,0.2 --push-runs 1 --nightly-runs 5,10,20
```

Thresholds come from `analyzer.toml` (`[analysis]` table).

## Gate algorithms

A gate decides whether a comparison (new state vs. baseline, paired by case)
shows a regression or an improvement, overall or per family. Registered gates:

| Gate | What it does |
|---|---|
| `zscore` (default) | Per scope, observed passes vs. those expected from the baseline; pooled variance, continuity correction; both directions. |
| `testimate_threshold` | testimate's gate in use (`Threshold4StdDev` + `OmegaCubic`), ported exactly: weighted failure rate with ω = p³, fails above reference + 10 σ. Total only, regressions only. |
| `poisson_binomial` | testimate's `PoissonBinomialGate` (implemented, not used), ported exactly: total passes vs. the Poisson-binomial distribution of the baseline rates, lower tail at α = 0.05. Total only, regressions only. |
| `bayes` | Posterior predictive check: are the new passes surprising under the baseline's Beta posteriors, shifted by `delta`? Per scope, both directions. |
| `zscore_split` | `zscore`'s statistic with the regression budget split between the total (50 %), the flaky cases (20 %) and the topics (rest), and a separate, stricter budget for improvements. |

`gate = "<name>"` in `analyzer.toml` (`[analysis]`) selects the gate that makes
the decisions; `[gates.<name>]` holds its settings.

```sh
uv run qt-analyzer gates --list
uv run qt-analyzer gates --example all          # two fixed examples, every gate
uv run qt-analyzer gates --benchmark            # 7 scenarios, random trials
uv run qt-analyzer gates --benchmark --gates zscore,poisson_binomial --runs 10
uv run qt-analyzer gates --benchmark --equal-false-alarms 0.05   # fair: same false-alarm rate
```

A gate can implement `tuned(level)` (the same gate at another sensitivity) so
`--equal-false-alarms` can compare it with the others at the same false-alarm
rate; gates without it are compared as configured.

### Adding a gate

```python
# my_gates.py
from dataclasses import dataclass
from qt_analyzer.gates import Verdict, register_gate


@register_gate("my_gate")
@dataclass(frozen=True)
class MyGate:
    threshold: float = 0.1

    def judge(self, comparison):  # analyse.Comparison
        verdict = Verdict()
        if comparison.overall.rate_s < comparison.overall.rate_b - self.threshold:
            verdict.regressed.append("overall")
        return verdict
```

```toml
# analyzer.toml
[analysis]
gate_modules = ["my_gates"]   # importable from where qt-analyzer runs
[gates.my_gate]
threshold = 0.05
```

Every registered gate then appears in `gates --example` and `--benchmark`, and
`gate = "my_gate"` makes it decide the analysis.

## Contract

Schema version 2. A run directory holds:

- `manifest.json`: run id, kind (`pr`, `main`, `nightly`), the tested state (one
  version per component, with its commit time, and for main states the
  component that created it), runs per case, shard count, seed. A `pr` run also
  names its **baseline**: the main state (component versions) to compare with.
  Producers do not send a state key; the analyzer computes it.
- `shard-<i>-of-<n>.jsonl.gz`: pytest `--report-log` output. Every test report
  carries the user properties `qt_case_key`, `qt_definition_hash`,
  `qt_family`, `qt_repeat` and `qt_hrid`.

A run is analysed only when all `n` shards are present and finished. If a PR's
baseline was never measured, it is compared with the nearest measured main
state, and the PR comment says so.
