# qt-det

Analyses the pytest report logs of a non-deterministic test suite, state by
state: which combination of component versions (e.g. `qt-app` + `qt-dep`)
was tested, how its pass rates compare with earlier states, and whether a drop
is noise, suspected or confirmed.

It never imports the code under test. Its only input is the run contract: a
`manifest.json` plus one pytest-reportlog file per shard (see
[the design](#contract)).

![Nightly pass rate per main state](https://raw.githubusercontent.com/shangzhihao/qt-det/qt-results/quality.svg)

The chart is redrawn by every analysis (`quality.svg` on the `qt-results`
branch): the nightly pass rate of each main state, where alerts opened, step
improvements, the reference, and where the case set changed.

## Usage

```sh
uv sync
uv run qt-det ingest --logs runs/ --results results/
uv run qt-det analyse --results results/
uv run qt-det status --results results/
```

- `ingest` reads every run in `runs/<run_id>/` that is not yet complete and
  stores per-case counts in `results/`.
- `analyse` rebuilds the timeline, nightly detection, alerts and PR results
  and writes them to `results/` (the layout of the `qt-results` branch).
  Verdicts are taken once and recorded in `decisions.jsonl`: a state or PR run
  that was decided keeps its verdict on every later pass, so a threshold change
  or late evidence never rewrites an alert that was already reported.
  A push or PR run (1 run per case) is compared with its base state pooled
  with earlier states, but only back past states the nightly judged unchanged,
  so a change on main that is not yet confirmed is not reported again on later
  states and PRs.
  Each nightly state is compared with the previous one (step) and with the
  reference. A drop in a step, or a scope below the reference that no open
  alert covers, opens an alert holding only those scopes. Each alert stays open
  while its own topics are still below the reference and closes on its own
  once they are not; a step improvement is reported as `improved`.
- `comment` creates or updates one comment per PR, found by its marker. CI runs
  it only after the results are pushed.
- `status` prints the same summary without writing.
- `simulate` measures detection power offline: it replays a population of
  per-case pass rates (from `--population results/`, or synthetic) over a
  timeline, injects a drop at one state and reports how often it is suspected,
  confirmed on time or late, and how often unchanged states raise false alarms.

```sh
uv run qt-det simulate --shifts 0,0.05,0.1,0.2 --push-runs 1 --nightly-runs 5,10,20
```

Thresholds come from `analyzer.toml` (`[analysis]` table).

## Checking single test cases locally

In CI every case runs once, too few to judge one case. Locally one case can run
many times; `case` compares those runs with the same case's history on main:

```sh
# in qt-app, with your change: run the cases of interest 30 times
uv run pytest tests/quality -k "taxes-03" --qt-runs 30 --report-log local.jsonl
# in qt-det
uv run qt-det case local.jsonl            # downloads the published results
uv run qt-det case local.jsonl --results path/to/qt-results
```

```
test_answer[taxes-03]
  main:   20/20 passed (100.0%, 95% CI 84%-100%)  from the last 1 main states
  local:  17/30 passed (56.7%, 95% CI 39%-73%)
  REGRESSION  (p = 0.00034, one-sided Fisher exact)
```

- Verdicts: `REGRESSION`, `IMPROVEMENT`, `no evidence of change` (with the number
  of local runs that would detect a `--delta` drop), `test definition changed`,
  `new test`. Several cases at once are corrected for multiple testing (Holm).
- The baseline pools the newest `--states` measured main states.
- Exit code 1 when any case regressed.

## HTML report

One self-contained HTML file (no server, works offline, light and dark mode):

```sh
uv run qt-det report --fetch --out report.html                    # published results
uv run qt-det report --results path/to/qt-results --out report.html
uv run qt-det report --fetch --logs-github shangzhihao/qt-app   # + failure messages
```

Sections: overview tiles; pass rate over main states with each verdict; topics ×
states heatmap (pass rate or change, flagged cells marked); alerts with where they
are and their decision history; pull requests; what every registered gate would
have said on each comparison (including testimate's two gates); a searchable
test-case explorer with each case's history, definition changes and failure
messages; runs; the decision log. Every chart has a table view.

Failure messages need raw logs: `--logs DIR` (a runs folder) or `--logs-github
OWNER/REPO` (Actions artifacts while they last, newest `--logs-runs` runs; needs
`QT_PAT` or `GITHUB_TOKEN`).

All numbers are computed in Python (`html_report/model.py`); the page only draws,
sorts and searches, so an interactive front end can reuse the same model later.

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
| `likelihood_ratio` | For 1-run comparisons: `zscore_strata`'s scopes, but each case weighted by its likelihood ratio for a drop of `delta` in its own pass rate, so a case that nearly always passes counts more when it fails. Flaky cases keep the plain count. |
| `ci_gate` | The PR check without re-runs. `likelihood_ratio`'s weights tested against a margin (drops under `margin` points pass), blocked at `alpha` and warned at `warn_alpha`; reliable cases without a margin (a few breaking is never forgiven); reliable and flaky cases chosen by the baseline alone; new cases (no baseline) tested together against `new_case_rate`. Only reports. |

`gate = "<name>"` in `analyzer.toml` (`[analysis]`) selects the gate that makes
the decisions — `zscore_strata` in this repository's configuration;
`[gates.<name>]` holds its settings, and `[gates.<name>.push]`,
`[gates.<name>.nightly]` and `[gates.<name>.pr]` override them per stage (a
1-run PR comparison and a 10-run nightly need different thresholds). The
values in `analyzer.toml` were tuned with `gates --benchmark
--equal-false-alarms` on the real suite's pass-rate distribution; a changed
gate or setting only affects states decided afterwards.
`[gates.<name>.reference]` is the loose check that decides whether the
reference may move to a nightly state (no scope may hint at a drop at that
level).

Real nightly runs vary together from night to night (shared infrastructure,
model behaviour), more than the binomial variance of independent runs allows.
For a gate with a `dispersion` setting (`zscore_strata`), the analysis
estimates this from the history itself on every pass: for each kind of scope
(overall, reliable cases, flaky cases, topics), φ = median(z²) / 0.4549 over
the step comparisons between consecutive nightly states (push and PR stages:
over the 1-run comparisons), at least 1, and divides each z by √φ. The median
keeps real changes and outages from inflating φ. The values are recorded with
each nightly decision and shown in the report. Reliable and flaky cases are
chosen by their pass rate over both sides of a comparison, not by the
baseline alone, which would bias the test.

```sh
uv run qt-det gates --list
uv run qt-det gates --example all          # two fixed examples, every gate
uv run qt-det gates --benchmark            # 7 scenarios, random trials
uv run qt-det gates --benchmark --gates zscore,poisson_binomial --runs 10
uv run qt-det gates --benchmark --equal-false-alarms 0.05   # fair: same false-alarm rate
```

A gate can implement `tuned(level)` (the same gate at another sensitivity) so
`--equal-false-alarms` can compare it with the others at the same false-alarm
rate; gates without it are compared as configured.

### Adding a gate

```python
# my_gates.py
from dataclasses import dataclass
from qt_det.gates import Verdict, register_gate


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
gate_modules = ["my_gates"]   # importable from where qt-det runs
[gates.my_gate]
threshold = 0.05
```

Every registered gate then appears in `gates --example` and `--benchmark`, and
`gate = "my_gate"` makes it decide the analysis.

## Locators: where is the change?

A gate answers whether quality changed; a locator says where. When the gate
flags a regression or improvement, the configured locator (`locator = "ranked"`
in `analyzer.toml`, settings in `[locators.<name>]`) ranks the scopes (topics,
reliable cases, flaky cases) and then the most affected test cases. The result
is recorded with the alert, shown in PR comments ("**Where:** …") and in
`status`. Locators register like gates (`@register_locator("name")`).

```sh
uv run qt-det gates --locate --runs 1  --baseline-runs 40   # PR-like
uv run qt-det gates --locate --runs 10 --baseline-runs 10   # nightly-like
```

On the real suite's distribution, whenever the gate fires, `ranked` puts the
right topic first in essentially every trial, and its top-10 cases are 98–99 %
right for a topic drop. A few collapsed cases among 1,934 need the nightly's
runs (98 % found in the top 10, vs 80 % from one run); `qt-det case`
confirms a single suspect locally.

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
