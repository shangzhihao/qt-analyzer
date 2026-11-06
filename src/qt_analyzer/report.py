"""Human-facing output: PR comments (and later issues and the summary SVG)."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from qt_analyzer.analyse import PrResult
from qt_analyzer.stats import ScopeTest
from qt_analyzer.stores.results import ResultStore

MARKER = "<!-- qt-report -->"
API = "https://api.github.com"
_BOOKKEEPING = "comments/prs.json"


def _pct(x: float) -> str:
    return "-" if x != x else f"{x:.1%}"


def _row(name: str, t: ScopeTest) -> str:
    return (
        f"| {name} | {t.cases} | {_pct(t.rate_b)} | {_pct(t.rate_s)} | "
        f"{t.delta * 100:+.1f} pts | {t.z:+.2f} |"
    )


def pr_comment(pr: PrResult) -> str:
    lines = [MARKER, "### Answer quality", ""]
    if pr.status == "baseline-pending":
        lines += [
            "⏳ **Baseline pending.** The `main` state this PR is based on has not been "
            "measured yet; this comment is updated once it is.",
        ]
    else:
        assert pr.comparison is not None
        c = pr.comparison
        verdict = (
            f"⚠️ **Possible regression** in {', '.join(f'`{s}`' for s in pr.flagged)}."
            if pr.flagged
            else "✅ **No significant drop** against `main`."
        )
        lines += [
            verdict,
            "",
            f"1 run per case, compared with recent `main` runs. Overall "
            f"{_pct(c.overall.rate_b)} → {_pct(c.overall.rate_s)} "
            f"({c.overall.delta * 100:+.1f} pts).",
            "",
            "| scope | cases | main | PR | change | z |",
            "|---|---:|---:|---:|---:|---:|",
            _row("overall", c.overall),
        ]
        changed = [(f, t) for f, t in c.families.items() if f in pr.flagged or f in pr.improved]
        lines += [_row(f"`{f}`", t) for f, t in changed]
        if pr.improved:
            lines += ["", f"Improved: {', '.join(f'`{s}`' for s in pr.improved)}."]
        if pr.suspected_cases:
            labels = {case.case_key: case.hrid for case in c.cases}
            shown = [f"`{labels.get(k, k)}`" for k in pr.suspected_cases[:10]]
            more = len(pr.suspected_cases) - len(shown)
            lines += [
                "",
                f"Suspected cases (one run, not confirmed): {', '.join(shown)}"
                + (f" and {more} more" if more > 0 else "")
                + ".",
            ]
        cov = c.coverage
        if cov.added or cov.removed or cov.edited:
            lines += [
                "",
                f"Coverage: {len(cov.added)} added, {len(cov.removed)} removed, "
                f"{len(cov.edited)} edited cases (not compared).",
            ]
    lines += ["", f"<sub>Run `{pr.run_id}` for `{pr.head_sha[:7]}`"]
    if pr.run_url:
        lines[-1] += f" · [logs]({pr.run_url})"
    lines[-1] += " · reports only, never blocks the merge</sub>"
    return "\n".join(lines) + "\n"


Request = Callable[[str, str, dict[str, Any] | None], Any]


def github_request(token: str) -> Request:
    def request(method: str, url: str, body: dict[str, Any] | None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
            return json.loads(payload) if payload else None

    return request


def sync_pr_comments(results: ResultStore, prs: list[PrResult], request: Request) -> list[str]:
    """Create or update one comment per PR when its newest result changed."""
    done: dict[str, dict[str, Any]] = results.read_json(_BOOKKEEPING) or {}
    updated = []
    for pr in prs:
        key = f"{pr.repo}#{pr.number}"
        posted = done.get(key, {})
        version = f"{pr.run_id}:{pr.status}"
        if posted.get("version") == version:
            continue
        body = {"body": pr_comment(pr)}
        comment_id = posted.get("comment_id")
        if comment_id:
            request("PATCH", f"{API}/repos/{pr.repo}/issues/comments/{comment_id}", body)
        else:
            created = request("POST", f"{API}/repos/{pr.repo}/issues/{pr.number}/comments", body)
            comment_id = created["id"]
        done[key] = {"comment_id": comment_id, "version": version}
        results.write_json(_BOOKKEEPING, done)
        updated.append(key)
    return updated
