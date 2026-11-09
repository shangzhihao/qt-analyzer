"""Human-facing output: PR comments (and later issues and the summary SVG)."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from qt_det.analyse import PrResult
from qt_det.locators import format_location
from qt_det.stats import ScopeTest

MARKER = "<!-- qt-report -->"
API = "https://api.github.com"


def _pct(x: float) -> str:
    return "-" if x != x else f"{x:.1%}"


def _row(name: str, t: ScopeTest) -> str:
    return (
        f"| {name} | {t.cases} | {_pct(t.rate_b)} | {_pct(t.rate_s)} | "
        f"{t.delta * 100:+.1f} pts | {t.z:+.2f} |"
    )


def pr_comment(pr: PrResult) -> str:
    lines = [MARKER, "### Answer quality", ""]
    if pr.status == "not-comparable":
        lines += [
            f"⚪ **Not comparable.** {pr.reason[:1].upper() + pr.reason[1:]}: the tests "
            "changed, so this PR's quality could not be checked against `main`.",
        ]
    elif pr.status == "baseline-pending":
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
        ]
        if pr.nearest_baseline:
            lines += [
                "The `main` state this PR is based on has not been measured; compared "
                "with the nearest measured `main` state instead.",
                "",
            ]
        lines += [
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
        if pr.location is not None:
            lines += [""] + [f"**{line.split(': ', 1)[0].capitalize()}:** {line.split(': ', 1)[1]}"
                             for line in format_location(pr.location)]  # fmt: skip
        elif pr.suspected_cases:
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


def _version(pr: PrResult) -> str:
    return f"<!-- qt-version {pr.run_id} {pr.status} {int(pr.nearest_baseline)} -->"


def _paged(request: Request, url: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = request("GET", f"{url}?per_page=100&page={page}", None) or []
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def sync_pr_comments(prs: list[PrResult], request: Request) -> list[str]:
    """Create or update one comment per PR so it matches the newest result.

    The comment is found by its marker among the PR's own comments (a
    consistent listing, not search), and updated only when its hidden version
    (run, verdict, baseline kind) differs. Safe to repeat after a crash.
    """
    updated = []
    for pr in prs:
        url = f"{API}/repos/{pr.repo}/issues/{pr.number}/comments"
        mine = [c for c in _paged(request, url) if MARKER in (c.get("body") or "")]
        version = _version(pr)
        if mine and version in mine[0]["body"]:
            continue
        body = {"body": pr_comment(pr) + version + "\n"}
        if mine:
            request("PATCH", f"{API}/repos/{pr.repo}/issues/comments/{mine[0]['id']}", body)
        else:
            request("POST", url, body)
        updated.append(f"{pr.repo}#{pr.number}")
    return updated
