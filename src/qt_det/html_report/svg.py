"""A small SVG chart of the nightly pass rate per main state, for the README.

`analyse` writes it next to the results (quality.svg on the results branch), so
the README always shows the latest analysis. Colours follow the HTML report.
"""

from __future__ import annotations

import math
from html import escape

from qt_det.analyse import Analysis
from qt_det.html_report.model import _alerts, _short, _state_rows

FILE = "quality.svg"
MAX_STATES = 30
WIDTH, HEIGHT = 800, 300
LEFT, RIGHT, TOP, BOTTOM = 48, 24, 56, 40
SUMMARY_SCOPES = ("overall", "reliable cases", "flaky cases")

STYLE = """
.bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .muted { fill: #898781; }
.grid { stroke: #e1e0d9; } .line { stroke: #2a78d6; } .dot { fill: #2a78d6; stroke: #fcfcfb; }
.ref { stroke: #52514e; } .alert { fill: #d03b3b; } .good { fill: #0ca30c; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .grid { stroke: #2c2c2a; }
  .line { stroke: #3987e5; } .dot { fill: #3987e5; stroke: #1a1a19; } .ref { stroke: #c3c2b7; }
}
text { font: 12px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.title { font-size: 14px; font-weight: 600; }
"""


def _scope(scopes: list[str]) -> str:
    """The most specific scope: a topic if there is one."""
    topics = [s for s in scopes if s not in SUMMARY_SCOPES]
    return (topics or scopes or [""])[0]


def quality_svg(analysis: Analysis, generated_at: str) -> str:
    rows = [r for r in _state_rows(analysis) if r["nightly_rate"] is not None][-MAX_STATES:]
    alerts = {a["state"]: a for a in _alerts(analysis)}
    reference = _short(analysis.reference)
    ref_rate = next((r["nightly_rate"] for r in rows if r["short"] == reference), None)
    open_alerts = sum(a["open"] for a in alerts.values())
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Nightly pass rate per main state">',
        f"<style>{STYLE}</style>",
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}" rx="6"/>',
        f'<text class="ink title" x="{LEFT}" y="22">Answer quality: nightly pass rate per main '
        "state</text>",
        f'<text class="muted" x="{LEFT}" y="40">Reference {reference or "none"} (ring, dashed '
        f"level) · open alerts {open_alerts} · updated {escape(generated_at)}</text>",
    ]
    if not rows:
        out.append(f'<text class="muted" x="{LEFT}" y="{TOP + 40}">No nightly runs yet.</text>')
        return "\n".join([*out, "</svg>"])

    rates = [r["nightly_rate"] for r in rows]
    low = math.floor((min(rates) - 0.01) * 20) / 20
    high = math.ceil((max(rates) + 0.01) * 20) / 20
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(i: int) -> float:
        return LEFT + (plot_w / 2 if len(rows) == 1 else plot_w * i / (len(rows) - 1))

    def y(rate: float) -> float:
        return TOP + plot_h * (high - rate) / (high - low)

    tick = low
    while tick <= high + 1e-9:
        out.append(f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y(tick):.1f}" '
                   f'y2="{y(tick):.1f}" stroke-width="1"/>')  # fmt: skip
        out.append(f'<text class="muted" x="{LEFT - 8}" y="{y(tick) + 4:.1f}" '
                   f'text-anchor="end">{tick:.0%}</text>')  # fmt: skip
        tick += 0.05
    day = ""
    for i, r in enumerate(rows):
        if r["committed_at"][:10] != day:
            day = r["committed_at"][:10]
            out.append(f'<text class="muted" x="{x(i):.1f}" y="{HEIGHT - 14}" '
                       f'text-anchor="middle">{day[5:]}</text>')  # fmt: skip

    if ref_rate is not None:
        start = next(i for i, r in enumerate(rows) if r["short"] == reference)
        out.append(f'<line class="ref" x1="{x(start):.1f}" x2="{WIDTH - RIGHT}" '
                   f'y1="{y(ref_rate):.1f}" y2="{y(ref_rate):.1f}" stroke-width="1" '
                   'stroke-dasharray="4 4"/>')  # fmt: skip
        out.append(f'<circle class="ref" cx="{x(start):.1f}" cy="{y(ref_rate):.1f}" r="8" '
                   'fill="none" stroke-width="1.5"/>')  # fmt: skip

    # A changed case set breaks the line: rates on either side are not comparable.
    segment: list[str] = []
    for i, r in enumerate(rows):
        if (r["nightly"] or {}).get("status") == "not-comparable" and segment:
            out.append(_polyline(segment))
            segment = []
            gap = (x(i - 1) + x(i)) / 2
            out.append(
                f'<line class="ref" x1="{gap:.1f}" x2="{gap:.1f}" y1="{TOP}" '
                f'y2="{HEIGHT - BOTTOM}" stroke-width="1" stroke-dasharray="2 4"/>'
            )
            out.append(f'<text class="muted" x="{gap - 6:.1f}" y="{TOP + 12}" '
                       'text-anchor="end">cases changed</text>')  # fmt: skip
        segment.append(f"{x(i):.1f},{y(r['nightly_rate']):.1f}")
    out.append(_polyline(segment))

    ys = [y(r["nightly_rate"]) for r in rows]
    for i, r in enumerate(rows):
        cx, cy = x(i), ys[i]
        # Labels go beyond the point and both neighbours, clear of the two segments.
        around = ys[max(i - 1, 0) : i + 2]
        alert = alerts.get(r["short"])
        improved = (r["nightly"] or {}).get("step_improved") or []
        tip = f"{r['short']} ({r['created_by']} {r['sha']}): {r['nightly_rate']:.1%}"
        out.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4" stroke-width="2">'
                   f"<title>{escape(tip)}</title></circle>")  # fmt: skip
        if alert:
            status = "open" if alert["open"] else "closed"
            out.append(_label(cx, max(around) + 20, "alert", "▼",
                              f"{_scope(alert['scopes'])} ({status})"))  # fmt: skip
        elif improved:
            out.append(_label(cx, min(around) - 12, "good", "▲", _scope(improved)))
    return "\n".join([*out, "</svg>"])


def _polyline(points: list[str]) -> str:
    return (f'<polyline class="line" points="{" ".join(points)}" fill="none" stroke-width="2" '
            'stroke-linejoin="round"/>')  # fmt: skip


def _label(x: float, y: float, marker: str, glyph: str, text: str) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle"><tspan class="{marker}">'
            f'{glyph}</tspan> <tspan class="ink">{escape(text)}</tspan></text>')  # fmt: skip
