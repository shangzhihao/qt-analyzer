"""Render the report model into one self-contained HTML file."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

PLACEHOLDER = "/*__MODEL__*/"


def render(model: dict[str, Any]) -> str:
    template = files("qt_det.html_report").joinpath("template.html").read_text(encoding="utf-8")
    data = json.dumps(model, ensure_ascii=False, allow_nan=False)
    # The data sits inside <script type="application/json">: keep "</" from closing it.
    data = data.replace("</", "<\\/")
    return template.replace(PLACEHOLDER, data, 1)
