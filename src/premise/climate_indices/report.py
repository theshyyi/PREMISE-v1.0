from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_markdown(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Climate Indices Report", "", f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    lines += [f"- Total tasks: {len(records)}", f"- Successful tasks: {sum(r['status']=='SUCCESS' for r in records)}", f"- Failed tasks: {sum(r['status']=='FAILED' for r in records)}", ""]
    lines += ["| Name | Family | Status | Output | Time (s) | Variables |", "|---|---|---|---|---:|---|"]
    for r in records:
        vars_text = ", ".join(r.get("variables", []) or [])
        lines.append(f"| {r.get('name','')} | {r.get('family','')} | {r.get('status','')} | {r.get('output','')} | {r.get('elapsed_seconds','')} | {vars_text} |")
    lines.append("")
    for r in records:
        lines.append(f"## {r.get('name','')}")
        lines.append("")
        for key in ["family", "input", "output", "status", "elapsed_seconds", "time_range", "note", "error_type", "error_message"]:
            lines.append(f"- {key}: {r.get(key)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
