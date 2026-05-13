
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def write_summary_csv(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary_json(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_summary_md(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Product Evaluation Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Total tasks: {len(records)}",
        f"- Successful tasks: {sum(1 for r in records if r.get('status') == 'SUCCESS')}",
        f"- Failed tasks: {sum(1 for r in records if r.get('status') == 'FAILED')}",
        "",
        "| name | mode | status | time_scale | group_by | elapsed_seconds | output_main | error |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.get('name','')} | {r.get('mode','')} | {r.get('status','')} | "
            f"{r.get('time_scale','')} | {r.get('group_by','')} | {r.get('elapsed_seconds','')} | "
            f"{r.get('output_main','')} | {r.get('error_message','')} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
