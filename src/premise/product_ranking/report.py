from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd


def write_excel_book(path: str | Path, sheets: Mapping[str, pd.DataFrame]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            safe = str(name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    return path


def write_markdown_summary(
    path: str | Path,
    *,
    task_name: str,
    methods: Mapping[str, pd.DataFrame],
    selected_products: list[str],
    fusion_note: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f'# Product ranking report: {task_name}\n')
    lines.append('## Selected top products\n')
    for i, p in enumerate(selected_products, start=1):
        lines.append(f'{i}. {p}')
    lines.append('')
    lines.append('## Method leaders\n')
    for name, df in methods.items():
        if df.empty:
            continue
        row = df.iloc[0]
        prod_col = [c for c in df.columns if c not in {'rank'} and not c.endswith('_score') and not c.endswith('_rank')][0]
        lines.append(f'- {name}: top-1 = **{row[prod_col]}**')
    lines.append('')
    lines.append('## Fusion\n')
    lines.append(fusion_note)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path
