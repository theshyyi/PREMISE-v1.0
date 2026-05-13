from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_metric_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f'Unsupported table format: {suffix}')
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df.to_csv(path, index=False, encoding='utf-8-sig')
    elif suffix in {'.xlsx', '.xls'}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f'Unsupported output format: {suffix}')
    return path


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')
