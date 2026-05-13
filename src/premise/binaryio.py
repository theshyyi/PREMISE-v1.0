"""Backward-compatible helpers for simple binary metadata files.

The main conversion functionality is implemented in ``premise.conversion``.
This module keeps the legacy ``premise.binaryio`` import path functional for
existing scripts and regression tests.
"""

from __future__ import annotations

from pathlib import Path


def parse_meta_file(meta_path: str | Path) -> dict[str, str]:
    """Parse a simple key-value metadata file.

    Parameters
    ----------
    meta_path:
        Path to a text file containing one ``key=value`` or ``key: value`` pair
        per line. Empty lines and lines beginning with ``#`` are ignored.

    Returns
    -------
    dict[str, str]
        Parsed metadata entries as strings.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    result: dict[str, str] = {}
    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        result[key.strip()] = value.strip()
    return result


__all__ = ["parse_meta_file"]
