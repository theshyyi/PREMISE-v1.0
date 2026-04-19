from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    if suffix in {'.yaml', '.yml'}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError('YAML config requires pyyaml. Install with `pip install pyyaml`.') from e
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    raise ValueError(f'Unsupported config format: {path}')


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def save_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def now_tag() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def normalize_bbox(bbox: tuple[float, float, float, float] | list[float] | dict[str, float] | None):
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        keys = {'min_lon', 'max_lon', 'min_lat', 'max_lat'}
        if not keys.issubset(set(bbox)):
            raise KeyError(f'bbox dict must contain {keys}')
        return {k: float(bbox[k]) for k in ['min_lon', 'max_lon', 'min_lat', 'max_lat']}
    if len(bbox) != 4:
        raise ValueError('bbox must have 4 values: (min_lon, min_lat, max_lon, max_lat) or dict form')
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        'min_lon': float(min_lon),
        'max_lon': float(max_lon),
        'min_lat': float(min_lat),
        'max_lat': float(max_lat),
    }
