from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

_PRODUCTS: Dict[str, Dict[str, Any]] = {}


def register_product(name: str, spec: Dict[str, Any]) -> None:
    _PRODUCTS[str(name)] = deepcopy(spec)


def get_product(name: str) -> Dict[str, Any]:
    if name not in _PRODUCTS:
        raise KeyError(f"Unknown product: {name}")
    return deepcopy(_PRODUCTS[name])


def list_products() -> Dict[str, Dict[str, Any]]:
    return deepcopy(_PRODUCTS)
