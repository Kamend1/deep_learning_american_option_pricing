"""Shared JSON export helpers for notebook canonical saves.

The helpers convert pandas, NumPy, pathlib, and non-finite floating-point
objects into values that can be serialized consistently by the standard
library ``json`` module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _export_json_safe(value: Any) -> Any:
    """Recursively convert common project objects into JSON-safe values."""
    if isinstance(value, Mapping):
        return {
            str(key): _export_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, pd.DataFrame):
        return _export_json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _export_json_safe(value.to_dict())
    if isinstance(value, pd.Index):
        return [_export_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_export_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_export_json(path: str | Path, payload: Any) -> Path:
    """Write a canonical, deterministic JSON export and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _export_json_safe(payload),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "_export_json_safe",
    "_write_export_json",
]
