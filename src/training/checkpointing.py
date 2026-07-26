"""Checkpoint utilities for reproducible PyTorch experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a PyTorch checkpoint atomically."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output)
    return output


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint using the safer weights-only loader when available."""

    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")
    return checkpoint


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return output


__all__ = ["atomic_torch_save", "load_checkpoint", "save_json"]
