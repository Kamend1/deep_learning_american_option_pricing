"""Shared management of reusable notebook training artifacts.

The helpers in this module validate completion manifests, compare external
dependencies, migrate legacy manifests, and write completion markers atomically.
They do not depend on notebook globals.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def canonicalize_json_value(value: Any) -> Any:
    """Recursively normalize common Python and NumPy values for JSON use."""

    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            canonicalize_json_value(item)
            for item in value
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return canonicalize_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def inspect_training_artifacts(
    manifest_path: str | Path,
    required_paths: Sequence[str | Path],
    *,
    expected_notebook: str,
    expected_profile: str,
    expected_dependencies: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return whether a complete and compatible training package is available."""

    manifest_file = Path(manifest_path)
    required_files = [Path(path) for path in required_paths]

    missing = [
        path
        for path in required_files
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return (
            False,
            "missing: " + ", ".join(path.name for path in missing),
        )

    if not manifest_file.is_file():
        return False, f"missing: {manifest_file.name}"

    try:
        manifest = json.loads(
            manifest_file.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"

    if manifest.get("status") != "complete":
        return False, "manifest status is not complete"
    if manifest.get("notebook") != expected_notebook:
        return False, "manifest belongs to another notebook"
    if manifest.get("training_profile") != expected_profile:
        return False, "training profile does not match"

    if expected_dependencies is not None:
        saved_dependencies = manifest.get("dependencies")
        if saved_dependencies is None:
            return (
                False,
                "training manifest has no dependency fingerprints",
            )

        if (
            canonicalize_json_value(saved_dependencies)
            != canonicalize_json_value(expected_dependencies)
        ):
            return False, "training dependencies do not match"

    return True, "complete compatible package found"


def write_training_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write a completion manifest atomically and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            canonicalize_json_value(payload),
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def backfill_manifest_dependencies(
    manifest_path: str | Path,
    dependencies: Mapping[str, Any],
    *,
    enabled: bool,
    force_train: bool,
) -> bool:
    """Add dependencies to a complete legacy manifest without retraining."""

    manifest_file = Path(manifest_path)

    if (
        not enabled
        or force_train
        or not manifest_file.is_file()
    ):
        return False

    try:
        manifest = json.loads(
            manifest_file.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    if (
        manifest.get("status") != "complete"
        or "dependencies" in manifest
    ):
        return False

    manifest["dependencies"] = canonicalize_json_value(
        dependencies
    )
    manifest["dependency_metadata_backfilled"] = True
    write_training_manifest(manifest_file, manifest)
    return True


__all__ = [
    "backfill_manifest_dependencies",
    "canonicalize_json_value",
    "inspect_training_artifacts",
    "write_training_manifest",
]
