"""Dependency fingerprints for reusable training artifacts.

This module records the external files and feature ordering used to train a
model package. The resulting dictionary can be stored in a training manifest
and compared before existing checkpoints are reused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def calculate_file_sha256(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a SHA-256 fingerprint without loading the full file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot fingerprint missing file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_training_dependencies(
    *,
    project_root: str | Path,
    feature_scaler_path: str | Path,
    production_manifest_path: str | Path,
    feature_columns: Sequence[str],
) -> dict[str, object]:
    """Return fingerprints for dependencies used by a training package."""

    root = Path(project_root).resolve()
    scaler_path = Path(feature_scaler_path).resolve()
    manifest_path = Path(production_manifest_path).resolve()

    try:
        scaler_relative_path = scaler_path.relative_to(root)
        manifest_relative_path = manifest_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Dependency files must be located inside PROJECT_ROOT."
        ) from exc

    columns = [str(column) for column in feature_columns]
    if not columns:
        raise ValueError("feature_columns cannot be empty.")

    return {
        "feature_scaler": {
            "path": scaler_relative_path.as_posix(),
            "sha256": calculate_file_sha256(scaler_path),
        },
        "production_manifest": {
            "path": manifest_relative_path.as_posix(),
            "sha256": calculate_file_sha256(manifest_path),
        },
        "feature_columns": columns,
    }


__all__ = [
    "build_training_dependencies",
    "calculate_file_sha256",
]
