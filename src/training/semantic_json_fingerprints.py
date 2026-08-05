"""Portable semantic fingerprints and safe legacy-manifest migration.

The helpers in this module separate JSON formatting from semantic content.
They also support a narrow, one-time migration from legacy raw-file hashes
to canonical JSON hashes without touching trained model weights.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.training.artifact_management import (
    canonicalize_json_value,
    write_training_manifest,
)


CANONICAL_JSON_FINGERPRINT_METHOD = "canonical_json_sha256_v1"


def calculate_canonical_json_sha256(path: str | Path) -> str:
    """Hash canonical JSON content rather than raw file bytes."""

    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    payload = json.loads(
        json_path.read_text(encoding="utf-8")
    )
    canonical_payload = canonicalize_json_value(payload)
    canonical_text = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(
        canonical_text.encode("utf-8")
    ).hexdigest()


def set_canonical_json_dependency(
    dependencies: Mapping[str, Any],
    *,
    dependency_key: str,
    path: str | Path,
) -> dict[str, Any]:
    """Return a copy with a canonical JSON fingerprint at one top-level key."""

    result = json.loads(json.dumps(dependencies))
    dependency = result.get(dependency_key)
    if not isinstance(dependency, dict):
        raise KeyError(
            f"Dependency {dependency_key!r} must be a dictionary."
        )

    dependency["sha256"] = calculate_canonical_json_sha256(path)
    dependency["fingerprint_method"] = (
        CANONICAL_JSON_FINGERPRINT_METHOD
    )
    return result


def _value_at_path(
    value: Any,
    path: Sequence[str],
    *,
    missing: Any,
) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return missing
        current = current[key]
    return current


def _dependency_differences(
    saved: Any,
    expected: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    if isinstance(saved, Mapping) and isinstance(expected, Mapping):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(saved) | set(expected)):
            if key not in saved or key not in expected:
                differences.append((*path, str(key)))
                continue
            differences.extend(
                _dependency_differences(
                    saved[key],
                    expected[key],
                    (*path, str(key)),
                )
            )
        return differences

    if isinstance(saved, list) and isinstance(expected, list):
        if len(saved) != len(expected):
            return [path]
        differences: list[tuple[str, ...]] = []
        for index, (saved_item, expected_item) in enumerate(
            zip(saved, expected)
        ):
            differences.extend(
                _dependency_differences(
                    saved_item,
                    expected_item,
                    (*path, str(index)),
                )
            )
        return differences

    return [] if saved == expected else [path]


def migrate_legacy_json_fingerprint(
    manifest_path: str | Path,
    *,
    expected_notebook: str,
    expected_profile: str,
    expected_dependencies: Mapping[str, Any],
    required_paths: Sequence[str | Path],
    hash_path: Sequence[str],
    method_path: Sequence[str],
    enabled: bool,
) -> tuple[bool, str]:
    """Migrate one legacy raw JSON hash when it is the only incompatibility.

    Migration is intentionally narrow:
    - all required artifacts must exist and be non-empty;
    - the manifest must be complete and belong to the expected notebook/profile;
    - the saved fingerprint method must be absent;
    - the only dependency differences may be the legacy hash and new method;
    - only manifest metadata is rewritten.
    """

    if not enabled:
        return False, "migration disabled"

    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        return False, f"missing: {manifest_file.name}"

    missing_artifacts = [
        Path(path)
        for path in required_paths
        if not Path(path).is_file()
        or Path(path).stat().st_size == 0
    ]
    if missing_artifacts:
        return (
            False,
            "missing required artifacts: "
            + ", ".join(path.name for path in missing_artifacts),
        )

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

    saved_dependencies = manifest.get("dependencies")
    if not isinstance(saved_dependencies, Mapping):
        return False, "manifest has no dependency dictionary"

    missing = object()
    saved_method = _value_at_path(
        saved_dependencies,
        tuple(method_path),
        missing=missing,
    )
    if saved_method is not missing:
        return False, "manifest already uses an explicit fingerprint method"

    differences = set(
        _dependency_differences(
            canonicalize_json_value(saved_dependencies),
            canonicalize_json_value(expected_dependencies),
        )
    )
    allowed = {
        tuple(hash_path),
        tuple(method_path),
    }

    if not differences:
        return False, "manifest is already compatible"
    if not differences.issubset(allowed):
        formatted = [
            ".".join(path)
            for path in sorted(differences)
        ]
        return (
            False,
            "non-migratable dependency differences: "
            + ", ".join(formatted),
        )

    manifest["dependencies"] = canonicalize_json_value(
        expected_dependencies
    )
    manifest["dependency_fingerprint_migration"] = {
        "schema_version": 1,
        "from": "raw_file_sha256",
        "to": CANONICAL_JSON_FINGERPRINT_METHOD,
        "weights_modified": False,
    }
    write_training_manifest(manifest_file, manifest)
    return True, "legacy JSON fingerprint metadata migrated"


__all__ = [
    "CANONICAL_JSON_FINGERPRINT_METHOD",
    "calculate_canonical_json_sha256",
    "migrate_legacy_json_fingerprint",
    "set_canonical_json_dependency",
]
