"""Lineage, package-coherence, and static prediction-alignment checks.

The final evaluation must establish that the static models are evaluated on the
same observations before their errors are compared.  This module performs that
validation without training or changing any upstream artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.artifact_registry import (
    assert_required_artifacts_valid,
    audit_artifacts,
)
from src.evaluation.final_artifact_adapters import (
    FinalNotebookPackage,
    build_package_summary,
    load_all_final_packages,
)


STATIC_NOTEBOOKS = ("04", "05", "06", "08", "08_scratch")
DEFAULT_ABSOLUTE_TOLERANCE = 1e-7
DEFAULT_RELATIVE_TOLERANCE = 1e-7
TARGET_COLUMNS = {
    "04": "normalized_american_price",
    "05": "normalized_american_price",
    "06": "normalized_american_price",
    "08": "true_normalized_american_price",
    "08_scratch": "true_normalized_american_price",
}
COMMON_STATE_COLUMNS = (
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
)


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training manifest must be a mapping: {path}")
    return payload


def audit_package_coherence(
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Validate final-package metadata against canonical files and manifests."""

    rows: list[dict[str, Any]] = []

    def add(notebook: str, check: str, valid: bool, details: str) -> None:
        rows.append(
            {
                "notebook": notebook,
                "check": check,
                "valid": bool(valid),
                "details": details,
            }
        )

    for notebook, package in sorted(packages.items()):
        add(
            notebook,
            "final_package_complete",
            package.status == "complete",
            f"status={package.status!r}",
        )
        add(
            notebook,
            "canonical_checkpoint_exists",
            package.checkpoint_path.is_file()
            and package.checkpoint_path.stat().st_size > 0,
            str(package.checkpoint_path),
        )
        add(
            notebook,
            "checkpoint_name_matches_file",
            package.checkpoint_name == package.checkpoint_path.name,
            (
                f"declared={package.checkpoint_name!r}; "
                f"resolved={package.checkpoint_path.name!r}"
            ),
        )

        for manifest_path in package.training_manifests:
            try:
                manifest = _load_manifest(manifest_path)
                status_valid = manifest.get("status") == "complete"
                profile_valid = (
                    str(manifest.get("training_profile"))
                    == package.training_profile
                )
                add(
                    notebook,
                    f"manifest_complete:{manifest_path.name}",
                    status_valid,
                    f"status={manifest.get('status')!r}",
                )
                add(
                    notebook,
                    f"manifest_profile_matches:{manifest_path.name}",
                    profile_valid,
                    (
                        f"manifest={manifest.get('training_profile')!r}; "
                        f"final={package.training_profile!r}"
                    ),
                )
            except Exception as exc:
                add(
                    notebook,
                    f"manifest_readable:{manifest_path.name}",
                    False,
                    repr(exc),
                )

        if package.test_predictions_path is not None:
            add(
                notebook,
                "test_predictions_exist",
                package.test_predictions_path.is_file(),
                str(package.test_predictions_path),
            )

    return pd.DataFrame(rows)


def _prepare_prediction_frame(
    frame: pd.DataFrame,
    notebook: str,
) -> pd.DataFrame:
    required = {"sample_id", TARGET_COLUMNS[notebook]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Notebook {notebook} predictions are missing columns: {missing}"
        )

    result = frame.copy()
    if result["sample_id"].isna().any():
        raise ValueError(f"Notebook {notebook} contains missing sample_id values")
    duplicate_count = int(result["sample_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Notebook {notebook} contains {duplicate_count} duplicate sample_id values"
        )

    result = result.rename(columns={TARGET_COLUMNS[notebook]: "_true_target"})
    return result.sort_values("sample_id").reset_index(drop=True)


def _prepare_prediction_table(
    package: FinalNotebookPackage,
    notebook: str,
) -> pd.DataFrame:
    return _prepare_prediction_frame(package.load_test_predictions(), notebook)


def audit_static_prediction_alignment(
    packages: Mapping[str, FinalNotebookPackage],
    *,
    reference_notebook: str = "04",
    numeric_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare sample IDs, targets, and shared state fields across static models."""

    required_packages = ("04", "05", "06", "08")
    missing_packages = [name for name in required_packages if name not in packages]
    if missing_packages:
        raise KeyError(f"Missing static notebook packages: {missing_packages}")

    tables = {
        notebook: _prepare_prediction_table(packages[notebook], notebook)
        for notebook in required_packages
    }
    tables["08_scratch"] = _prepare_prediction_frame(
        packages["08"].load_benchmark_test_predictions(),
        "08_scratch",
    )
    reference = tables[reference_notebook]
    reference_ids = pd.Index(reference["sample_id"])

    summary_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for notebook in STATIC_NOTEBOOKS:
        table = tables[notebook]
        ids = pd.Index(table["sample_id"])
        missing_ids = reference_ids.difference(ids)
        extra_ids = ids.difference(reference_ids)
        same_ids = len(missing_ids) == 0 and len(extra_ids) == 0

        target_max_abs_diff = float("nan")
        target_matches = False
        if same_ids:
            aligned = reference[["sample_id", "_true_target"]].merge(
                table[["sample_id", "_true_target"]],
                on="sample_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_reference", "_candidate"),
            )
            differences = np.abs(
                pd.to_numeric(
                    aligned["_true_target_reference"], errors="coerce"
                ).to_numpy(dtype=float)
                - pd.to_numeric(
                    aligned["_true_target_candidate"], errors="coerce"
                ).to_numpy(dtype=float)
            )
            target_max_abs_diff = (
                float(np.nanmax(differences)) if len(differences) else 0.0
            )
            reference_target = pd.to_numeric(
                aligned["_true_target_reference"], errors="coerce"
            ).to_numpy(dtype=float)
            candidate_target = pd.to_numeric(
                aligned["_true_target_candidate"], errors="coerce"
            ).to_numpy(dtype=float)
            target_matches = bool(
                np.isfinite(reference_target).all()
                and np.isfinite(candidate_target).all()
                and np.allclose(
                    reference_target,
                    candidate_target,
                    rtol=relative_tolerance,
                    atol=numeric_tolerance,
                    equal_nan=False,
                )
            )

        summary_rows.append(
            {
                "notebook": notebook,
                "reference_notebook": reference_notebook,
                "observations": int(len(table)),
                "reference_observations": int(len(reference)),
                "duplicate_sample_ids": int(table["sample_id"].duplicated().sum()),
                "missing_reference_ids": int(len(missing_ids)),
                "extra_candidate_ids": int(len(extra_ids)),
                "same_sample_id_set": same_ids,
                "target_max_absolute_difference": target_max_abs_diff,
                "same_true_target": target_matches,
                "valid": bool(same_ids and target_matches),
            }
        )

        if not same_ids:
            continue

        shared_columns = [
            column
            for column in COMMON_STATE_COLUMNS
            if column in reference.columns and column in table.columns
        ]
        if not shared_columns:
            field_rows.append(
                {
                    "notebook": notebook,
                    "field": "<no shared state fields exported>",
                    "observations": int(len(table)),
                    "max_absolute_difference": np.nan,
                    "matches": True,
                    "note": "sample_id and target alignment remain authoritative",
                }
            )
            continue

        aligned = reference[["sample_id", *shared_columns]].merge(
            table[["sample_id", *shared_columns]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_candidate"),
        )
        for column in shared_columns:
            left = pd.to_numeric(
                aligned[f"{column}_reference"], errors="coerce"
            ).to_numpy(dtype=float)
            right = pd.to_numeric(
                aligned[f"{column}_candidate"], errors="coerce"
            ).to_numpy(dtype=float)
            differences = np.abs(left - right)
            maximum = float(np.nanmax(differences)) if len(differences) else 0.0
            field_rows.append(
                {
                    "notebook": notebook,
                    "field": column,
                    "observations": int(len(aligned)),
                    "max_absolute_difference": maximum,
                    "matches": bool(
                        np.isfinite(left).all()
                        and np.isfinite(right).all()
                        and np.allclose(
                            left,
                            right,
                            rtol=relative_tolerance,
                            atol=numeric_tolerance,
                            equal_nan=False,
                        )
                    ),
                    "note": (
                        f"rtol={relative_tolerance:g}; "
                        f"atol={numeric_tolerance:g}"
                    ),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(field_rows)


def _find_dependency_fingerprints(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and ("sha256" in value or "semantic_sha256" in value):
            rows.append(
                {
                    "dependency_path": ".".join(path),
                    "file_path": value.get("path"),
                    "sha256": value.get("sha256", value.get("semantic_sha256")),
                    "fingerprint_method": value.get("fingerprint_method", "unspecified"),
                }
            )
        for key, item in value.items():
            rows.extend(_find_dependency_fingerprints(item, (*path, str(key))))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.extend(_find_dependency_fingerprints(item, (*path, str(index))))
    return rows


def build_lineage_inventory(
    project_root: Path,
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Inventory saved dependency fingerprints without equating unlike methods."""

    root = Path(project_root).resolve()
    current_manifest = root / "data/manifests/production_dataset_manifest.json"
    current_semantic_hash = (
        _canonical_json_hash(current_manifest) if current_manifest.is_file() else None
    )

    rows: list[dict[str, Any]] = []
    for notebook, package in sorted(packages.items()):
        sources = [("final_metrics", package.final_metrics)]
        for manifest_path in package.training_manifests:
            try:
                sources.append((manifest_path.name, _load_manifest(manifest_path)))
            except Exception:
                continue
        for source_name, payload in sources:
            for record in _find_dependency_fingerprints(payload):
                rows.append(
                    {
                        "notebook": notebook,
                        "source": source_name,
                        **record,
                        "current_production_manifest_semantic_sha256": (
                            current_semantic_hash
                            if str(record.get("file_path", "")).endswith(
                                "production_dataset_manifest.json"
                            )
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_phase_1_3_audit(project_root: Path) -> dict[str, Any]:
    """Run the complete Phase 1-3 validation package."""

    root = Path(project_root).resolve()
    artifact_audit = audit_artifacts(root)
    assert_required_artifacts_valid(artifact_audit)

    packages = load_all_final_packages(root)
    package_summary = build_package_summary(packages)
    package_coherence = audit_package_coherence(packages)
    static_alignment, static_field_alignment = audit_static_prediction_alignment(
        packages
    )
    lineage_inventory = build_lineage_inventory(root, packages)

    return {
        "packages": packages,
        "artifact_audit": artifact_audit,
        "package_summary": package_summary,
        "package_coherence": package_coherence,
        "static_prediction_alignment": static_alignment,
        "static_field_alignment": static_field_alignment,
        "lineage_inventory": lineage_inventory,
    }


def assert_phase_1_3_ready(results: Mapping[str, Any]) -> None:
    """Raise when package coherence or static alignment is not clean."""

    coherence = results["package_coherence"]
    invalid_coherence = coherence.loc[~coherence["valid"]]
    if not invalid_coherence.empty:
        details = "; ".join(
            f"NB{row.notebook} {row.check}: {row.details}"
            for row in invalid_coherence.itertuples(index=False)
        )
        raise RuntimeError(f"Final package coherence failed: {details}")

    alignment = results["static_prediction_alignment"]
    invalid_alignment = alignment.loc[~alignment["valid"]]
    if not invalid_alignment.empty:
        details = "; ".join(
            (
                f"NB{row.notebook}: missing={row.missing_reference_ids}, "
                f"extra={row.extra_candidate_ids}, "
                f"target_diff={row.target_max_absolute_difference}"
            )
            for row in invalid_alignment.itertuples(index=False)
        )
        raise RuntimeError(f"Static prediction alignment failed: {details}")

    field_alignment = results["static_field_alignment"]
    if not field_alignment.empty:
        invalid_fields = field_alignment.loc[~field_alignment["matches"]]
        if not invalid_fields.empty:
            details = "; ".join(
                f"NB{row.notebook} {row.field}: {row.max_absolute_difference}"
                for row in invalid_fields.itertuples(index=False)
            )
            raise RuntimeError(f"Static state-field alignment failed: {details}")


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "STATIC_NOTEBOOKS",
    "TARGET_COLUMNS",
    "assert_phase_1_3_ready",
    "audit_package_coherence",
    "audit_static_prediction_alignment",
    "build_lineage_inventory",
    "run_phase_1_3_audit",
]
