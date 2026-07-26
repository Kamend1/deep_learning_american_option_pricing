"""Artifact discovery and validation for the final project evaluation.

The registry deliberately supports multiple candidate paths for each logical
artifact because earlier notebooks may save CSV, Parquet, or JSON variants.
Missing artifacts are reported as structured status records rather than causing
Notebook 09 to fail during the pre-execution skeleton phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd


ArtifactLoader = Literal["json", "csv", "parquet", "torch", "joblib", "directory", "any"]


@dataclass(frozen=True)
class ArtifactSpec:
    """Description of one logical project artifact."""

    name: str
    category: str
    candidate_paths: tuple[str, ...]
    required_for_final: bool = True
    loader: ArtifactLoader = "any"
    description: str = ""


@dataclass(frozen=True)
class ArtifactStatus:
    """Resolved state of one artifact specification."""

    name: str
    category: str
    required_for_final: bool
    found: bool
    valid: bool
    resolved_path: str | None
    loader: str
    notes: str


def default_artifact_registry() -> tuple[ArtifactSpec, ...]:
    """Return the canonical Step 9 artifact registry."""

    return (
        ArtifactSpec(
            "production_dataset_manifest",
            "data",
            (
                "data/manifests/production_dataset_manifest.json",
                "data/manifests/production_generation_manifest.json",
                "data/manifests/production_dataset_config.json",
            ),
            True,
            "json",
            "Production design, row counts, generation settings, and chunk metadata.",
        ),
        ArtifactSpec(
            "split_manifest",
            "data",
            (
                "data/manifests/pilot_split_manifest.json",
                "data/manifests/production_split_manifest.json",
                "data/manifests/dataset_design.json",
            ),
            True,
            "json",
            "Frozen train, validation, test, and out-of-domain split definitions.",
        ),
        ArtifactSpec(
            "direct_model_metrics",
            "static_model",
            (
                "artifacts/direct_mlp/evaluation_summary.json",
                "artifacts/direct_mlp/test_metrics.json",
            ),
            True,
            "json",
            "Notebook 04 direct MLP evaluation metrics.",
        ),
        ArtifactSpec(
            "direct_model_predictions",
            "static_model",
            (
                "artifacts/direct_mlp/test_predictions.parquet",
                "artifacts/direct_mlp/test_predictions.csv",
            ),
            True,
            "any",
            "Notebook 04 aligned test predictions.",
        ),
        ArtifactSpec(
            "premium_model_metrics",
            "static_model",
            (
                "artifacts/premium_models/evaluation_summary.json",
                "artifacts/premium_models/test_metrics.json",
            ),
            True,
            "json",
            "Notebook 05 premium and constrained residual metrics.",
        ),
        ArtifactSpec(
            "premium_model_comparison",
            "static_model",
            (
                "artifacts/premium_models/model_comparison.csv",
                "artifacts/premium_models/model_comparison.parquet",
            ),
            False,
            "any",
            "Notebook 05 ablation table.",
        ),
        ArtifactSpec(
            "multitask_metrics",
            "static_model",
            (
                "artifacts/multitask_model/evaluation_summary.json",
                "artifacts/multitask_model/classification_metrics.json",
                "artifacts/multitask_model/test_metrics.json",
            ),
            True,
            "json",
            "Notebook 06 price and exercise metrics.",
        ),
        ArtifactSpec(
            "multitask_boundary_results",
            "static_model",
            (
                "artifacts/multitask_model/boundary_curves.parquet",
                "artifacts/multitask_model/boundary_curves.csv",
                "artifacts/multitask_model/boundary_metrics.json",
            ),
            True,
            "any",
            "Notebook 06 exercise-boundary results.",
        ),
        ArtifactSpec(
            "lsm_evaluation_summary",
            "simulation_model",
            (
                "artifacts/neural_lsm/evaluation_summary.json",
                "artifacts/neural_lsm/runtime_summary.json",
            ),
            True,
            "json",
            "Notebook 07 classical and neural LSM summary.",
        ),
        ArtifactSpec(
            "lsm_pricing_results",
            "simulation_model",
            (
                "artifacts/neural_lsm/heldout_pricing_results.parquet",
                "artifacts/neural_lsm/heldout_pricing_results.csv",
            ),
            True,
            "any",
            "Notebook 07 held-out contract pricing results.",
        ),
        ArtifactSpec(
            "lsm_policy_metrics",
            "simulation_model",
            (
                "artifacts/neural_lsm/heldout_policy_metrics.parquet",
                "artifacts/neural_lsm/heldout_policy_metrics.csv",
            ),
            False,
            "any",
            "Notebook 07 stopping-policy comparison.",
        ),
        ArtifactSpec(
            "integrated_model_metrics",
            "static_model",
            (
                "artifacts/final_multihead/test_metrics.json",
                "artifacts/final_multihead/pricing_metrics.json",
            ),
            True,
            "json",
            "Notebook 08 final integrated model metrics.",
        ),
        ArtifactSpec(
            "integrated_model_predictions",
            "static_model",
            (
                "artifacts/final_multihead/test_predictions.parquet",
                "artifacts/final_multihead/test_predictions.csv",
            ),
            True,
            "any",
            "Notebook 08 aligned test predictions for all heads.",
        ),
        ArtifactSpec(
            "integrated_model_checkpoint",
            "checkpoint",
            (
                "artifacts/final_multihead/best_integrated_multihead.pt",
                "artifacts/final_multihead/best_balanced.pt",
            ),
            True,
            "torch",
            "Authoritative final static checkpoint.",
        ),
        ArtifactSpec(
            "ood_results",
            "evaluation",
            (
                "artifacts/final_multihead/ood_predictions.parquet",
                "artifacts/final_multihead/ood_predictions.csv",
                "artifacts/final_multihead/ood_metrics.json",
            ),
            True,
            "any",
            "Aligned out-of-domain results.",
        ),
        ArtifactSpec(
            "runtime_summary",
            "evaluation",
            (
                "artifacts/final_multihead/runtime_summary.json",
                "artifacts/final_evaluation/runtime_comparison.csv",
                "artifacts/neural_lsm/runtime_records.csv",
            ),
            True,
            "any",
            "Numerical and neural runtime records.",
        ),
        ArtifactSpec(
            "hypothesis_evidence",
            "final_evaluation",
            (
                "artifacts/final_evaluation/hypothesis_evidence.json",
                "artifacts/final_evaluation/final_results_summary.json",
            ),
            False,
            "json",
            "Normalized evidence used by the H1-H6 decision rules.",
        ),
    )


def resolve_artifact_path(project_root: Path, spec: ArtifactSpec) -> Path | None:
    """Resolve the first existing candidate path for an artifact."""

    root = Path(project_root).resolve()
    for candidate in spec.candidate_paths:
        path = root / candidate
        if path.exists():
            return path
    return None


def _basic_validate(path: Path, loader: ArtifactLoader) -> tuple[bool, str]:
    if loader == "directory":
        return path.is_dir(), "directory exists" if path.is_dir() else "not a directory"
    if path.is_dir():
        return False, "expected a file but found a directory"
    if path.stat().st_size == 0:
        return False, "file is empty"
    try:
        if loader == "json" or path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        elif loader == "csv" or path.suffix.lower() == ".csv":
            pd.read_csv(path, nrows=5)
        elif loader == "parquet" or path.suffix.lower() == ".parquet":
            pd.read_parquet(path).head()
    except Exception as exc:  # structured audit should not crash
        return False, f"could not read artifact: {exc}"
    return True, "artifact found and passed basic validation"


def audit_artifacts(
    project_root: Path,
    registry: Iterable[ArtifactSpec] | None = None,
) -> pd.DataFrame:
    """Return one audit row per registered artifact."""

    specs = tuple(registry or default_artifact_registry())
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = resolve_artifact_path(project_root, spec)
        if path is None:
            status = ArtifactStatus(
                name=spec.name,
                category=spec.category,
                required_for_final=spec.required_for_final,
                found=False,
                valid=False,
                resolved_path=None,
                loader=spec.loader,
                notes="PENDING — no candidate path exists",
            )
        else:
            valid, notes = _basic_validate(path, spec.loader)
            status = ArtifactStatus(
                name=spec.name,
                category=spec.category,
                required_for_final=spec.required_for_final,
                found=True,
                valid=valid,
                resolved_path=str(path),
                loader=spec.loader,
                notes=notes,
            )
        rows.append(asdict(status))
    return pd.DataFrame(rows)


def load_artifact(path: Path) -> Any:
    """Load a supported JSON, CSV, or Parquet artifact."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported artifact type: {path}")


def load_registered_artifact(
    project_root: Path,
    artifact_name: str,
    registry: Iterable[ArtifactSpec] | None = None,
) -> Any | None:
    """Resolve and load one logical artifact, returning None when unavailable."""

    specs = {spec.name: spec for spec in (registry or default_artifact_registry())}
    if artifact_name not in specs:
        raise KeyError(f"Unknown artifact: {artifact_name}")
    path = resolve_artifact_path(project_root, specs[artifact_name])
    return None if path is None else load_artifact(path)


__all__ = [
    "ArtifactSpec",
    "ArtifactStatus",
    "audit_artifacts",
    "default_artifact_registry",
    "load_artifact",
    "load_registered_artifact",
    "resolve_artifact_path",
]
