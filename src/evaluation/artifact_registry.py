"""Canonical artifact discovery and schema validation for Notebook 09.

Notebook 09 consumes explicit final-result packages produced by Notebooks 04-08.
The registry does not treat Notebook 09's own exports as upstream evidence and it
validates useful schema content rather than file existence alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pandas as pd


ArtifactLoader = Literal["json", "csv", "parquet", "torch", "joblib", "any"]


@dataclass(frozen=True)
class ArtifactSpec:
    """Description of one logical project artifact."""

    name: str
    category: str
    candidate_paths: tuple[str, ...]
    required_for_final: bool = True
    loader: ArtifactLoader = "any"
    description: str = ""
    required_key_paths: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    minimum_rows: int = 0
    json_root_type: Literal["mapping", "list", "any"] = "any"


@dataclass(frozen=True)
class ArtifactStatus:
    """Resolved and validated state of one artifact specification."""

    name: str
    category: str
    required_for_final: bool
    found: bool
    valid: bool
    resolved_path: str | None
    loader: str
    notes: str


def _has_key_path(payload: Any, key_path: str) -> bool:
    current = payload
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def default_artifact_registry() -> tuple[ArtifactSpec, ...]:
    """Return the canonical final-evaluation artifact contract."""

    return (
        ArtifactSpec(
            name="production_dataset_manifest",
            category="data",
            candidate_paths=(
                "data/manifests/production_dataset_manifest.json",
                "data/manifests/production_generation_manifest.json",
            ),
            loader="json",
            description="Production design, component counts, split metadata, and hashes.",
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="direct_final_metrics",
            category="notebook_04",
            candidate_paths=("artifacts/direct_mlp/final_metrics.json",),
            loader="json",
            description="Canonical Notebook 04 pricing, consistency, OOD, and runtime package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "pricing.black_scholes_proxy",
                "pricing.direct_mlp",
                "financial_consistency",
                "ood",
                "runtime",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="direct_test_predictions",
            category="notebook_04",
            candidate_paths=(
                "artifacts/direct_mlp/test_predictions.parquet",
                "artifacts/direct_mlp/test_predictions.csv",
            ),
            loader="any",
            description="Aligned Notebook 04 test predictions.",
            required_columns=("sample_id",),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="premium_final_metrics",
            category="notebook_05",
            candidate_paths=("artifacts/premium_models/final_metrics.json",),
            loader="json",
            description="Canonical Notebook 05 model comparison and residual-learning package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "selected_candidate",
                "pricing",
                "premium_error",
                "financial_consistency",
                "ood",
                "runtime",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="premium_test_predictions",
            category="notebook_05",
            candidate_paths=(
                "artifacts/premium_models/test_predictions.parquet",
                "artifacts/premium_models/test_predictions.csv",
            ),
            loader="any",
            description="Aligned Notebook 05 test predictions.",
            required_columns=("sample_id",),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="premium_checkpoint",
            category="checkpoint",
            candidate_paths=("artifacts/premium_models/best_premium_model.pt",),
            loader="torch",
            description="Canonical selected Notebook 05 checkpoint.",
        ),
        ArtifactSpec(
            name="multitask_final_metrics",
            category="notebook_06",
            candidate_paths=("artifacts/multitask_model/final_metrics.json",),
            loader="json",
            description="Canonical Notebook 06 classifier, boundary, price, and OOD package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "classification",
                "pricing",
                "boundary_bands",
                "boundary_pricing",
                "boundary_location",
                "ood",
                "hypothesis",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="multitask_test_predictions",
            category="notebook_06",
            candidate_paths=(
                "artifacts/multitask_model/test_predictions.parquet",
                "artifacts/multitask_model/test_predictions.csv",
            ),
            loader="any",
            description="Aligned Notebook 06 test predictions.",
            required_columns=("sample_id",),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="multitask_checkpoint",
            category="checkpoint",
            candidate_paths=("artifacts/multitask_model/best_multitask_pricer.pt",),
            loader="torch",
            description="Canonical selected Notebook 06 checkpoint.",
        ),
        ArtifactSpec(
            name="lsm_final_metrics",
            category="notebook_07",
            candidate_paths=("artifacts/neural_lsm/final_metrics.json",),
            loader="json",
            description="Canonical Notebook 07 held-out, policy, OOD, and runtime package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "heldout_pricing",
                "coverage",
                "policy_summary",
                "ood_pricing",
                "runtime",
                "runtime_context",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="lsm_heldout_pricing",
            category="notebook_07",
            candidate_paths=(
                "artifacts/neural_lsm/heldout_pricing_results.parquet",
                "artifacts/neural_lsm/heldout_pricing_results.csv",
            ),
            loader="any",
            description="Notebook 07 held-out contract-level pricing results.",
            required_columns=(
                "contract_id",
                "crr_price",
                "classical_lsm_price",
                "neural_lsm_price",
            ),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="lsm_policy_metrics",
            category="notebook_07",
            candidate_paths=(
                "artifacts/neural_lsm/heldout_policy_metrics.parquet",
                "artifacts/neural_lsm/heldout_policy_metrics.csv",
            ),
            required_for_final=False,
            loader="any",
            description="Notebook 07 contract-level stopping-policy comparison.",
            required_columns=("contract_id",),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="lsm_policy_checkpoint",
            category="checkpoint",
            candidate_paths=("artifacts/neural_lsm/neural_lsm_policy.pt",),
            loader="torch",
            description="Saved neural Longstaff-Schwartz policy.",
        ),
        ArtifactSpec(
            name="integrated_test_metrics",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/test_metrics.json",),
            loader="json",
            description="Notebook 08 integrated test metrics.",
            required_key_paths=(
                "constrained_mae",
                "constrained_rmse",
                "direct_rmse",
                "continuation_rmse",
                "exercise_f1",
                "exercise_balanced_accuracy",
                "consistency_decision_disagreement_rate",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="integrated_pricing_metrics",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/pricing_metrics.csv",),
            loader="csv",
            description="Notebook 08 constrained and direct pricing-head metrics.",
            required_columns=("head", "mae", "rmse"),
            minimum_rows=2,
        ),
        ArtifactSpec(
            name="integrated_exercise_metrics",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/exercise_metrics.json",),
            loader="json",
            description="Notebook 08 exercise-head classification metrics.",
            required_key_paths=("balanced_accuracy", "precision", "recall", "f1"),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="integrated_continuation_metrics",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/continuation_metrics.json",),
            loader="json",
            description="Notebook 08 continuation-value regression metrics.",
            required_key_paths=("mae", "rmse"),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="integrated_consistency_metrics",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/consistency_metrics.json",),
            loader="json",
            description="Notebook 08 internal financial-consistency metrics.",
            required_key_paths=("decision_disagreement_rate",),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="integrated_boundary_analysis",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/boundary_analysis.csv",),
            loader="csv",
            description="Notebook 08 performance by exercise-boundary distance.",
            required_columns=(
                "boundary_band",
                "observations",
                "price_mae",
                "exercise_accuracy",
            ),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="integrated_ood_metrics",
            category="notebook_08",
            candidate_paths=(
                "artifacts/final_multihead/ood_metrics.json",
                "artifacts/final_multihead/ood_metrics.csv",
            ),
            loader="any",
            description="Notebook 08 out-of-domain results.",
            minimum_rows=1,
            json_root_type="list",
        ),
        ArtifactSpec(
            name="integrated_runtime",
            category="notebook_08",
            candidate_paths=("artifacts/final_multihead/runtime_summary.json",),
            loader="json",
            description="Notebook 08 marginal inference runtime.",
            required_key_paths=(
                "observations",
                "seconds",
                "seconds_per_observation",
                "observations_per_second",
                "device",
            ),
            json_root_type="mapping",
        ),
        ArtifactSpec(
            name="integrated_test_predictions",
            category="notebook_08",
            candidate_paths=(
                "artifacts/final_multihead/test_predictions.parquet",
                "artifacts/final_multihead/test_predictions.csv",
            ),
            loader="any",
            description="Aligned Notebook 08 test predictions for all heads.",
            required_columns=("sample_id",),
            minimum_rows=1,
        ),
        ArtifactSpec(
            name="integrated_checkpoint",
            category="checkpoint",
            candidate_paths=("artifacts/final_multihead/best_integrated_multihead.pt",),
            loader="torch",
            description="Canonical selected Notebook 08 checkpoint.",
        ),
    )


def resolve_artifact_path(project_root: Path, spec: ArtifactSpec) -> Path | None:
    """Resolve the first existing candidate path for an artifact."""

    root = Path(project_root).resolve()
    for candidate in spec.candidate_paths:
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.exists():
            return path
    return None


def _read_table_sample(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=10)
    if suffix == ".parquet":
        return pd.read_parquet(path).head(10)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def _validate_json(payload: Any, spec: ArtifactSpec) -> tuple[bool, str]:
    if spec.json_root_type == "mapping" and not isinstance(payload, dict):
        return False, "JSON root must be a mapping"
    if spec.json_root_type == "list" and not isinstance(payload, list):
        return False, "JSON root must be a list"
    missing = [key for key in spec.required_key_paths if not _has_key_path(payload, key)]
    if missing:
        return False, "missing JSON keys: " + ", ".join(missing)
    if isinstance(payload, dict) and "status" in payload and payload["status"] != "complete":
        return False, "JSON package status is not complete"
    if isinstance(payload, list) and spec.minimum_rows and len(payload) < spec.minimum_rows:
        return False, f"JSON list has fewer than {spec.minimum_rows} rows"
    return True, "JSON schema validated"


def _basic_validate(path: Path, spec: ArtifactSpec) -> tuple[bool, str]:
    if not path.is_file():
        return False, "expected a file"
    if path.stat().st_size == 0:
        return False, "file is empty"

    suffix = path.suffix.lower()
    try:
        if spec.loader == "json" or suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _validate_json(payload, spec)

        if spec.required_columns or spec.minimum_rows:
            frame = _read_table_sample(path)
            missing_columns = sorted(set(spec.required_columns).difference(frame.columns))
            if missing_columns:
                return False, "missing table columns: " + ", ".join(missing_columns)
            if spec.minimum_rows and frame.empty:
                return False, "table contains no rows"
            return True, "table schema validated"

        if suffix in {".csv", ".parquet"}:
            _read_table_sample(path)
            return True, "table is readable"

    except Exception as exc:
        return False, f"could not read artifact: {exc}"

    return True, "artifact exists and is non-empty"


def audit_artifacts(
    project_root: Path,
    registry: Iterable[ArtifactSpec] | None = None,
) -> pd.DataFrame:
    """Return one validation row per registered artifact."""

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
                notes="missing — no candidate path exists",
            )
        else:
            valid, notes = _basic_validate(path, spec)
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
