"""Canonical artifact discovery and schema validation for Notebook 09.

This registry describes the current final-result packages produced by
Notebooks 04-08.  It validates file presence, basic readability, declared
package status, expected notebook identity, execution profile, JSON fields,
and table schemas.  Cross-file coherence and prediction alignment are handled
by :mod:`src.evaluation.final_lineage_audit`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pandas as pd


ArtifactLoader = Literal["json", "csv", "parquet", "torch", "joblib", "any"]
JsonRootType = Literal["mapping", "list", "any"]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Contract for one logical project artifact."""

    name: str
    category: str
    notebook: str | None
    candidate_paths: tuple[str, ...]
    required_for_final: bool = True
    loader: ArtifactLoader = "any"
    description: str = ""
    required_key_paths: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    minimum_rows: int = 0
    json_root_type: JsonRootType = "any"
    expected_notebook: str | None = None
    allowed_profiles: tuple[str, ...] = ()
    require_complete_status: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    """Resolved and validated state of one artifact specification."""

    name: str
    category: str
    notebook: str | None
    required_for_final: bool
    found: bool
    valid: bool
    resolved_path: str | None
    loader: str
    rows: int | None
    notes: str


def _spec(
    name: str,
    category: str,
    notebook: str | None,
    *candidate_paths: str,
    **kwargs: Any,
) -> ArtifactSpec:
    return ArtifactSpec(
        name=name,
        category=category,
        notebook=notebook,
        candidate_paths=tuple(candidate_paths),
        **kwargs,
    )


def default_artifact_registry() -> tuple[ArtifactSpec, ...]:
    """Return the current final-evaluation artifact contract.

    The contract intentionally includes both aggregate result packages and the
    prediction files required for a paired cross-notebook comparison.
    """

    return (
        _spec(
            "production_dataset_manifest",
            "data",
            None,
            "data/manifests/production_dataset_manifest.json",
            loader="json",
            description="Production dataset design and component metadata.",
            json_root_type="mapping",
        ),
        # Notebook 04 -----------------------------------------------------
        _spec(
            "nb04_final_metrics",
            "final_metrics",
            "04",
            "artifacts/direct_mlp/final_metrics.json",
            loader="json",
            description="Direct model final result package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "training_profile",
                "selected_model",
                "checkpoint",
                "pricing.black_scholes_proxy",
                "pricing.direct_mlp",
                "financial_consistency",
                "ood",
                "runtime",
            ),
            json_root_type="mapping",
            expected_notebook="04_direct_mlp_pricer",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb04_test_predictions",
            "test_predictions",
            "04",
            "artifacts/direct_mlp/test_predictions.parquet",
            "artifacts/direct_mlp/test_predictions.csv",
            loader="any",
            description="Notebook 04 predictions on the common static test set.",
            required_columns=(
                "sample_id",
                "normalized_american_price",
                "direct_mlp_prediction",
            ),
            minimum_rows=1,
        ),
        _spec(
            "nb04_checkpoint",
            "checkpoint",
            "04",
            "artifacts/direct_mlp/best_direct_mlp.pt",
            loader="torch",
            description="Canonical Notebook 04 checkpoint.",
        ),
        _spec(
            "nb04_training_manifest",
            "training_manifest",
            "04",
            "artifacts/direct_mlp/training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "checkpoint",
            ),
            json_root_type="mapping",
            expected_notebook="04_direct_mlp_pricer",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        # Notebook 05 -----------------------------------------------------
        _spec(
            "nb05_final_metrics",
            "final_metrics",
            "05",
            "artifacts/premium_models/final_metrics.json",
            loader="json",
            description="Residual-model final result package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "training_profile",
                "selected_model",
                "selected_candidate",
                "checkpoint",
                "pricing",
                "premium_error",
                "financial_consistency",
                "segmented_results",
                "ood",
                "runtime",
                "hypotheses",
            ),
            json_root_type="mapping",
            expected_notebook="05_early_exercise_premium_model",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb05_test_predictions",
            "test_predictions",
            "05",
            "artifacts/premium_models/test_predictions.parquet",
            "artifacts/premium_models/test_predictions.csv",
            loader="any",
            description="Notebook 05 predictions on the common static test set.",
            required_columns=(
                "sample_id",
                "normalized_american_price",
                "constrained_floor_prediction",
            ),
            minimum_rows=1,
        ),
        _spec(
            "nb05_checkpoint",
            "checkpoint",
            "05",
            "artifacts/premium_models/best_premium_model.pt",
            loader="torch",
            description="Canonical validation-selected Notebook 05 checkpoint.",
        ),
        _spec(
            "nb05_training_manifest",
            "training_manifest",
            "05",
            "artifacts/premium_models/training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
                "candidates",
            ),
            json_root_type="mapping",
            expected_notebook="05_early_exercise_premium_model",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        # Notebook 06 -----------------------------------------------------
        _spec(
            "nb06_final_metrics",
            "final_metrics",
            "06",
            "artifacts/multitask_model/final_metrics.json",
            loader="json",
            description="Exercise and multi-task final result package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "training_profile",
                "selected_candidate",
                "checkpoint",
                "dependencies",
                "thresholds",
                "classification",
                "pricing",
                "boundary_bands",
                "boundary_pricing",
                "boundary_location",
                "financial_consistency",
                "ood_classification",
                "ood_pricing",
                "inference",
                "hypothesis",
            ),
            json_root_type="mapping",
            expected_notebook="06_exercise_boundary_analysis",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb06_test_predictions",
            "test_predictions",
            "06",
            "artifacts/multitask_model/test_predictions.parquet",
            "artifacts/multitask_model/test_predictions.csv",
            loader="any",
            description="Notebook 06 predictions on the common static test set.",
            required_columns=(
                "sample_id",
                "normalized_american_price",
                "exercise_now",
                "classifier_probability",
                "multitask_probability",
                "predicted_normalized_american_price",
                "price_only_normalized_price",
            ),
            minimum_rows=1,
        ),
        _spec(
            "nb06_multitask_checkpoint",
            "checkpoint",
            "06",
            "artifacts/multitask_model/best_multitask_pricer.pt",
            loader="torch",
            description="Canonical validation-selected Notebook 06 multi-task checkpoint.",
        ),
        _spec(
            "nb06_classifier_checkpoint",
            "checkpoint",
            "06",
            "artifacts/multitask_model/best_exercise_classifier.pt",
            loader="torch",
            description="Notebook 06 specialist exercise classifier.",
        ),
        _spec(
            "nb06_multitask_manifest",
            "training_manifest",
            "06",
            "artifacts/multitask_model/multitask_training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
                "candidates",
            ),
            json_root_type="mapping",
            expected_notebook="06_multitask_model",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb06_classifier_manifest",
            "training_manifest",
            "06",
            "artifacts/multitask_model/exercise_classifier_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
                "checkpoint",
            ),
            json_root_type="mapping",
            expected_notebook="06_exercise_classifier",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        # Notebook 07 -----------------------------------------------------
        _spec(
            "nb07_final_metrics",
            "final_metrics",
            "07",
            "artifacts/neural_lsm/final_metrics.json",
            loader="json",
            description="Classical and neural LSM final result package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "training_profile",
                "selected_model",
                "neural_policy_checkpoint",
                "selected_classical_basis",
                "selected_classical_degree",
                "training",
                "heldout_pricing",
                "paired_mae_bootstrap",
                "coverage",
                "financial_bounds",
                "policy_summary",
                "ood_pricing",
                "ood_policy",
                "runtime",
                "runtime_context",
                "runtime_break_even",
                "h5_decision",
            ),
            json_root_type="mapping",
            expected_notebook="07_neural_longstaff_schwartz",
            allowed_profiles=("final",),
            require_complete_status=True,
        ),
        _spec(
            "nb07_heldout_pricing",
            "heldout_results",
            "07",
            "artifacts/neural_lsm/heldout_pricing_results.parquet",
            "artifacts/neural_lsm/heldout_pricing_results.csv",
            "artifacts/neural_lsm/heldout_comparison.csv",
            loader="any",
            description="Held-out contract-level pricing results.",
            required_columns=("contract_id",),
            minimum_rows=1,
        ),
        _spec(
            "nb07_policy_checkpoint",
            "checkpoint",
            "07",
            "artifacts/neural_lsm/neural_lsm_policy.pt",
            loader="torch",
            description="Saved neural continuation policy.",
        ),
        _spec(
            "nb07_training_manifest",
            "training_manifest",
            "07",
            "artifacts/neural_lsm/training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
                "checkpoint",
            ),
            json_root_type="mapping",
            expected_notebook="07_neural_longstaff_schwartz",
            allowed_profiles=("final",),
            require_complete_status=True,
        ),
        # Notebook 08 -----------------------------------------------------
        _spec(
            "nb08_final_metrics",
            "final_metrics",
            "08",
            "artifacts/final_multihead/final_metrics.json",
            loader="json",
            description="Integrated static model final result package.",
            required_key_paths=(
                "schema_version",
                "status",
                "notebook",
                "training_profile",
                "selected_scratch_configuration",
                "preferred_integrated_candidate",
                "checkpoint",
                "deployment_checkpoint",
                "scratch_checkpoint",
                "authoritative_price_output",
                "deployment_scope",
                "fallback_method",
                "thresholds.exercise_head",
                "thresholds.continuation_path",
                "selection",
                "pricing",
                "continuation",
                "classification",
                "consistency",
                "boundary",
                "ood",
                "runtime",
                "scratch_benchmark",
                "integrated_candidate_comparison",
                "dependencies",
            ),
            json_root_type="mapping",
            expected_notebook="08_final_multihead_model",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb08_selection",
            "selection",
            "08",
            "artifacts/final_multihead/selection.json",
            loader="json",
            required_key_paths=(
                "schema_version",
                "configuration",
                "selected_scratch_configuration",
                "selected_scratch_checkpoint",
                "preferred_integrated_candidate",
                "preferred_integrated_checkpoint",
                "canonical_checkpoint",
                "authoritative_price_output",
                "selection_basis",
                "test_metrics_used_for_selection",
                "ood_metrics_used_for_selection",
                "deployment_scope",
                "fallback_method",
                "dependencies",
            ),
            json_root_type="mapping",
        ),
        _spec(
            "nb08_test_predictions",
            "test_predictions",
            "08",
            "artifacts/final_multihead/test_predictions.parquet",
            "artifacts/final_multihead/test_predictions.csv",
            loader="any",
            description="Notebook 08 predictions for all four outputs.",
            required_columns=(
                "sample_id",
                "true_normalized_american_price",
                "predicted_normalized_american_price",
                "predicted_direct_normalized_american_price",
                "exercise_target",
                "exercise_probability",
                "continuation_exercise_probability",
                "true_normalized_continuation_value",
                "predicted_normalized_continuation_value",
            ),
            minimum_rows=1,
        ),
        _spec(
            "nb08_boundary_analysis",
            "boundary_results",
            "08",
            "artifacts/final_multihead/boundary_analysis.csv",
            loader="csv",
            required_columns=(
                "decision_path",
                "boundary_band",
                "boundary_limit",
                "observations",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "price_mae",
                "decision_errors",
                "total_regret",
            ),
            minimum_rows=2,
        ),
        _spec(
            "nb08_runtime",
            "runtime",
            "08",
            "artifacts/final_multihead/runtime_summary.json",
            loader="json",
            required_key_paths=(
                "observations",
                "median_seconds",
                "seconds_per_observation",
                "observations_per_second",
                "device",
            ),
            json_root_type="mapping",
        ),
        _spec(
            "nb08_checkpoint",
            "checkpoint",
            "08",
            "artifacts/final_multihead/best_integrated_multihead.pt",
            loader="torch",
            description="Backward-compatible canonical deployment checkpoint.",
        ),
        _spec(
            "nb08_deployment_checkpoint",
            "checkpoint",
            "08",
            "artifacts/final_multihead/best_integrated_deployment.pt",
            loader="torch",
            description="Preferred warm-start in-domain deployment checkpoint.",
        ),
        _spec(
            "nb08_scratch_checkpoint",
            "checkpoint",
            "08",
            "artifacts/final_multihead/best_integrated_scratch.pt",
            loader="torch",
            description="Balanced scratch winner retained as robustness benchmark.",
        ),
        _spec(
            "nb08_scratch_test_predictions",
            "test_predictions",
            "08",
            "artifacts/final_multihead/scratch_test_predictions.parquet",
            "artifacts/final_multihead/scratch_test_predictions.csv",
            loader="any",
            description="Balanced scratch predictions on the common static test set.",
            required_columns=(
                "sample_id",
                "true_normalized_american_price",
                "predicted_normalized_american_price",
                "predicted_direct_normalized_american_price",
                "exercise_target",
                "exercise_probability",
                "continuation_exercise_probability",
            ),
            minimum_rows=1,
        ),
        _spec(
            "nb08_deployment_selection",
            "selection",
            "08",
            "artifacts/final_multihead/deployment_selection.json",
            loader="json",
            required_key_paths=(
                "preferred_integrated_candidate",
                "selection_scope",
                "selection_rule",
                "all_checks_passed",
                "test_metrics_used_for_selection",
                "ood_metrics_used_for_selection",
                "checks",
            ),
            json_root_type="mapping",
        ),
        _spec(
            "nb08_deployment_policy",
            "deployment_policy",
            "08",
            "artifacts/final_multihead/deployment_policy.json",
            loader="json",
            required_key_paths=(
                "preferred_integrated_candidate",
                "neural_scope",
                "fallback_method",
                "fallback_trigger",
                "price_only_preference",
                "exercise_only_preference",
            ),
            json_root_type="mapping",
        ),
        _spec(
            "nb08_domain_bounds",
            "deployment_policy",
            "08",
            "artifacts/final_multihead/domain_bounds.json",
            loader="json",
            required_key_paths=(
                "moneyness",
                "time_to_maturity",
                "volatility",
                "risk_free_rate",
                "dividend_yield",
            ),
            json_root_type="mapping",
        ),
        _spec(
            "nb08_scratch_manifest",
            "training_manifest",
            "08",
            "artifacts/final_multihead/scratch_training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
            ),
            json_root_type="mapping",
            expected_notebook="08_final_multihead_scratch",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
        _spec(
            "nb08_warm_manifest",
            "training_manifest",
            "08",
            "artifacts/final_multihead/warm_start_training_complete.json",
            loader="json",
            required_key_paths=(
                "status",
                "notebook",
                "training_profile",
                "dependencies",
                "checkpoint",
            ),
            json_root_type="mapping",
            expected_notebook="08_final_multihead_warm_start",
            allowed_profiles=("full",),
            require_complete_status=True,
        ),
    )


def _has_key_path(payload: Any, key_path: str) -> bool:
    current = payload
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def resolve_artifact_path(project_root: Path, spec: ArtifactSpec) -> Path | None:
    """Resolve the first existing candidate path inside the project root."""

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


def _table_schema(path: Path) -> tuple[tuple[str, ...], int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=20)
        # Count lines without materialising the entire CSV.  Header is removed.
        with path.open("rb") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
        return tuple(str(column) for column in frame.columns), row_count
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            return tuple(parquet.schema.names), int(parquet.metadata.num_rows)
        except ImportError:
            frame = pd.read_parquet(path)
            return tuple(str(column) for column in frame.columns), int(len(frame))
    raise ValueError(f"Unsupported table suffix: {suffix}")


def _validate_json(payload: Any, spec: ArtifactSpec) -> tuple[bool, str, int | None]:
    if spec.json_root_type == "mapping" and not isinstance(payload, dict):
        return False, "JSON root must be a mapping", None
    if spec.json_root_type == "list" and not isinstance(payload, list):
        return False, "JSON root must be a list", None

    missing = [
        key_path
        for key_path in spec.required_key_paths
        if not _has_key_path(payload, key_path)
    ]
    if missing:
        return False, "missing JSON keys: " + ", ".join(missing), None

    if spec.require_complete_status:
        status = payload.get("status") if isinstance(payload, dict) else None
        if status != "complete":
            return False, f"package status is {status!r}, expected 'complete'", None

    if spec.expected_notebook and isinstance(payload, dict):
        notebook = payload.get("notebook")
        if notebook != spec.expected_notebook:
            return (
                False,
                f"notebook identity is {notebook!r}, expected {spec.expected_notebook!r}",
                None,
            )

    if spec.allowed_profiles and isinstance(payload, dict):
        profile = payload.get("training_profile")
        if profile not in spec.allowed_profiles:
            return (
                False,
                f"training profile is {profile!r}, expected one of {spec.allowed_profiles}",
                None,
            )

    rows: int | None = len(payload) if isinstance(payload, list) else None
    if spec.minimum_rows and rows is not None and rows < spec.minimum_rows:
        return False, f"JSON list has {rows} rows; expected at least {spec.minimum_rows}", rows
    return True, "JSON schema validated", rows


def _basic_validate(path: Path, spec: ArtifactSpec) -> tuple[bool, str, int | None]:
    if not path.is_file():
        return False, "expected a file", None
    if path.stat().st_size <= 0:
        return False, "file is empty", None

    suffix = path.suffix.lower()
    try:
        if spec.loader == "json" or suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _validate_json(payload, spec)

        if suffix in {".csv", ".parquet"}:
            columns, rows = _table_schema(path)
            missing_columns = sorted(set(spec.required_columns).difference(columns))
            if missing_columns:
                return (
                    False,
                    "missing table columns: " + ", ".join(missing_columns),
                    rows,
                )
            if spec.minimum_rows and rows < spec.minimum_rows:
                return (
                    False,
                    f"table has {rows} rows; expected at least {spec.minimum_rows}",
                    rows,
                )
            return True, "table schema validated", rows
    except Exception as exc:  # pragma: no cover - exact reader exceptions vary
        return False, f"could not read artifact: {exc}", None

    return True, "artifact exists and is non-empty", None


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
                notebook=spec.notebook,
                required_for_final=spec.required_for_final,
                found=False,
                valid=False,
                resolved_path=None,
                loader=spec.loader,
                rows=None,
                notes="missing — no candidate path exists",
            )
        else:
            valid, notes, row_count = _basic_validate(path, spec)
            status = ArtifactStatus(
                name=spec.name,
                category=spec.category,
                notebook=spec.notebook,
                required_for_final=spec.required_for_final,
                found=True,
                valid=valid,
                resolved_path=str(path),
                loader=spec.loader,
                rows=row_count,
                notes=notes,
            )
        rows.append(asdict(status))
    return pd.DataFrame(rows)


def get_artifact_spec(name: str) -> ArtifactSpec:
    """Return one registered specification by name."""

    matches = [spec for spec in default_artifact_registry() if spec.name == name]
    if len(matches) != 1:
        raise KeyError(f"Artifact specification not found or not unique: {name!r}")
    return matches[0]


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


def load_registered_artifact(project_root: Path, name: str) -> Any:
    """Resolve, validate, and load one registered artifact."""

    spec = get_artifact_spec(name)
    path = resolve_artifact_path(project_root, spec)
    if path is None:
        raise FileNotFoundError(
            f"No candidate path exists for registered artifact {name!r}: "
            f"{spec.candidate_paths}"
        )
    valid, notes, _ = _basic_validate(path, spec)
    if not valid:
        raise ValueError(f"Artifact {name!r} is invalid: {notes}")
    return load_artifact(path)


def assert_required_artifacts_valid(audit: pd.DataFrame) -> None:
    """Raise when any required artifact is missing or invalid."""

    required = audit.loc[audit["required_for_final"]]
    invalid = required.loc[~required["valid"]]
    if invalid.empty:
        return
    details = "; ".join(
        f"{row.name}: {row.notes}"
        for row in invalid.itertuples(index=False)
    )
    raise RuntimeError(f"Required final artifacts are invalid: {details}")


__all__ = [
    "ArtifactSpec",
    "ArtifactStatus",
    "assert_required_artifacts_valid",
    "audit_artifacts",
    "default_artifact_registry",
    "get_artifact_spec",
    "load_artifact",
    "load_registered_artifact",
    "resolve_artifact_path",
]
