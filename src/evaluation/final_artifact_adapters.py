"""Explicit adapters for the final result packages from Notebooks 04-08.

The adapters replace the old inference-by-column-position approach.  Every
package has a declared notebook identity, selected model, canonical checkpoint,
training profile, aggregate result payload, and (where applicable) test
prediction table.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.evaluation.artifact_registry import (
    get_artifact_spec,
    load_artifact,
    load_registered_artifact,
    resolve_artifact_path,
)


class FinalPackageError(RuntimeError):
    """Raised when a final notebook package is internally inconsistent."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalPackageError(f"{label} must be a JSON mapping")
    return dict(value)


def _registered_path(project_root: Path, name: str) -> Path:
    spec = get_artifact_spec(name)
    path = resolve_artifact_path(project_root, spec)
    if path is None:
        raise FileNotFoundError(
            f"Missing registered artifact {name!r}: {spec.candidate_paths}"
        )
    return path


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FinalPackageError(f"{label} is missing or empty: {path}")
    return path


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise FinalPackageError(
            f"{label} mismatch: found {actual!r}, expected {expected!r}"
        )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalPackageError(f"Could not read {label}: {exc}") from exc
    return _mapping(payload, label=label)


@dataclass(frozen=True, slots=True)
class FinalNotebookPackage:
    notebook_number: str
    notebook_id: str
    training_profile: str
    selected_model: str
    checkpoint_name: str
    checkpoint_path: Path
    final_metrics_path: Path
    final_metrics: dict[str, Any]
    training_manifests: tuple[Path, ...]
    test_predictions_path: Path | None = None
    benchmark_checkpoint_path: Path | None = None
    benchmark_test_predictions_path: Path | None = None
    deployment_policy_path: Path | None = None

    def load_test_predictions(self) -> pd.DataFrame:
        if self.test_predictions_path is None:
            raise FinalPackageError(
                f"Notebook {self.notebook_number} has no registered test predictions"
            )
        table = load_artifact(self.test_predictions_path)
        if not isinstance(table, pd.DataFrame):
            raise FinalPackageError("Prediction artifact did not load as a DataFrame")
        return table

    def load_benchmark_test_predictions(self) -> pd.DataFrame:
        if self.benchmark_test_predictions_path is None:
            raise FinalPackageError(
                f"Notebook {self.notebook_number} has no benchmark prediction package"
            )
        table = load_artifact(self.benchmark_test_predictions_path)
        if not isinstance(table, pd.DataFrame):
            raise FinalPackageError(
                "Benchmark prediction artifact did not load as a DataFrame"
            )
        return table

    @property
    def status(self) -> str:
        return str(self.final_metrics.get("status", ""))

    def summary_record(self) -> dict[str, Any]:
        prediction_rows: int | None = None
        if self.test_predictions_path is not None:
            try:
                if self.test_predictions_path.suffix.lower() == ".parquet":
                    import pyarrow.parquet as pq

                    prediction_rows = int(
                        pq.ParquetFile(self.test_predictions_path).metadata.num_rows
                    )
                else:
                    with self.test_predictions_path.open("rb") as handle:
                        prediction_rows = max(sum(1 for _ in handle) - 1, 0)
            except Exception:
                prediction_rows = None
        return {
            "notebook": self.notebook_number,
            "notebook_id": self.notebook_id,
            "status": self.status,
            "training_profile": self.training_profile,
            "selected_model": self.selected_model,
            "checkpoint": self.checkpoint_name,
            "checkpoint_path": str(self.checkpoint_path),
            "benchmark_checkpoint_path": (
                str(self.benchmark_checkpoint_path)
                if self.benchmark_checkpoint_path is not None
                else None
            ),
            "test_prediction_rows": prediction_rows,
            "benchmark_test_predictions_path": (
                str(self.benchmark_test_predictions_path)
                if self.benchmark_test_predictions_path is not None
                else None
            ),
            "final_metrics_path": str(self.final_metrics_path),
        }


def _build_base_package(
    project_root: Path,
    *,
    notebook_number: str,
    metrics_artifact: str,
    checkpoint_artifact: str,
    prediction_artifact: str | None,
    manifest_artifacts: tuple[str, ...],
    expected_notebook: str,
    allowed_profiles: tuple[str, ...],
    selected_model: str,
    checkpoint_name: str,
) -> FinalNotebookPackage:
    root = Path(project_root).resolve()
    metrics_path = _registered_path(root, metrics_artifact)
    payload = _mapping(
        load_registered_artifact(root, metrics_artifact),
        label=metrics_artifact,
    )
    _require_equal(payload.get("status"), "complete", label="package status")
    _require_equal(payload.get("notebook"), expected_notebook, label="notebook identity")
    profile = str(payload.get("training_profile"))
    if profile not in allowed_profiles:
        raise FinalPackageError(
            f"Notebook {notebook_number} profile {profile!r} is not final; "
            f"expected one of {allowed_profiles}"
        )

    checkpoint_path = _require_file(
        _registered_path(root, checkpoint_artifact),
        label=f"Notebook {notebook_number} checkpoint",
    )
    _require_equal(checkpoint_path.name, checkpoint_name, label="checkpoint filename")

    prediction_path = (
        _registered_path(root, prediction_artifact)
        if prediction_artifact is not None
        else None
    )
    manifest_paths = tuple(_registered_path(root, name) for name in manifest_artifacts)

    return FinalNotebookPackage(
        notebook_number=notebook_number,
        notebook_id=expected_notebook,
        training_profile=profile,
        selected_model=selected_model,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        final_metrics_path=metrics_path,
        final_metrics=payload,
        training_manifests=manifest_paths,
        test_predictions_path=prediction_path,
    )


def load_notebook04_package(project_root: Path) -> FinalNotebookPackage:
    payload = _mapping(
        load_registered_artifact(project_root, "nb04_final_metrics"),
        label="nb04_final_metrics",
    )
    return _build_base_package(
        project_root,
        notebook_number="04",
        metrics_artifact="nb04_final_metrics",
        checkpoint_artifact="nb04_checkpoint",
        prediction_artifact="nb04_test_predictions",
        manifest_artifacts=("nb04_training_manifest",),
        expected_notebook="04_direct_mlp_pricer",
        allowed_profiles=("full",),
        selected_model=str(payload["selected_model"]),
        checkpoint_name=str(payload["checkpoint"]),
    )


def load_notebook05_package(project_root: Path) -> FinalNotebookPackage:
    payload = _mapping(
        load_registered_artifact(project_root, "nb05_final_metrics"),
        label="nb05_final_metrics",
    )
    selected_candidate = str(payload["selected_candidate"])
    if not selected_candidate:
        raise FinalPackageError("Notebook 05 selected_candidate is empty")
    return _build_base_package(
        project_root,
        notebook_number="05",
        metrics_artifact="nb05_final_metrics",
        checkpoint_artifact="nb05_checkpoint",
        prediction_artifact="nb05_test_predictions",
        manifest_artifacts=("nb05_training_manifest",),
        expected_notebook="05_early_exercise_premium_model",
        allowed_profiles=("full",),
        selected_model=str(payload["selected_model"]),
        checkpoint_name=str(payload["checkpoint"]),
    )


def load_notebook06_package(project_root: Path) -> FinalNotebookPackage:
    payload = _mapping(
        load_registered_artifact(project_root, "nb06_final_metrics"),
        label="nb06_final_metrics",
    )
    selected_candidate = str(payload["selected_candidate"])
    if not selected_candidate:
        raise FinalPackageError("Notebook 06 selected_candidate is empty")
    return _build_base_package(
        project_root,
        notebook_number="06",
        metrics_artifact="nb06_final_metrics",
        checkpoint_artifact="nb06_multitask_checkpoint",
        prediction_artifact="nb06_test_predictions",
        manifest_artifacts=(
            "nb06_multitask_manifest",
            "nb06_classifier_manifest",
        ),
        expected_notebook="06_exercise_boundary_analysis",
        allowed_profiles=("full",),
        selected_model=f"Multi-task candidate: {selected_candidate}",
        checkpoint_name=str(payload["checkpoint"]),
    )


def load_notebook07_package(project_root: Path) -> FinalNotebookPackage:
    payload = _mapping(
        load_registered_artifact(project_root, "nb07_final_metrics"),
        label="nb07_final_metrics",
    )
    return _build_base_package(
        project_root,
        notebook_number="07",
        metrics_artifact="nb07_final_metrics",
        checkpoint_artifact="nb07_policy_checkpoint",
        prediction_artifact=None,
        manifest_artifacts=("nb07_training_manifest",),
        expected_notebook="07_neural_longstaff_schwartz",
        allowed_profiles=("final",),
        selected_model=str(payload["selected_model"]),
        checkpoint_name=str(payload["neural_policy_checkpoint"]),
    )


def load_notebook08_package(project_root: Path) -> FinalNotebookPackage:
    root = Path(project_root).resolve()
    payload = _mapping(
        load_registered_artifact(root, "nb08_final_metrics"),
        label="nb08_final_metrics",
    )
    selection = _mapping(
        load_registered_artifact(root, "nb08_selection"),
        label="nb08_selection",
    )
    deployment_selection = _mapping(
        load_registered_artifact(root, "nb08_deployment_selection"),
        label="nb08_deployment_selection",
    )
    deployment_policy = _mapping(
        load_registered_artifact(root, "nb08_deployment_policy"),
        label="nb08_deployment_policy",
    )

    _require_equal(
        payload.get("selected_scratch_configuration"),
        selection.get("selected_scratch_configuration"),
        label="Notebook 08 scratch configuration",
    )
    _require_equal(
        payload.get("preferred_integrated_candidate"),
        selection.get("preferred_integrated_candidate"),
        label="Notebook 08 preferred deployment candidate",
    )
    _require_equal(
        payload.get("checkpoint"),
        selection.get("canonical_checkpoint"),
        label="Notebook 08 canonical checkpoint",
    )
    _require_equal(
        payload.get("deployment_checkpoint"),
        selection.get("preferred_integrated_checkpoint"),
        label="Notebook 08 deployment checkpoint",
    )
    _require_equal(
        payload.get("scratch_checkpoint"),
        selection.get("selected_scratch_checkpoint"),
        label="Notebook 08 scratch checkpoint",
    )
    _require_equal(
        payload.get("authoritative_price_output"),
        "constrained_price",
        label="Notebook 08 authoritative price output",
    )
    _require_equal(
        selection.get("authoritative_price_output"),
        "constrained_price",
        label="Notebook 08 selection authoritative output",
    )
    _require_equal(
        selection.get("test_metrics_used_for_selection"),
        False,
        label="Notebook 08 test-data selection exclusion",
    )
    _require_equal(
        selection.get("ood_metrics_used_for_selection"),
        False,
        label="Notebook 08 OOD selection exclusion",
    )
    _require_equal(
        deployment_selection.get("preferred_integrated_candidate"),
        "warm_start",
        label="Notebook 08 deployment selection",
    )
    _require_equal(
        deployment_policy.get("preferred_integrated_candidate"),
        "warm_start",
        label="Notebook 08 deployment policy candidate",
    )
    _require_equal(
        deployment_policy.get("fallback_method"),
        "high_resolution_crr",
        label="Notebook 08 fallback method",
    )

    package = _build_base_package(
        root,
        notebook_number="08",
        metrics_artifact="nb08_final_metrics",
        checkpoint_artifact="nb08_checkpoint",
        prediction_artifact="nb08_test_predictions",
        manifest_artifacts=(
            "nb08_scratch_manifest",
            "nb08_warm_manifest",
        ),
        expected_notebook="08_final_multihead_model",
        allowed_profiles=("full",),
        selected_model="Integrated deployment: warm_start",
        checkpoint_name=str(payload["checkpoint"]),
    )

    deployment_checkpoint = _require_file(
        _registered_path(root, "nb08_deployment_checkpoint"),
        label="Notebook 08 deployment checkpoint",
    )
    scratch_checkpoint = _require_file(
        _registered_path(root, "nb08_scratch_checkpoint"),
        label="Notebook 08 scratch benchmark checkpoint",
    )
    _require_equal(
        deployment_checkpoint.name,
        str(payload["deployment_checkpoint"]),
        label="Notebook 08 deployment checkpoint filename",
    )
    _require_equal(
        scratch_checkpoint.name,
        str(payload["scratch_checkpoint"]),
        label="Notebook 08 scratch checkpoint filename",
    )

    return FinalNotebookPackage(
        notebook_number=package.notebook_number,
        notebook_id=package.notebook_id,
        training_profile=package.training_profile,
        selected_model=package.selected_model,
        checkpoint_name=package.checkpoint_name,
        checkpoint_path=package.checkpoint_path,
        final_metrics_path=package.final_metrics_path,
        final_metrics=package.final_metrics,
        training_manifests=package.training_manifests,
        test_predictions_path=package.test_predictions_path,
        benchmark_checkpoint_path=scratch_checkpoint,
        benchmark_test_predictions_path=_registered_path(
            root,
            "nb08_scratch_test_predictions",
        ),
        deployment_policy_path=_registered_path(
            root,
            "nb08_deployment_policy",
        ),
    )

def load_all_final_packages(project_root: Path) -> dict[str, FinalNotebookPackage]:
    """Load and cross-check all final packages from Notebooks 04-08."""

    return {
        "04": load_notebook04_package(project_root),
        "05": load_notebook05_package(project_root),
        "06": load_notebook06_package(project_root),
        "07": load_notebook07_package(project_root),
        "08": load_notebook08_package(project_root),
    }


def build_package_summary(
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Return one transparent summary row per final notebook package."""

    rows = [package.summary_record() for _, package in sorted(packages.items())]
    return pd.DataFrame(rows)


__all__ = [
    "FinalNotebookPackage",
    "FinalPackageError",
    "build_package_summary",
    "load_all_final_packages",
    "load_notebook04_package",
    "load_notebook05_package",
    "load_notebook06_package",
    "load_notebook07_package",
    "load_notebook08_package",
]
