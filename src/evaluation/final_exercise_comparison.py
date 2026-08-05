"""Common exercise-decision evaluation for Notebooks 06 and 08.

The two notebooks use the same static test observations but expose four distinct
decision paths.  This module aligns them by ``sample_id`` and recomputes the
classification and near-boundary results under one implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.final_artifact_adapters import FinalNotebookPackage


DEFAULT_BOUNDARY_BANDS = (0.001, 0.005, 0.010)


@dataclass(frozen=True, slots=True)
class ExerciseModelSpec:
    """One decision path in the common exercise comparison."""

    model_id: str
    model: str
    source_notebook: str
    probability_column: str
    threshold: float
    evaluation_role: str
    source_selected: bool


MODEL_NAME_TO_ID = {
    "Exercise-only classifier": "exercise_only_classifier",
    "Multi-task model": "multitask_exercise_head",
}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _finite_probability(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        invalid = int((~np.isfinite(values)).sum())
        raise ValueError(f"{label} contains {invalid} non-finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{label} contains probabilities outside [0, 1]")
    return values


def _binary_target(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded) or not np.isin(rounded, [0.0, 1.0]).all():
        raise ValueError(f"{label} must contain binary zero/one labels")
    return rounded.astype(np.int8)


def _extract_scratch_thresholds(nb08: Mapping[str, Any]) -> tuple[float, float]:
    scratch = nb08.get("scratch_benchmark") or {}
    classification = scratch.get("classification") or {}
    exercise_threshold = classification.get("threshold")
    continuation_threshold = None
    for record in scratch.get("boundary") or []:
        path = str(record.get("decision_path", "")).lower()
        if "continuation" in path and record.get("threshold") is not None:
            continuation_threshold = record.get("threshold")
            break
    selection = (nb08.get("selection") or {}).get("scratch_selection") or {}
    if exercise_threshold is None:
        exercise_threshold = selection.get("exercise_threshold")
    if continuation_threshold is None:
        continuation_threshold = selection.get("continuation_threshold")
    if exercise_threshold is None or continuation_threshold is None:
        raise ValueError("Notebook 08 scratch thresholds are missing from final package")
    return float(exercise_threshold), float(continuation_threshold)


def build_exercise_model_specs(
    packages: Mapping[str, FinalNotebookPackage],
) -> tuple[ExerciseModelSpec, ...]:
    """Return specialist, warm-start deployment, and scratch decision paths."""

    for notebook in ("06", "08"):
        if notebook not in packages:
            raise KeyError(f"Missing Notebook {notebook} final package")

    nb06 = packages["06"].final_metrics
    nb08 = packages["08"].final_metrics
    nb06_thresholds = nb06.get("thresholds") or {}
    warm_thresholds = nb08.get("thresholds") or {}
    scratch_exercise, scratch_continuation = _extract_scratch_thresholds(nb08)

    specs = (
        ExerciseModelSpec(
            "exercise_only_classifier",
            "Exercise-only classifier",
            "06",
            "probability__exercise_only_classifier",
            float(nb06_thresholds["exercise_classifier"]),
            "specialist exercise model",
            True,
        ),
        ExerciseModelSpec(
            "multitask_exercise_head",
            "Multi-task exercise head",
            "06",
            "probability__multitask_exercise_head",
            float(nb06_thresholds["multitask"]),
            "joint price-decision specialist",
            True,
        ),
        ExerciseModelSpec(
            "integrated_warm_start_exercise_head",
            "Integrated warm-start exercise head",
            "08",
            "probability__integrated_warm_start_exercise_head",
            float(warm_thresholds["exercise_head"]),
            "preferred in-domain combined deployment path",
            True,
        ),
        ExerciseModelSpec(
            "integrated_warm_start_continuation_path",
            "Integrated warm-start continuation-implied decision",
            "08",
            "probability__integrated_warm_start_continuation_path",
            float(warm_thresholds["continuation_path"]),
            "decision inferred from deployment continuation value",
            False,
        ),
        ExerciseModelSpec(
            "integrated_scratch_exercise_head",
            "Integrated balanced-scratch exercise head",
            "08_scratch",
            "probability__integrated_scratch_exercise_head",
            scratch_exercise,
            "controlled scratch robustness benchmark",
            False,
        ),
        ExerciseModelSpec(
            "integrated_scratch_continuation_path",
            "Integrated balanced-scratch continuation-implied decision",
            "08_scratch",
            "probability__integrated_scratch_continuation_path",
            scratch_continuation,
            "scratch continuation benchmark",
            False,
        ),
    )
    for spec in specs:
        if not 0.0 < spec.threshold < 1.0:
            raise ValueError(f"Invalid threshold for {spec.model}: {spec.threshold}")
    return specs

def build_exercise_prediction_matrix(
    packages: Mapping[str, FinalNotebookPackage],
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> pd.DataFrame:
    """Align Notebook 06 and Notebook 08 decision outputs on ``sample_id``."""

    specs = build_exercise_model_specs(packages)
    nb06 = packages["06"].load_test_predictions().copy()
    nb08 = packages["08"].load_test_predictions().copy()
    nb08_scratch = packages["08"].load_benchmark_test_predictions().copy()

    _require_columns(
        nb06,
        (
            "sample_id",
            "exercise_now",
            "classifier_probability",
            "multitask_probability",
            "boundary_distance_normalized",
        ),
        label="Notebook 06 predictions",
    )
    for label, frame in (("Notebook 08 deployment predictions", nb08), ("Notebook 08 scratch predictions", nb08_scratch)):
        _require_columns(
            frame,
            (
                "sample_id",
                "exercise_target",
                "exercise_probability",
                "continuation_exercise_probability",
            ),
            label=label,
        )

    for notebook, frame in (("06", nb06), ("08", nb08), ("08_scratch", nb08_scratch)):
        if frame["sample_id"].isna().any():
            raise ValueError(f"Notebook {notebook} contains missing sample IDs")
        duplicates = int(frame["sample_id"].duplicated().sum())
        if duplicates:
            raise ValueError(
                f"Notebook {notebook} contains {duplicates} duplicate sample IDs"
            )

    nb06 = nb06.sort_values("sample_id").reset_index(drop=True)
    nb08 = nb08.sort_values("sample_id").reset_index(drop=True)
    nb08_scratch = nb08_scratch.sort_values("sample_id").reset_index(drop=True)
    warm = nb08[["sample_id", "exercise_target", "exercise_probability", "continuation_exercise_probability"]].rename(columns={
        "exercise_target": "warm_exercise_target",
        "exercise_probability": "warm_exercise_probability",
        "continuation_exercise_probability": "warm_continuation_probability",
    })
    scratch = nb08_scratch[["sample_id", "exercise_target", "exercise_probability", "continuation_exercise_probability"]].rename(columns={
        "exercise_target": "scratch_exercise_target",
        "exercise_probability": "scratch_exercise_probability",
        "continuation_exercise_probability": "scratch_continuation_probability",
    })
    merged = nb06.merge(warm, on="sample_id", how="outer", validate="one_to_one", indicator="_warm_merge")
    if not merged["_warm_merge"].eq("both").all():
        raise ValueError(f"Notebook 06/08 deployment exercise samples do not align: {merged['_warm_merge'].value_counts().to_dict()}")
    merged = merged.drop(columns="_warm_merge").merge(scratch, on="sample_id", how="outer", validate="one_to_one", indicator="_scratch_merge")
    if not merged["_scratch_merge"].eq("both").all():
        raise ValueError(f"Notebook 06/08 scratch exercise samples do not align: {merged['_scratch_merge'].value_counts().to_dict()}")
    merged = merged.drop(columns="_scratch_merge")

    target06 = _binary_target(merged["exercise_now"], label="Notebook 06 target")
    target08 = _binary_target(merged["warm_exercise_target"], label="Notebook 08 deployment target")
    target08_scratch = _binary_target(merged["scratch_exercise_target"], label="Notebook 08 scratch target")
    for label, candidate in (("deployment", target08), ("scratch", target08_scratch)):
        if not np.array_equal(target06, candidate):
            mismatches = int(np.sum(target06 != candidate))
            raise ValueError(f"Notebook 06 and Notebook 08 {label} targets differ on {mismatches} rows")

    matrix = pd.DataFrame(
        {
            "sample_id": merged["sample_id"].to_numpy(),
            "exercise_target": target06,
            "boundary_distance_normalized": pd.to_numeric(
                merged["boundary_distance_normalized"], errors="coerce"
            ),
            "probability__exercise_only_classifier": merged[
                "classifier_probability"
            ],
            "probability__multitask_exercise_head": merged[
                "multitask_probability"
            ],
            "probability__integrated_warm_start_exercise_head": merged["warm_exercise_probability"],
            "probability__integrated_warm_start_continuation_path": merged["warm_continuation_probability"],
            "probability__integrated_scratch_exercise_head": merged["scratch_exercise_probability"],
            "probability__integrated_scratch_continuation_path": merged["scratch_continuation_probability"],
        }
    )
    if "signed_boundary_margin" in merged.columns:
        matrix["signed_boundary_margin"] = pd.to_numeric(
            merged["signed_boundary_margin"], errors="coerce"
        )
    else:
        matrix["signed_boundary_margin"] = np.where(
            target06.astype(bool),
            matrix["boundary_distance_normalized"],
            -matrix["boundary_distance_normalized"],
        )

    for column in (
        "boundary_distance_normalized",
        "signed_boundary_margin",
    ):
        values = pd.to_numeric(matrix[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Exercise matrix column {column!r} is incomplete")

    for spec in specs:
        _finite_probability(matrix[spec.probability_column], label=spec.model)
        matrix[f"prediction__{spec.model_id}"] = (
            pd.to_numeric(matrix[spec.probability_column], errors="raise")
            >= spec.threshold
        )
    return matrix.sort_values("sample_id").reset_index(drop=True)


def _classification_record(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    positive = target == 1
    negative = ~positive

    tp = int(np.sum(prediction & positive))
    tn = int(np.sum((~prediction) & negative))
    fp = int(np.sum(prediction & negative))
    fn = int(np.sum((~prediction) & positive))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    unique = np.unique(target)
    roc_auc = float(roc_auc_score(target, probability)) if len(unique) == 2 else np.nan
    pr_auc = (
        float(average_precision_score(target, probability))
        if len(unique) == 2
        else np.nan
    )
    return {
        "observations": int(len(target)),
        "positive_rate": float(np.mean(positive)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(target)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def build_exercise_model_metrics(
    prediction_matrix: pd.DataFrame,
    specs: Sequence[ExerciseModelSpec],
) -> pd.DataFrame:
    """Recompute full-test metrics for all four decision paths."""

    target = _binary_target(
        prediction_matrix["exercise_target"], label="exercise target"
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        probability = _finite_probability(
            prediction_matrix[spec.probability_column], label=spec.model
        )
        rows.append(
            {
                **asdict(spec),
                **_classification_record(
                    target,
                    probability,
                    threshold=spec.threshold,
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["f1", "balanced_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "exercise_rank", np.arange(1, len(table) + 1))
    return table


def build_exercise_boundary_metrics(
    prediction_matrix: pd.DataFrame,
    specs: Sequence[ExerciseModelSpec],
    *,
    bands: Sequence[float] = DEFAULT_BOUNDARY_BANDS,
) -> pd.DataFrame:
    """Recompute cumulative near-boundary classification and decision regret."""

    target_all = _binary_target(
        prediction_matrix["exercise_target"], label="exercise target"
    )
    distance = pd.to_numeric(
        prediction_matrix["boundary_distance_normalized"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    margin = pd.to_numeric(
        prediction_matrix["signed_boundary_margin"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(distance).all() or not np.isfinite(margin).all():
        raise ValueError("Boundary metrics require finite distance and margin values")

    rows: list[dict[str, Any]] = []
    for band in bands:
        limit = float(band)
        if limit <= 0.0:
            raise ValueError("Boundary bands must be positive")
        positions = np.flatnonzero(distance <= limit)
        if len(positions) == 0:
            continue
        target = target_all[positions]
        for spec in specs:
            probability = _finite_probability(
                prediction_matrix[spec.probability_column], label=spec.model
            )[positions]
            record = _classification_record(
                target,
                probability,
                threshold=spec.threshold,
            )
            prediction = probability >= spec.threshold
            wrong = prediction != target.astype(bool)
            regret = np.where(wrong, np.abs(margin[positions]), 0.0)
            record.update(
                {
                    "boundary_limit": limit,
                    "boundary_band": f"≤{limit:.3f}",
                    "decision_errors": int(wrong.sum()),
                    "normalized_total_regret": float(regret.sum()),
                    "normalized_mean_regret_all": float(regret.mean()),
                    "normalized_mean_regret_when_wrong": (
                        float(regret[wrong].mean()) if wrong.any() else 0.0
                    ),
                    "normalized_maximum_regret": float(regret.max()),
                }
            )
            rows.append({**asdict(spec), **record})
    return pd.DataFrame(rows).sort_values(
        ["boundary_limit", "f1", "model"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _normalize_ood_name(value: Any) -> str:
    name = str(value)
    for prefix in ("american_put_ood_", "ood_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in (".parquet", ".csv", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _prefixed_metric(record: Mapping[str, Any], prefix: str, metric: str) -> Any:
    candidates = (
        f"{prefix}_{metric}",
        f"{prefix}_classification_{metric}",
        f"{prefix}_decision_{metric}",
    )
    for candidate in candidates:
        if candidate in record:
            return record[candidate]
    return None


def build_exercise_ood_comparison(
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Normalize OOD exercise metrics from Notebook 06 and Notebook 08."""

    rows: list[dict[str, Any]] = []
    nb06 = packages["06"].final_metrics
    for record in nb06.get("ood_classification") or []:
        model_id = MODEL_NAME_TO_ID.get(str(record.get("model")))
        if model_id is None:
            continue
        rows.append(
            {
                "ood_set": _normalize_ood_name(record.get("ood_set")),
                "model_id": model_id,
                "model": (
                    "Exercise-only classifier"
                    if model_id == "exercise_only_classifier"
                    else "Multi-task exercise head"
                ),
                "source_notebook": "06",
                **{
                    key: record.get(key)
                    for key in (
                        "observations",
                        "positive_rate",
                        "threshold",
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "brier_score",
                        "roc_auc",
                        "pr_auc",
                    )
                },
            }
        )

    nb08 = packages["08"].final_metrics
    ood_sources = (
        (nb08.get("ood") or [], "integrated_warm_start", "Integrated warm-start", "08"),
        ((nb08.get("scratch_benchmark") or {}).get("ood") or [], "integrated_scratch", "Integrated balanced-scratch", "08_scratch"),
    )
    for records, id_prefix, label_prefix, source in ood_sources:
        for record in records:
            ood_set = _normalize_ood_name(record.get("component", record.get("ood_set")))
            for metric_prefix, suffix, label in (
                ("exercise_head", "exercise_head", "exercise head"),
                ("continuation_path", "continuation_path", "continuation-implied decision"),
            ):
                row = {
                    "ood_set": ood_set,
                    "model_id": f"{id_prefix}_{suffix}",
                    "model": f"{label_prefix} {label}",
                    "source_notebook": source,
                    "observations": record.get("observations"),
                }
                for metric in (
                    "positive_rate", "threshold", "accuracy", "balanced_accuracy",
                    "precision", "recall", "f1", "brier_score", "roc_auc", "pr_auc",
                    "mean_regret_all", "mean_regret_when_wrong", "maximum_regret", "total_regret",
                ):
                    row[metric] = _prefixed_metric(record, metric_prefix, metric)
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if table[["ood_set", "model_id"]].duplicated().any():
        raise ValueError("Duplicate exercise OOD model/regime rows")
    return table.sort_values(["ood_set", "model_id"]).reset_index(drop=True)


def run_phase_5_exercise_comparison(
    packages: Mapping[str, FinalNotebookPackage],
) -> dict[str, pd.DataFrame]:
    """Build the complete common exercise evidence package."""

    specs = build_exercise_model_specs(packages)
    matrix = build_exercise_prediction_matrix(packages)
    return {
        "exercise_model_registry": pd.DataFrame([asdict(spec) for spec in specs]),
        "exercise_prediction_matrix": matrix,
        "exercise_model_metrics": build_exercise_model_metrics(matrix, specs),
        "exercise_boundary_metrics": build_exercise_boundary_metrics(matrix, specs),
        "exercise_ood_comparison": build_exercise_ood_comparison(packages),
    }


def assert_exercise_evidence_ready(results: Mapping[str, pd.DataFrame]) -> None:
    """Fail fast when common exercise evidence is incomplete or inconsistent."""

    required = (
        "exercise_model_registry",
        "exercise_prediction_matrix",
        "exercise_model_metrics",
        "exercise_boundary_metrics",
        "exercise_ood_comparison",
    )
    missing = [name for name in required if name not in results]
    if missing:
        raise RuntimeError(f"Missing exercise evidence tables: {missing}")
    for name in required[:-1]:
        table = results[name]
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise RuntimeError(f"Exercise evidence table {name!r} is empty")

    metrics = results["exercise_model_metrics"]
    expected_ids = {
        "exercise_only_classifier",
        "multitask_exercise_head",
        "integrated_warm_start_exercise_head",
        "integrated_warm_start_continuation_path",
        "integrated_scratch_exercise_head",
        "integrated_scratch_continuation_path",
    }
    if set(metrics["model_id"]) != expected_ids:
        raise RuntimeError("Exercise model comparison is missing one or more decision paths")
    numeric = metrics[
        [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("Exercise metrics contain non-finite core values")


__all__ = [
    "ExerciseModelSpec",
    "assert_exercise_evidence_ready",
    "build_exercise_boundary_metrics",
    "build_exercise_model_metrics",
    "build_exercise_model_specs",
    "build_exercise_ood_comparison",
    "build_exercise_prediction_matrix",
    "run_phase_5_exercise_comparison",
]
