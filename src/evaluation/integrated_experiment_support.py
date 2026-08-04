"""Reusable support for the final integrated static-model experiment.

This module keeps Notebook 08 focused on orchestration. It provides strict
checkpoint reconstruction, dependency fingerprints, two-path boundary and OOD
diagnostics, and compact comparisons with the specialized static models from
Notebooks 05 and 06.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.regression_metrics import regression_metrics
from src.models.integrated_multihead_pricer import (
    IntegratedAmericanPutMultiHeadMLP,
    IntegratedMultiHeadConfig,
)
from src.models.multitask_pricer import (
    ExerciseClassifierConfig,
    ExerciseClassifierMLP,
)
from src.models.premium_pricer import (
    PremiumAmericanPutMLP,
    PremiumMLPConfig,
)
from src.training.artifact_management import canonicalize_json_value
from src.training.dependency_fingerprints import calculate_file_sha256
from src.training.multihead_losses import MultiHeadLossConfig


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside project root: {path}") from exc


def build_integrated_training_dependencies(
    *,
    project_root: str | Path,
    production_manifest_path: str | Path | None,
    feature_columns: Sequence[str],
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    loss_configurations: Mapping[str, Mapping[str, Any]],
    loader_config: Mapping[str, Any],
    seed: int,
    smoke_mode: bool,
) -> dict[str, object]:
    """Build the strict external/configuration dependency record for scratch training."""

    root = Path(project_root).resolve()
    manifest_path = (
        Path(production_manifest_path).resolve()
        if production_manifest_path is not None
        else None
    )
    if manifest_path is None or not manifest_path.is_file():
        if not smoke_mode:
            raise FileNotFoundError(
                "Production manifest is required for full training: "
                f"{manifest_path}"
            )
        manifest_dependency: dict[str, object] = {
            "mode": "synthetic_smoke_data",
        }
    else:
        manifest_dependency = {
            "path": _relative_path(manifest_path, root),
            "sha256": calculate_file_sha256(manifest_path),
        }
    columns = [str(column) for column in feature_columns]
    if not columns:
        raise ValueError("feature_columns cannot be empty.")

    return canonicalize_json_value(
        {
            "production_manifest": manifest_dependency,
            "feature_columns": columns,
            "model_config": dict(model_config),
            "training_config": dict(training_config),
            "loss_configurations": {
                str(name): dict(config)
                for name, config in loss_configurations.items()
            },
            "loader_config": dict(loader_config),
            "seed": int(seed),
            "smoke_mode": bool(smoke_mode),
        }
    )


def build_file_dependency(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, str]:
    """Return a repository-relative path and SHA-256 fingerprint."""

    root = Path(project_root).resolve()
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return {
        "path": _relative_path(file_path, root),
        "sha256": calculate_file_sha256(file_path),
    }


def load_integrated_model_package(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[
    IntegratedAmericanPutMultiHeadMLP,
    dict[str, Any],
    IntegratedMultiHeadConfig,
    MultiHeadLossConfig,
]:
    """Reconstruct an integrated model and loss settings from checkpoint metadata."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(
        path,
        map_location=torch.device(device),
        weights_only=False,
    )
    required = {"model_state_dict", "model_config", "loss_config"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Checkpoint {path.name} is missing metadata: {sorted(missing)}"
        )

    model_config = IntegratedMultiHeadConfig.from_dict(
        checkpoint["model_config"]
    )
    loss_config = MultiHeadLossConfig.from_dict(
        checkpoint["loss_config"]
    )
    model = IntegratedAmericanPutMultiHeadMLP(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, model_config, loss_config


def load_premium_model_package(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[PremiumAmericanPutMLP, dict[str, Any]]:
    """Reconstruct the canonical Notebook 05 premium model from metadata."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(
        path,
        map_location=torch.device(device),
        weights_only=False,
    )
    metadata = checkpoint.get("model_config")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Checkpoint {path.name} has no model_config metadata.")
    required = {
        "input_features",
        "hidden_sizes",
        "batch_norm_after",
        "output_activation",
        "residual_base",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(
            f"Checkpoint {path.name} is missing metadata: {sorted(missing)}"
        )
    config = PremiumMLPConfig(
        input_features=int(metadata["input_features"]),
        hidden_sizes=tuple(metadata["hidden_sizes"]),
        batch_norm_after=tuple(metadata["batch_norm_after"]),
        output_activation=str(metadata["output_activation"]),
        residual_base=str(metadata["residual_base"]),
    )
    model = PremiumAmericanPutMLP(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_exercise_classifier_package(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[ExerciseClassifierMLP, dict[str, Any]]:
    """Reconstruct the canonical Notebook 06 exercise classifier."""

    checkpoint_file = Path(checkpoint_path)
    config_file = Path(config_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    config = ExerciseClassifierConfig(
        input_features=int(raw_config["input_features"]),
        hidden_sizes=tuple(raw_config["hidden_sizes"]),
        batch_norm_after=tuple(raw_config["batch_norm_after"]),
    )
    checkpoint = torch.load(
        checkpoint_file,
        map_location=torch.device(device),
        weights_only=False,
    )
    model = ExerciseClassifierMLP(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _classification_counts(
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    actual = np.asarray(actual, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    if actual.shape != probability.shape or actual.size == 0:
        raise ValueError("actual and probability require equal non-empty shapes.")
    predicted = probability >= threshold
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum((~actual) & (~predicted)))
    fp = int(np.sum((~actual) & predicted))
    fn = int(np.sum(actual & (~predicted)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": float((tp + tn) / len(actual)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_exercise_count": fp,
        "missed_exercise_count": fn,
        "predicted_exercise_count": int(predicted.sum()),
    }


def _decision_regret(
    actual: np.ndarray,
    probability: np.ndarray,
    decision_gap: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    actual_bool = np.asarray(actual, dtype=bool)
    predicted = np.asarray(probability, dtype=np.float64) >= threshold
    gap = np.asarray(decision_gap, dtype=np.float64)
    wrong = actual_bool != predicted
    regret = np.where(wrong, gap, 0.0)
    wrong_regret = regret[wrong]
    return {
        "decision_errors": int(wrong.sum()),
        "mean_regret_all": float(regret.mean()) if len(regret) else np.nan,
        "mean_regret_when_wrong": (
            float(wrong_regret.mean()) if len(wrong_regret) else 0.0
        ),
        "maximum_regret": (
            float(wrong_regret.max()) if len(wrong_regret) else 0.0
        ),
        "total_regret": float(regret.sum()),
    }


def build_integrated_segmented_report(
    frame: pd.DataFrame,
    *,
    exercise_threshold: float,
    continuation_threshold: float,
) -> pd.DataFrame:
    """Summarize pricing and both decision paths across economic segments."""

    required = [
        "moneyness",
        "time_to_maturity",
        "volatility",
        "strike",
        "intrinsic_value",
        "continuation_value",
        "exercise_target",
        "exercise_probability",
        "continuation_exercise_probability",
        "true_normalized_american_price",
        "predicted_normalized_american_price",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Segmented frame is missing columns: {missing}")

    work = frame.copy()
    work["exercise_region"] = np.where(
        work["exercise_target"].astype(bool),
        "exercise",
        "continue",
    )
    work["moneyness_bucket"] = pd.cut(
        work["moneyness"],
        bins=[-np.inf, 0.80, 0.95, 1.05, 1.20, np.inf],
        labels=["deep ITM", "ITM", "near ATM", "OTM", "deep OTM"],
    )
    work["maturity_bucket"] = pd.cut(
        work["time_to_maturity"],
        bins=[-np.inf, 0.50, 1.00, np.inf],
        labels=["short", "medium", "long"],
    )
    work["volatility_bucket"] = pd.cut(
        work["volatility"],
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["low", "medium", "high"],
    )

    rows: list[dict[str, object]] = []
    for segment_type in (
        "exercise_region",
        "moneyness_bucket",
        "maturity_bucket",
        "volatility_bucket",
    ):
        for segment, group in work.groupby(segment_type, observed=True, sort=False):
            actual = group["exercise_target"].to_numpy(dtype=bool)
            exercise_probability = group["exercise_probability"].to_numpy(dtype=float)
            continuation_probability = group[
                "continuation_exercise_probability"
            ].to_numpy(dtype=float)
            normalized_error = (
                group["predicted_normalized_american_price"].to_numpy(dtype=float)
                - group["true_normalized_american_price"].to_numpy(dtype=float)
            )
            price_error = normalized_error * group["strike"].to_numpy(dtype=float)
            decision_gap = np.abs(
                group["intrinsic_value"].to_numpy(dtype=float)
                - group["continuation_value"].to_numpy(dtype=float)
            )
            exercise_metrics = _classification_counts(
                actual,
                exercise_probability,
                threshold=exercise_threshold,
            )
            continuation_metrics = _classification_counts(
                actual,
                continuation_probability,
                threshold=continuation_threshold,
            )
            exercise_regret = _decision_regret(
                actual,
                exercise_probability,
                decision_gap,
                threshold=exercise_threshold,
            )
            continuation_regret = _decision_regret(
                actual,
                continuation_probability,
                decision_gap,
                threshold=continuation_threshold,
            )
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": str(segment),
                    "observations": int(len(group)),
                    "exercise_rate": float(actual.mean()),
                    "normalized_price_mae": float(np.mean(np.abs(normalized_error))),
                    "normalized_price_rmse": float(np.sqrt(np.mean(normalized_error**2))),
                    "price_mae": float(np.mean(np.abs(price_error))),
                    "price_rmse": float(np.sqrt(np.mean(price_error**2))),
                    "exercise_head_f1": exercise_metrics["f1"],
                    "continuation_path_f1": continuation_metrics["f1"],
                    "exercise_head_errors": exercise_regret["decision_errors"],
                    "continuation_path_errors": continuation_regret["decision_errors"],
                    "exercise_head_total_regret": exercise_regret["total_regret"],
                    "continuation_path_total_regret": continuation_regret["total_regret"],
                }
            )
    return pd.DataFrame(rows)


def build_two_path_boundary_report(
    frame: pd.DataFrame,
    *,
    exercise_threshold: float,
    continuation_threshold: float,
    bands: Sequence[float] = (0.001, 0.005, 0.010),
) -> pd.DataFrame:
    """Evaluate both integrated exercise paths in cumulative boundary bands."""

    required = [
        "boundary_distance_normalized",
        "exercise_target",
        "exercise_probability",
        "continuation_exercise_probability",
        "intrinsic_value",
        "continuation_value",
        "strike",
        "true_normalized_american_price",
        "predicted_normalized_american_price",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Boundary frame is missing columns: {missing}")

    rows: list[dict[str, object]] = []
    decision_gap = np.abs(
        frame["intrinsic_value"].to_numpy(dtype=np.float64)
        - frame["continuation_value"].to_numpy(dtype=np.float64)
    )

    for raw_band in bands:
        band = float(raw_band)
        if band <= 0.0:
            raise ValueError("Boundary bands must be positive.")
        mask = frame["boundary_distance_normalized"].to_numpy() <= band
        subset = frame.loc[mask]
        if subset.empty:
            continue
        price_errors = (
            subset["predicted_normalized_american_price"].to_numpy(dtype=float)
            - subset["true_normalized_american_price"].to_numpy(dtype=float)
        )
        price_errors_units = price_errors * subset["strike"].to_numpy(dtype=float)
        actual = subset["exercise_target"].to_numpy(dtype=bool)
        subset_gap = decision_gap[mask]

        paths = (
            ("Exercise head", "exercise_probability", exercise_threshold),
            (
                "Continuation-implied",
                "continuation_exercise_probability",
                continuation_threshold,
            ),
        )
        for model_name, probability_column, threshold in paths:
            probability = subset[probability_column].to_numpy(dtype=float)
            rows.append(
                {
                    "decision_path": model_name,
                    "boundary_band": f"≤{band:.3f}",
                    "boundary_limit": band,
                    "observations": int(len(subset)),
                    "exercise_observations": int(actual.sum()),
                    "continuation_observations": int((~actual).sum()),
                    "threshold": float(threshold),
                    "normalized_price_mae": float(np.mean(np.abs(price_errors))),
                    "normalized_price_rmse": float(
                        np.sqrt(np.mean(price_errors**2))
                    ),
                    "price_mae": float(np.mean(np.abs(price_errors_units))),
                    "price_rmse": float(
                        np.sqrt(np.mean(price_errors_units**2))
                    ),
                    **_classification_counts(
                        actual,
                        probability,
                        threshold=threshold,
                    ),
                    **_decision_regret(
                        actual,
                        probability,
                        subset_gap,
                        threshold=threshold,
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_integrated_ood(
    frame: pd.DataFrame,
    *,
    component: str,
    exercise_threshold: float,
    continuation_threshold: float,
) -> dict[str, object]:
    """Return pricing, classification, consistency, bound, and regret OOD metrics."""

    required = [
        "strike",
        "intrinsic_value",
        "continuation_value",
        "normalized_european_price",
        "normalized_intrinsic_value",
        "exercise_target",
        "exercise_probability",
        "continuation_exercise_probability",
        "true_normalized_american_price",
        "predicted_normalized_american_price",
        "predicted_direct_normalized_american_price",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"OOD frame is missing columns: {missing}")

    actual_price = frame["true_normalized_american_price"].to_numpy(dtype=float)
    constrained = frame["predicted_normalized_american_price"].to_numpy(dtype=float)
    direct = frame["predicted_direct_normalized_american_price"].to_numpy(dtype=float)
    strikes = frame["strike"].to_numpy(dtype=float)
    actual_exercise = frame["exercise_target"].to_numpy(dtype=bool)
    exercise_probability = frame["exercise_probability"].to_numpy(dtype=float)
    continuation_probability = frame[
        "continuation_exercise_probability"
    ].to_numpy(dtype=float)
    decision_gap = np.abs(
        frame["intrinsic_value"].to_numpy(dtype=float)
        - frame["continuation_value"].to_numpy(dtype=float)
    )
    constrained_metrics = regression_metrics(actual_price, constrained)
    direct_metrics = regression_metrics(actual_price, direct)
    constrained_errors_units = (constrained - actual_price) * strikes
    direct_errors_units = (direct - actual_price) * strikes
    financial_floor = np.maximum(
        frame["normalized_european_price"].to_numpy(dtype=float),
        frame["normalized_intrinsic_value"].to_numpy(dtype=float),
    )
    exercise_predicted = exercise_probability >= exercise_threshold
    continuation_predicted = continuation_probability >= continuation_threshold

    return {
        "component": str(component),
        "observations": int(len(frame)),
        "constrained_mae": constrained_metrics["mae"],
        "constrained_rmse": constrained_metrics["rmse"],
        "constrained_median_absolute_error": constrained_metrics[
            "median_absolute_error"
        ],
        "constrained_max_absolute_error": constrained_metrics[
            "max_absolute_error"
        ],
        "constrained_mean_error": constrained_metrics["mean_error"],
        "constrained_price_mae": float(np.mean(np.abs(constrained_errors_units))),
        "constrained_price_rmse": float(
            np.sqrt(np.mean(constrained_errors_units**2))
        ),
        "direct_mae": direct_metrics["mae"],
        "direct_rmse": direct_metrics["rmse"],
        "direct_price_mae": float(np.mean(np.abs(direct_errors_units))),
        "exercise_threshold": float(exercise_threshold),
        "continuation_threshold": float(continuation_threshold),
        **{
            f"exercise_head_{key}": value
            for key, value in _classification_counts(
                actual_exercise,
                exercise_probability,
                threshold=exercise_threshold,
            ).items()
        },
        **{
            f"continuation_path_{key}": value
            for key, value in _classification_counts(
                actual_exercise,
                continuation_probability,
                threshold=continuation_threshold,
            ).items()
        },
        "decision_disagreement_rate": float(
            np.mean(exercise_predicted != continuation_predicted)
        ),
        **{
            f"exercise_head_{key}": value
            for key, value in _decision_regret(
                actual_exercise,
                exercise_probability,
                decision_gap,
                threshold=exercise_threshold,
            ).items()
        },
        **{
            f"continuation_path_{key}": value
            for key, value in _decision_regret(
                actual_exercise,
                continuation_probability,
                decision_gap,
                threshold=continuation_threshold,
            ).items()
        },
        "constrained_negative_rate": float(np.mean(constrained < 0.0)),
        "constrained_below_european_rate": float(
            np.mean(
                constrained
                < frame["normalized_european_price"].to_numpy(dtype=float)
            )
        ),
        "constrained_below_intrinsic_rate": float(
            np.mean(
                constrained
                < frame["normalized_intrinsic_value"].to_numpy(dtype=float)
            )
        ),
        "direct_negative_rate": float(np.mean(direct < 0.0)),
        "direct_below_european_rate": float(
            np.mean(
                direct
                < frame["normalized_european_price"].to_numpy(dtype=float)
            )
        ),
        "direct_below_intrinsic_rate": float(
            np.mean(
                direct
                < frame["normalized_intrinsic_value"].to_numpy(dtype=float)
            )
        ),
        "direct_below_financial_floor_rate": float(
            np.mean(direct < financial_floor)
        ),
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {file_path}")
    return value


def _find_record(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str,
    value: str,
) -> Mapping[str, Any]:
    for record in records:
        if str(record.get(key)) == value:
            return record
    raise KeyError(f"Could not find {key}={value!r}")


def build_static_model_comparison(
    *,
    premium_final_metrics_path: str | Path,
    multitask_final_metrics_path: str | Path,
    integrated_metrics: Mapping[str, float],
    integrated_financial_violations: int,
    strike: float = 100.0,
) -> pd.DataFrame:
    """Compare the integrated model with the strongest specialized static models."""

    premium = _read_json(premium_final_metrics_path)
    multitask = _read_json(multitask_final_metrics_path)
    premium_pricing = premium.get("pricing", [])
    multitask_pricing = multitask.get("pricing", [])
    multitask_classification = multitask.get("classification", [])
    multitask_financial = multitask.get("financial_consistency", [])

    direct = _find_record(
        premium_pricing,
        key="model",
        value="Direct MLP",
    )
    constrained = _find_record(
        premium_pricing,
        key="model",
        value="Constrained floor residual",
    )
    multitask_price = _find_record(
        multitask_pricing,
        key="model",
        value="Multi-task constrained residual",
    )
    classifier = _find_record(
        multitask_classification,
        key="model",
        value="Exercise-only classifier",
    )
    multitask_classifier = _find_record(
        multitask_classification,
        key="model",
        value="Multi-task model",
    )

    violation_map = {
        str(record.get("model")): int(
            record.get("below_financial_floor_count", 0)
        )
        for record in multitask_financial
    }

    rows = [
        {
            "model": "Notebook 04 direct MLP",
            "pricing_role": "specialized price",
            "normalized_price_mae": float(direct["mae"]),
            "price_mae": float(direct["mae"]) * strike,
            "exercise_f1": np.nan,
            "financial_bound_violations": int(
                direct.get("total_bound_violations", 0)
            ),
        },
        {
            "model": "Notebook 05 constrained residual",
            "pricing_role": "authoritative price",
            "normalized_price_mae": float(constrained["mae"]),
            "price_mae": float(constrained["mae"]) * strike,
            "exercise_f1": np.nan,
            "financial_bound_violations": int(
                constrained.get("total_bound_violations", 0)
            ),
        },
        {
            "model": "Notebook 06 exercise classifier",
            "pricing_role": "specialized decision",
            "normalized_price_mae": np.nan,
            "price_mae": np.nan,
            "exercise_f1": float(classifier["f1"]),
            "financial_bound_violations": np.nan,
        },
        {
            "model": "Notebook 06 multi-task model",
            "pricing_role": "joint price and decision",
            "normalized_price_mae": float(multitask_price["mae"]),
            "price_mae": float(multitask_price["mae"]) * strike,
            "exercise_f1": float(multitask_classifier["f1"]),
            "financial_bound_violations": violation_map.get(
                "Multi-task constrained residual",
                0,
            ),
        },
        {
            "model": "Notebook 08 integrated model",
            "pricing_role": "four-head integrated",
            "normalized_price_mae": float(integrated_metrics["constrained_mae"]),
            "price_mae": float(integrated_metrics["constrained_mae"]) * strike,
            "exercise_f1": float(integrated_metrics["exercise_f1"]),
            "financial_bound_violations": int(integrated_financial_violations),
        },
    ]
    return pd.DataFrame(rows).set_index("model")


def _normalize_ood_name(value: object) -> str:
    name = str(value)
    return name.removeprefix("american_put_ood_")


def build_ood_specialized_comparison(
    *,
    integrated_ood: pd.DataFrame,
    premium_final_metrics_path: str | Path,
    multitask_final_metrics_path: str | Path,
) -> pd.DataFrame:
    """Align Notebook 08 OOD results with specialized price/classifier baselines."""

    if integrated_ood.empty:
        return pd.DataFrame(
            columns=[
                "notebook05_price_mae",
                "notebook08_price_mae",
                "notebook06_classifier_f1",
                "notebook06_multitask_f1",
                "notebook08_exercise_f1",
                "notebook08_continuation_f1",
                "notebook08_disagreement_rate",
            ],
            index=pd.Index([], name="ood_set"),
        )

    premium = _read_json(premium_final_metrics_path)
    multitask = _read_json(multitask_final_metrics_path)
    premium_rows = premium.get("ood", [])
    classification_rows = multitask.get("ood_classification", [])

    premium_map: dict[str, float] = {}
    for record in premium_rows:
        if str(record.get("model")) == "Constrained floor residual":
            premium_map[_normalize_ood_name(record.get("ood_set"))] = float(
                record["mae"]
            )

    classifier_map: dict[str, float] = {}
    multitask_map: dict[str, float] = {}
    for record in classification_rows:
        name = _normalize_ood_name(record.get("ood_set"))
        model = str(record.get("model"))
        if model == "Exercise-only classifier":
            classifier_map[name] = float(record["f1"])
        elif model == "Multi-task model":
            multitask_map[name] = float(record["f1"])

    rows: list[dict[str, object]] = []
    for record in integrated_ood.to_dict(orient="records"):
        name = _normalize_ood_name(record["component"])
        rows.append(
            {
                "ood_set": name,
                "notebook05_price_mae": premium_map.get(name, np.nan),
                "notebook08_price_mae": float(record["constrained_mae"]),
                "notebook06_classifier_f1": classifier_map.get(name, np.nan),
                "notebook06_multitask_f1": multitask_map.get(name, np.nan),
                "notebook08_exercise_f1": float(record["exercise_head_f1"]),
                "notebook08_continuation_f1": float(
                    record["continuation_path_f1"]
                ),
                "notebook08_disagreement_rate": float(
                    record["decision_disagreement_rate"]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("ood_set")


__all__ = [
    "build_file_dependency",
    "build_integrated_segmented_report",
    "build_integrated_training_dependencies",
    "build_ood_specialized_comparison",
    "build_static_model_comparison",
    "build_two_path_boundary_report",
    "load_exercise_classifier_package",
    "load_integrated_model_package",
    "load_premium_model_package",
    "summarize_integrated_ood",
]
