"""Classification metrics for American-option exercise decisions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_arrays(
    actual: Iterable[float],
    probability: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(actual, dtype=np.float64).reshape(-1)
    y_probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_probability):
        raise ValueError("Actual and probability arrays must have equal non-zero length.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_probability).all():
        raise ValueError("Classification arrays contain non-finite values.")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError("Actual labels must be binary.")
    if np.any((y_probability < 0.0) | (y_probability > 1.0)):
        raise ValueError("Probabilities must lie between zero and one.")
    return y_true.astype(np.int64), y_probability


def binary_classification_metrics(
    actual: Iterable[float],
    probability: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return thresholded and ranking metrics for exercise prediction."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one.")
    y_true, y_probability = _validated_arrays(actual, probability)
    y_pred = (y_probability >= threshold).astype(np.int64)
    result = {
        "observations": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_probability))
        result["pr_auc"] = float(average_precision_score(y_true, y_probability))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def confusion_matrix_frame(
    actual: Iterable[float],
    probability: Iterable[float],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a labelled 2x2 confusion matrix."""

    y_true, y_probability = _validated_arrays(actual, probability)
    y_pred = (y_probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index(["actual_continue", "actual_exercise"], name="actual"),
        columns=pd.Index(
            ["predicted_continue", "predicted_exercise"],
            name="prediction",
        ),
    )


def choose_f1_threshold(
    actual: Iterable[float],
    probability: Iterable[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    """Choose an F1-maximizing threshold from validation observations only."""

    y_true, y_probability = _validated_arrays(actual, probability)
    grid = np.asarray(
        list(thresholds) if thresholds is not None else np.linspace(0.05, 0.95, 181),
        dtype=np.float64,
    )
    if len(grid) == 0 or np.any((grid <= 0.0) | (grid >= 1.0)):
        raise ValueError("Threshold grid must lie strictly between zero and one.")
    scores = np.asarray(
        [
            f1_score(
                y_true,
                (y_probability >= threshold).astype(np.int64),
                zero_division=0,
            )
            for threshold in grid
        ]
    )
    best_score = float(scores.max())
    candidates = grid[np.isclose(scores, best_score)]
    best_threshold = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return {"threshold": best_threshold, "validation_f1": best_score}


def calibration_frame(
    actual: Iterable[float],
    probability: Iterable[float],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Return reliability-curve coordinates."""

    if bins <= 1:
        raise ValueError("bins must exceed one.")
    y_true, y_probability = _validated_arrays(actual, probability)
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_probability,
        n_bins=bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted,
            "observed_exercise_rate": fraction_positive,
        }
    )


__all__ = [
    "binary_classification_metrics",
    "calibration_frame",
    "choose_f1_threshold",
    "confusion_matrix_frame",
]
