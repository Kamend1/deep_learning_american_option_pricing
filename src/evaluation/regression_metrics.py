"""Regression metrics and segmented error analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    tolerance_bands: Sequence[float] = (0.001, 0.005, 0.01, 0.05),
) -> dict[str, float]:
    """Calculate robust normalized-price regression metrics."""

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("Truth and prediction must have equal non-zero length.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics cannot be calculated with non-finite values.")

    error = prediction - truth
    absolute = np.abs(error)
    squared = error**2
    result = {
        "observations": float(len(truth)),
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(squared.mean())),
        "median_absolute_error": float(np.median(absolute)),
        "max_absolute_error": float(absolute.max()),
        "mean_error": float(error.mean()),
        "r2": float(r2_score(truth, prediction)),
    }
    for threshold in tolerance_bands:
        key = f"within_{threshold:g}"
        result[key] = float((absolute <= threshold).mean())
    return result


def compare_models(
    y_true: Sequence[float] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, values in predictions.items():
        rows.append({"model": name, **regression_metrics(y_true, values)})
    return pd.DataFrame(rows).set_index("model")


def segmented_regression_metrics(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    segment_column: str,
) -> pd.DataFrame:
    """Calculate the same metrics independently for each named segment."""

    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby(segment_column, observed=True, sort=True):
        metrics = regression_metrics(
            group[actual_column].to_numpy(),
            group[prediction_column].to_numpy(),
        )
        rows.append({"segment": segment, **metrics})
    return pd.DataFrame(rows)


__all__ = [
    "compare_models",
    "regression_metrics",
    "segmented_regression_metrics",
]
