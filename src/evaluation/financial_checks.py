"""Financial-consistency diagnostics for predicted American put prices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def financial_bound_report(
    frame: pd.DataFrame,
    *,
    normalized_prediction_column: str,
    strike_column: str = "strike",
    intrinsic_column: str = "intrinsic_value",
    european_column: str = "european_price",
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Measure non-negativity, intrinsic, and European lower-bound violations."""

    required = [
        normalized_prediction_column,
        strike_column,
        intrinsic_column,
        european_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Financial-check frame is missing columns: {missing}")

    predicted_price = (
        frame[normalized_prediction_column].to_numpy(dtype=np.float64)
        * frame[strike_column].to_numpy(dtype=np.float64)
    )
    intrinsic = frame[intrinsic_column].to_numpy(dtype=np.float64)
    european = frame[european_column].to_numpy(dtype=np.float64)

    checks = {
        "negative_price": predicted_price < -tolerance,
        "below_intrinsic": predicted_price < intrinsic - tolerance,
        "below_european": predicted_price < european - tolerance,
    }
    rows = []
    for name, mask in checks.items():
        magnitude = np.zeros_like(predicted_price)
        if name == "negative_price":
            magnitude = np.maximum(-predicted_price, 0.0)
        elif name == "below_intrinsic":
            magnitude = np.maximum(intrinsic - predicted_price, 0.0)
        elif name == "below_european":
            magnitude = np.maximum(european - predicted_price, 0.0)
        rows.append(
            {
                "check": name,
                "violations": int(mask.sum()),
                "violation_rate": float(mask.mean()),
                "max_violation": float(magnitude.max()),
                "mean_violation_when_failed": (
                    float(magnitude[mask].mean()) if mask.any() else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def monotonicity_violation_rate(
    x_values: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
    *,
    expected_direction: str,
    tolerance: float = 1e-8,
) -> float:
    """Measure adjacent monotonicity failures on an ordered one-dimensional grid."""

    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Monotonicity inputs must have equal length of at least two.")
    order = np.argsort(x)
    differences = np.diff(y[order])
    if expected_direction == "increasing":
        return float((differences < -tolerance).mean())
    if expected_direction == "decreasing":
        return float((differences > tolerance).mean())
    raise ValueError("expected_direction must be 'increasing' or 'decreasing'.")


__all__ = ["financial_bound_report", "monotonicity_violation_rate"]
