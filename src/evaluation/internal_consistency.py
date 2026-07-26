"""Internal financial and cross-head consistency checks for the final model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "predicted_normalized_american_price",
    "predicted_direct_normalized_american_price",
    "predicted_normalized_continuation_value",
    "exercise_probability",
    "continuation_exercise_probability",
    "normalized_european_price",
    "normalized_intrinsic_value",
)


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    numeric = frame.loc[:, REQUIRED_PREDICTION_COLUMNS].to_numpy(dtype=np.float64)
    if len(frame) == 0:
        raise ValueError("Prediction frame cannot be empty.")
    if not np.isfinite(numeric).all():
        raise ValueError("Prediction frame contains NaN or infinite values.")
    probabilities = frame.loc[
        :,
        ["exercise_probability", "continuation_exercise_probability"],
    ].to_numpy(dtype=np.float64)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Exercise probabilities must lie between zero and one.")


def contradictory_output_flags(
    frame: pd.DataFrame,
    *,
    classification_threshold: float = 0.5,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return row-level flags for financially or internally contradictory outputs."""

    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie between zero and one.")
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative.")
    _validate_prediction_frame(frame)

    result = pd.DataFrame(index=frame.index)
    constrained = frame["predicted_normalized_american_price"].to_numpy(
        dtype=np.float64
    )
    direct = frame[
        "predicted_direct_normalized_american_price"
    ].to_numpy(dtype=np.float64)
    european = frame["normalized_european_price"].to_numpy(dtype=np.float64)
    intrinsic = frame["normalized_intrinsic_value"].to_numpy(dtype=np.float64)
    class_decision = (
        frame["exercise_probability"].to_numpy(dtype=np.float64)
        >= classification_threshold
    )
    continuation_decision = (
        frame["continuation_exercise_probability"].to_numpy(dtype=np.float64)
        >= classification_threshold
    )

    result["constrained_negative"] = constrained < -tolerance
    result["constrained_below_european"] = constrained < european - tolerance
    result["constrained_below_intrinsic"] = constrained < intrinsic - tolerance
    result["direct_negative"] = direct < -tolerance
    result["direct_below_european"] = direct < european - tolerance
    result["direct_below_intrinsic"] = direct < intrinsic - tolerance
    result["decision_disagreement"] = class_decision != continuation_decision
    result["any_contradiction"] = result.any(axis=1)
    return result


def internal_consistency_metrics(
    frame: pd.DataFrame,
    *,
    classification_threshold: float = 0.5,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Summarize price-head agreement, decision agreement, and bound violations."""

    _validate_prediction_frame(frame)
    flags = contradictory_output_flags(
        frame,
        classification_threshold=classification_threshold,
        tolerance=tolerance,
    )
    direct_gap = (
        frame["predicted_direct_normalized_american_price"].to_numpy(
            dtype=np.float64
        )
        - frame["predicted_normalized_american_price"].to_numpy(
            dtype=np.float64
        )
    )
    probability_gap = (
        frame["exercise_probability"].to_numpy(dtype=np.float64)
        - frame["continuation_exercise_probability"].to_numpy(
            dtype=np.float64
        )
    )
    result: dict[str, float] = {
        "observations": float(len(frame)),
        "direct_constrained_mae": float(np.abs(direct_gap).mean()),
        "direct_constrained_rmse": float(
            math.sqrt(np.square(direct_gap).mean())
        ),
        "direct_constrained_max_absolute_gap": float(np.abs(direct_gap).max()),
        "exercise_probability_mae": float(np.abs(probability_gap).mean()),
        "exercise_probability_rmse": float(
            math.sqrt(np.square(probability_gap).mean())
        ),
        "decision_disagreement_rate": float(
            flags["decision_disagreement"].mean()
        ),
        "constrained_negative_rate": float(flags["constrained_negative"].mean()),
        "constrained_below_european_rate": float(
            flags["constrained_below_european"].mean()
        ),
        "constrained_below_intrinsic_rate": float(
            flags["constrained_below_intrinsic"].mean()
        ),
        "direct_negative_rate": float(flags["direct_negative"].mean()),
        "direct_below_european_rate": float(
            flags["direct_below_european"].mean()
        ),
        "direct_below_intrinsic_rate": float(
            flags["direct_below_intrinsic"].mean()
        ),
        "any_contradiction_rate": float(flags["any_contradiction"].mean()),
    }

    if {
        "predicted_floor_residual",
        "normalized_european_price",
        "normalized_intrinsic_value",
    }.issubset(frame.columns):
        reconstructed = np.maximum(
            frame["normalized_european_price"].to_numpy(dtype=np.float64),
            frame["normalized_intrinsic_value"].to_numpy(dtype=np.float64),
        ) + frame["predicted_floor_residual"].to_numpy(dtype=np.float64)
        constrained = frame["predicted_normalized_american_price"].to_numpy(
            dtype=np.float64
        )
        result["residual_reconstruction_mae"] = float(
            np.abs(reconstructed - constrained).mean()
        )
    return result


def consistency_by_segment(
    frame: pd.DataFrame,
    *,
    segment_column: str,
    classification_threshold: float = 0.5,
) -> pd.DataFrame:
    """Calculate internal-consistency metrics separately by segment."""

    if segment_column not in frame:
        raise ValueError(f"Missing segment column {segment_column!r}.")
    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby(segment_column, observed=True, sort=True):
        rows.append(
            {
                "segment": segment,
                **internal_consistency_metrics(
                    group,
                    classification_threshold=classification_threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "REQUIRED_PREDICTION_COLUMNS",
    "consistency_by_segment",
    "contradictory_output_flags",
    "internal_consistency_metrics",
]
