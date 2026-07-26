"""Target preparation and validation for the final integrated static model."""

from __future__ import annotations

import numpy as np
import pandas as pd


INTEGRATED_TARGET_COLUMNS: tuple[str, ...] = (
    "normalized_european_price",
    "normalized_intrinsic_value",
    "normalized_american_price",
    "normalized_continuation_value",
    "normalized_financial_floor",
    "normalized_floor_residual",
    "exercise_now",
)


def add_integrated_targets(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
    exercise_tolerance: float = 1e-12,
    validate_existing_exercise_labels: bool = True,
) -> pd.DataFrame:
    """Add normalized targets required by the final multi-head model.

    The function can work with the canonical production schema or a frame that
    already contains some normalized columns. Setting ``copy=False`` is useful
    for the full 1.25 million in-domain observations because it avoids an extra
    full-frame copy.
    """

    if exercise_tolerance < 0.0:
        raise ValueError("exercise_tolerance cannot be negative.")
    result = frame.copy() if copy else frame

    required_raw = [
        "strike",
        "intrinsic_value",
        "continuation_value",
        "european_price",
        "american_price",
    ]
    missing_raw = [column for column in required_raw if column not in result]
    if missing_raw:
        raise ValueError(f"Frame is missing raw pricing columns: {missing_raw}")

    numeric = result.loc[:, required_raw].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Raw pricing columns contain NaN or infinite values.")
    if (result["strike"] <= 0.0).any():
        raise ValueError("strike must be positive for normalization.")

    strike = result["strike"].to_numpy(dtype=np.float64)
    normalized_european = (
        result["european_price"].to_numpy(dtype=np.float64) / strike
    )
    normalized_intrinsic = (
        result["intrinsic_value"].to_numpy(dtype=np.float64) / strike
    )
    normalized_american = (
        result["american_price"].to_numpy(dtype=np.float64) / strike
    )
    normalized_continuation = (
        result["continuation_value"].to_numpy(dtype=np.float64) / strike
    )
    financial_floor = np.maximum(normalized_european, normalized_intrinsic)
    floor_residual = np.maximum(normalized_american - financial_floor, 0.0)

    result["normalized_european_price"] = normalized_european
    result["normalized_intrinsic_value"] = normalized_intrinsic
    result["normalized_american_price"] = normalized_american
    result["normalized_continuation_value"] = normalized_continuation
    result["normalized_financial_floor"] = financial_floor
    result["normalized_floor_residual"] = floor_residual

    derived_exercise = (
        result["intrinsic_value"].to_numpy(dtype=np.float64)
        >= result["continuation_value"].to_numpy(dtype=np.float64)
        - exercise_tolerance
    )
    if "exercise_now" in result.columns and validate_existing_exercise_labels:
        existing = result["exercise_now"].to_numpy(dtype=bool)
        mismatches = int(np.count_nonzero(existing != derived_exercise))
        if mismatches:
            raise ValueError(
                f"exercise_now disagrees with intrinsic-versus-continuation "
                f"logic for {mismatches} observations."
            )
    else:
        result["exercise_now"] = derived_exercise

    validate_integrated_targets(result)
    return result


def validate_integrated_targets(
    frame: pd.DataFrame,
    *,
    tolerance: float = 1e-7,
) -> None:
    """Validate identities, bounds, and exercise labels for integrated targets."""

    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative.")
    missing = [column for column in INTEGRATED_TARGET_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Frame is missing integrated target columns: {missing}")

    values = frame.loc[:, INTEGRATED_TARGET_COLUMNS[:-1]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Integrated targets contain NaN or infinite values.")
    if (values < -tolerance).any():
        raise ValueError("Normalized pricing targets cannot be negative.")

    european = frame["normalized_european_price"].to_numpy(dtype=np.float64)
    intrinsic = frame["normalized_intrinsic_value"].to_numpy(dtype=np.float64)
    american = frame["normalized_american_price"].to_numpy(dtype=np.float64)
    floor = frame["normalized_financial_floor"].to_numpy(dtype=np.float64)
    residual = frame["normalized_floor_residual"].to_numpy(dtype=np.float64)

    if not np.allclose(floor, np.maximum(european, intrinsic), atol=tolerance):
        raise ValueError("normalized_financial_floor identity failed.")
    if not np.allclose(american, floor + residual, atol=tolerance):
        raise ValueError("American price reconstruction identity failed.")
    if np.any(american < european - tolerance):
        raise ValueError("American price is below the European lower bound.")
    if np.any(american < intrinsic - tolerance):
        raise ValueError("American price is below the intrinsic lower bound.")

    exercise = frame["exercise_now"].to_numpy()
    if not np.isin(exercise, [False, True, 0, 1]).all():
        raise ValueError("exercise_now must be binary.")


__all__ = [
    "INTEGRATED_TARGET_COLUMNS",
    "add_integrated_targets",
    "validate_integrated_targets",
]
