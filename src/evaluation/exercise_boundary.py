"""Exercise-boundary diagnostics and H4 decision logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import binary_classification_metrics


@dataclass(frozen=True, slots=True)
class BoundaryEstimate:
    """Estimated exercise boundary for one parameter slice."""

    boundary_moneyness: float
    crossing_found: bool
    threshold: float
    minimum_moneyness: float
    maximum_moneyness: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class H4Decision:
    hypothesis: str
    decision: str
    rationale: str
    evidence: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_boundary_distance_normalized(
    intrinsic_value: Iterable[float],
    continuation_value: Iterable[float],
    strike: Iterable[float] | float,
) -> np.ndarray:
    """Calculate (intrinsic - continuation) / strike."""

    intrinsic = np.asarray(intrinsic_value, dtype=np.float64)
    continuation = np.asarray(continuation_value, dtype=np.float64)
    strike_values = np.asarray(strike, dtype=np.float64)
    try:
        intrinsic, continuation, strike_values = np.broadcast_arrays(
            intrinsic,
            continuation,
            strike_values,
        )
    except ValueError as exc:
        raise ValueError("Boundary-distance inputs are not broadcast compatible.") from exc
    if not (
        np.isfinite(intrinsic).all()
        and np.isfinite(continuation).all()
        and np.isfinite(strike_values).all()
    ):
        raise ValueError("Boundary-distance inputs contain non-finite values.")
    if np.any(strike_values <= 0.0):
        raise ValueError("strike must be positive.")
    return (intrinsic - continuation) / strike_values


def validate_exercise_labels(
    frame: pd.DataFrame,
    *,
    intrinsic_column: str = "intrinsic_value",
    continuation_column: str = "continuation_value",
    exercise_column: str = "exercise_now",
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Compare stored exercise labels with the numerical decision rule."""

    required = [intrinsic_column, continuation_column, exercise_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing columns: {missing}")
    expected = (
        frame[intrinsic_column].to_numpy(dtype=np.float64)
        + tolerance
        >= frame[continuation_column].to_numpy(dtype=np.float64)
    )
    observed = frame[exercise_column].astype(bool).to_numpy()
    mismatches = int(np.count_nonzero(expected != observed))
    return {
        "observations": float(len(frame)),
        "mismatches": float(mismatches),
        "mismatch_rate": float(mismatches / len(frame)) if len(frame) else 0.0,
    }


def boundary_band_metrics(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    exercise_column: str = "exercise_now",
    distance_column: str = "boundary_distance_normalized",
    bands: tuple[float, ...] = (0.001, 0.005, 0.010),
    threshold: float = 0.5,
    actual_price_column: str | None = None,
    predicted_price_column: str | None = None,
) -> pd.DataFrame:
    """Evaluate exercise classification and optional pricing near the boundary."""

    required = [probability_column, exercise_column, distance_column]
    if actual_price_column is not None or predicted_price_column is not None:
        if actual_price_column is None or predicted_price_column is None:
            raise ValueError("Both price columns must be supplied together.")
        required.extend([actual_price_column, predicted_price_column])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing columns: {missing}")

    rows: list[dict[str, float]] = []
    for band in bands:
        if band <= 0.0:
            raise ValueError("Boundary bands must be positive.")
        subset = frame.loc[frame[distance_column].abs() <= band]
        if len(subset) == 0:
            rows.append(
                {
                    "boundary_band": float(band),
                    "observations": 0.0,
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "false_exercise_rate": float("nan"),
                    "missed_exercise_rate": float("nan"),
                    "price_mae": float("nan"),
                }
            )
            continue
        metrics = binary_classification_metrics(
            subset[exercise_column],
            subset[probability_column],
            threshold=threshold,
        )
        actual = subset[exercise_column].astype(int).to_numpy()
        predicted = (
            subset[probability_column].to_numpy(dtype=np.float64) >= threshold
        ).astype(int)
        continue_count = max(int((actual == 0).sum()), 1)
        exercise_count = max(int((actual == 1).sum()), 1)
        false_exercise = int(((actual == 0) & (predicted == 1)).sum())
        missed_exercise = int(((actual == 1) & (predicted == 0)).sum())
        price_mae = float("nan")
        if actual_price_column is not None and predicted_price_column is not None:
            price_mae = float(
                np.mean(
                    np.abs(
                        subset[actual_price_column].to_numpy(dtype=np.float64)
                        - subset[predicted_price_column].to_numpy(dtype=np.float64)
                    )
                )
            )
        rows.append(
            {
                "boundary_band": float(band),
                "observations": float(len(subset)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_exercise_rate": float(false_exercise / continue_count),
                "missed_exercise_rate": float(missed_exercise / exercise_count),
                "price_mae": price_mae,
            }
        )
    return pd.DataFrame(rows)


def extract_probability_boundary(
    moneyness: Iterable[float],
    exercise_probability: Iterable[float],
    *,
    threshold: float = 0.5,
) -> BoundaryEstimate:
    """Estimate a crossing in a monotone-style moneyness probability slice.

    American put exercise probability is expected to be higher at lower
    moneyness. The function sorts by moneyness and linearly interpolates the
    first probability crossing from above the threshold to below it.
    """

    x = np.asarray(moneyness, dtype=np.float64).reshape(-1)
    p = np.asarray(exercise_probability, dtype=np.float64).reshape(-1)
    if len(x) < 2 or len(x) != len(p):
        raise ValueError("Boundary arrays must have equal length of at least two.")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Boundary arrays contain non-finite values.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("exercise_probability must lie between zero and one.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one.")

    order = np.argsort(x)
    x = x[order]
    p = p[order]
    above = p >= threshold
    crossing_indices = np.flatnonzero(above[:-1] & ~above[1:])
    if len(crossing_indices) == 0:
        return BoundaryEstimate(
            boundary_moneyness=float("nan"),
            crossing_found=False,
            threshold=float(threshold),
            minimum_moneyness=float(x.min()),
            maximum_moneyness=float(x.max()),
        )

    index = int(crossing_indices[0])
    x0, x1 = x[index], x[index + 1]
    p0, p1 = p[index], p[index + 1]
    if np.isclose(p0, p1):
        boundary = 0.5 * (x0 + x1)
    else:
        boundary = x0 + (threshold - p0) * (x1 - x0) / (p1 - p0)
    return BoundaryEstimate(
        boundary_moneyness=float(boundary),
        crossing_found=True,
        threshold=float(threshold),
        minimum_moneyness=float(x.min()),
        maximum_moneyness=float(x.max()),
    )


def extract_label_boundary(
    moneyness: Iterable[float],
    exercise_label: Iterable[float],
) -> BoundaryEstimate:
    """Estimate the CRR boundary from binary labels on a moneyness slice."""

    labels = np.asarray(exercise_label, dtype=np.float64).reshape(-1)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("exercise_label must be binary.")
    return extract_probability_boundary(
        moneyness,
        labels,
        threshold=0.5,
    )


def boundary_location_error(
    actual_boundaries: Iterable[float],
    predicted_boundaries: Iterable[float],
) -> dict[str, float]:
    """Summarize absolute boundary-location errors over valid slices."""

    actual = np.asarray(actual_boundaries, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_boundaries, dtype=np.float64).reshape(-1)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Boundary arrays must have equal non-zero length.")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    errors = np.abs(actual[valid] - predicted[valid])
    return {
        "slices": float(len(actual)),
        "valid_slices": float(valid.sum()),
        "coverage": float(valid.mean()),
        "boundary_mae": float(errors.mean()) if len(errors) else float("nan"),
        "boundary_max_error": float(errors.max()) if len(errors) else float("nan"),
    }


def boundary_monotonicity_report(
    frame: pd.DataFrame,
    *,
    parameter_column: str,
    boundary_column: str,
    expected_direction: str,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Count adjacent monotonicity violations in estimated boundary curves."""

    if expected_direction not in {"increasing", "decreasing"}:
        raise ValueError("expected_direction must be increasing or decreasing.")
    required = [parameter_column, boundary_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing columns: {missing}")
    ordered = frame[[parameter_column, boundary_column]].dropna().sort_values(
        parameter_column
    )
    differences = np.diff(ordered[boundary_column].to_numpy(dtype=np.float64))
    if expected_direction == "increasing":
        violations = differences < -tolerance
    else:
        violations = differences > tolerance
    return {
        "comparisons": float(len(differences)),
        "violations": float(violations.sum()),
        "violation_rate": float(violations.mean()) if len(differences) else 0.0,
    }


def decide_h4_multitask_learning(
    *,
    classifier_boundary_f1: float,
    multitask_boundary_f1: float,
    price_only_boundary_mae: float,
    multitask_boundary_mae: float,
    minimum_f1_improvement: float = 0.01,
    minimum_mae_improvement: float = 0.01,
) -> H4Decision:
    """Decide H4 using predefined boundary F1 and price-MAE improvements."""

    values = {
        "classifier_boundary_f1": classifier_boundary_f1,
        "multitask_boundary_f1": multitask_boundary_f1,
        "price_only_boundary_mae": price_only_boundary_mae,
        "multitask_boundary_mae": multitask_boundary_mae,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("H4 evidence values must be finite.")

    f1_improvement = multitask_boundary_f1 - classifier_boundary_f1
    mae_improvement = (
        price_only_boundary_mae - multitask_boundary_mae
    ) / max(price_only_boundary_mae, np.finfo(np.float64).eps)

    f1_supported = f1_improvement >= minimum_f1_improvement
    mae_supported = mae_improvement >= minimum_mae_improvement
    if f1_supported and mae_supported:
        decision = "supported"
        rationale = (
            "The multi-task model exceeds both predefined boundary-F1 and "
            "boundary-price-MAE improvement thresholds."
        )
    elif f1_supported or mae_supported:
        decision = "partially supported"
        rationale = (
            "The multi-task model improves one, but not both, predefined "
            "boundary criteria."
        )
    else:
        decision = "not supported"
        rationale = "The multi-task model does not meet either improvement threshold."

    return H4Decision(
        hypothesis="H4 — Multi-task exercise learning",
        decision=decision,
        rationale=rationale,
        evidence={
            **{key: float(value) for key, value in values.items()},
            "boundary_f1_improvement": float(f1_improvement),
            "boundary_mae_relative_improvement": float(mae_improvement),
            "required_f1_improvement": float(minimum_f1_improvement),
            "required_mae_improvement": float(minimum_mae_improvement),
        },
    )


__all__ = [
    "BoundaryEstimate",
    "H4Decision",
    "boundary_band_metrics",
    "boundary_location_error",
    "boundary_monotonicity_report",
    "calculate_boundary_distance_normalized",
    "decide_h4_multitask_learning",
    "extract_label_boundary",
    "extract_probability_boundary",
    "validate_exercise_labels",
]
