"""Evaluation utilities for classical and neural Longstaff–Schwartz results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math

import numpy as np
import pandas as pd


REQUIRED_ID_COLUMN = "contract_id"


def align_contract_results(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_column: str = REQUIRED_ID_COLUMN,
) -> pd.DataFrame:
    """Align two contract result frames and reject duplicates or missing IDs."""

    for name, frame in {"reference": reference, "candidate": candidate}.items():
        if id_column not in frame.columns:
            raise ValueError(f"{name} is missing {id_column!r}.")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate contract IDs.")
    merged = reference.merge(
        candidate,
        on=id_column,
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Contract IDs do not match exactly.")
    return merged.sort_values(id_column, ignore_index=True)


def pricing_error_metrics(
    reference_prices: Sequence[float] | np.ndarray,
    predicted_prices: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate contract-level pricing errors."""

    reference = np.asarray(reference_prices, dtype=np.float64)
    predicted = np.asarray(predicted_prices, dtype=np.float64)
    if reference.shape != predicted.shape or reference.size == 0:
        raise ValueError("reference and predicted prices need equal non-empty shapes.")
    if not (np.isfinite(reference).all() and np.isfinite(predicted).all()):
        raise ValueError("prices must be finite.")
    errors = predicted - reference
    absolute = np.abs(errors)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "mean_bias": float(np.mean(errors)),
        "normalized_mae": float(
            np.mean(absolute / np.maximum(np.abs(reference), 1e-8))
        ),
    }


def compare_lsm_methods(
    results: pd.DataFrame,
    *,
    benchmark_column: str,
    method_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a method-level pricing comparison table."""

    missing = [
        column
        for column in [benchmark_column, *method_columns]
        if column not in results.columns
    ]
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}.")
    rows = []
    for method in method_columns:
        row = {"method": method}
        row.update(pricing_error_metrics(results[benchmark_column], results[method]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True)


def confidence_interval_coverage(
    benchmark: Sequence[float] | np.ndarray,
    interval_low: Sequence[float] | np.ndarray,
    interval_high: Sequence[float] | np.ndarray,
) -> float:
    """Return the fraction of benchmark values covered by reported intervals."""

    benchmark = np.asarray(benchmark, dtype=np.float64)
    low = np.asarray(interval_low, dtype=np.float64)
    high = np.asarray(interval_high, dtype=np.float64)
    if not (benchmark.shape == low.shape == high.shape) or benchmark.size == 0:
        raise ValueError("benchmark and interval arrays need equal non-empty shapes.")
    if np.any(low > high):
        raise ValueError("interval_low cannot exceed interval_high.")
    return float(np.mean((benchmark >= low) & (benchmark <= high)))


def exercise_policy_metrics(
    reference_exercise_indices: Sequence[int] | np.ndarray,
    predicted_exercise_indices: Sequence[int] | np.ndarray,
    *,
    maturity_index: int,
) -> dict[str, float]:
    """Compare stopping decisions for paired paths."""

    reference = np.asarray(reference_exercise_indices, dtype=np.int64)
    predicted = np.asarray(predicted_exercise_indices, dtype=np.int64)
    if reference.shape != predicted.shape or reference.size == 0:
        raise ValueError("exercise-index arrays need equal non-empty shapes.")
    if np.any(reference < 0) or np.any(predicted < 0):
        raise ValueError("exercise indices cannot be negative.")
    reference_early = reference < maturity_index
    predicted_early = predicted < maturity_index
    return {
        "exact_step_agreement": float(np.mean(reference == predicted)),
        "mean_absolute_step_error": float(np.mean(np.abs(reference - predicted))),
        "early_exercise_agreement": float(np.mean(reference_early == predicted_early)),
        "false_early_exercise_rate": float(
            np.mean(predicted_early & ~reference_early)
        ),
        "missed_early_exercise_rate": float(
            np.mean(~predicted_early & reference_early)
        ),
    }


def stopping_time_total_variation(
    first_indices: Sequence[int] | np.ndarray,
    second_indices: Sequence[int] | np.ndarray,
    *,
    n_steps: int,
) -> float:
    """Total-variation distance between stopping-index distributions."""

    first = np.asarray(first_indices, dtype=np.int64)
    second = np.asarray(second_indices, dtype=np.int64)
    if first.size == 0 or second.size == 0:
        raise ValueError("stopping-index arrays cannot be empty.")
    if np.any((first < 0) | (first > n_steps)) or np.any(
        (second < 0) | (second > n_steps)
    ):
        raise ValueError("stopping indices must lie between zero and n_steps.")
    bins = np.arange(n_steps + 2) - 0.5
    first_hist = np.histogram(first, bins=bins)[0] / len(first)
    second_hist = np.histogram(second, bins=bins)[0] / len(second)
    return float(0.5 * np.sum(np.abs(first_hist - second_hist)))


def runtime_summary(
    records: pd.DataFrame,
    *,
    method_column: str = "method",
    runtime_column: str = "runtime_seconds",
) -> pd.DataFrame:
    """Summarize repeated runtime measurements by method."""

    if method_column not in records or runtime_column not in records:
        raise ValueError("runtime records are missing required columns.")
    if (records[runtime_column] < 0.0).any():
        raise ValueError("runtime values cannot be negative.")
    return (
        records.groupby(method_column, as_index=False)[runtime_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={method_column: "method"})
    )


def aggregate_seed_results(
    results: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("contract_id", "method"),
    value_column: str = "price",
) -> pd.DataFrame:
    """Aggregate multi-seed prices into mean, standard deviation, and count."""

    required = [*group_columns, value_column]
    missing = [column for column in required if column not in results]
    if missing:
        raise ValueError(f"Missing seed-aggregation columns: {missing}.")
    return (
        results.groupby(list(group_columns), as_index=False)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{value_column}_mean",
                "std": f"{value_column}_std",
                "count": "seed_count",
            }
        )
    )


__all__ = [
    "aggregate_seed_results",
    "align_contract_results",
    "compare_lsm_methods",
    "confidence_interval_coverage",
    "exercise_policy_metrics",
    "pricing_error_metrics",
    "runtime_summary",
    "stopping_time_total_variation",
]
