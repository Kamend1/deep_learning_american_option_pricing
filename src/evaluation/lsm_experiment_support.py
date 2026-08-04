"""Shared support utilities for the Notebook 07 LSM experiment.

The module keeps experiment orchestration in the notebook while moving reusable
artifact checks, Monte Carlo summaries, diagnostics, path-batch construction,
and table persistence into the project source package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.training.artifact_management import (
    inspect_training_artifacts,
    write_training_manifest,
)

from src.evaluation.lsm_comparison import pricing_error_metrics
from src.models.neural_longstaff_schwartz import ContractPathBatch
from src.pricing.simulation import GBMContract, simulate_contract_paths




def antithetic_pair_summary(
    discounted_payoffs: Sequence[float] | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Return a Monte Carlo summary using independent antithetic-pair means."""

    values = np.asarray(
        discounted_payoffs,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) < 4 or len(values) % 2:
        raise ValueError(
            "Antithetic-pair inference requires an even number "
            "of at least four paths."
        )
    if not np.isfinite(values).all():
        raise ValueError("Discounted payoffs contain non-finite values.")
    if confidence_level != 0.95:
        raise ValueError("Only 95% confidence intervals are supported.")

    half = len(values) // 2
    pair_means = 0.5 * (
        values[:half] + values[half:]
    )
    price = float(pair_means.mean())
    standard_error = float(
        pair_means.std(ddof=1)
        / math.sqrt(len(pair_means))
    )
    z_value = 1.959963984540054

    return {
        "price": price,
        "standard_error": standard_error,
        "confidence_interval_low": (
            price - z_value * standard_error
        ),
        "confidence_interval_high": (
            price + z_value * standard_error
        ),
        "independent_pairs": int(len(pair_means)),
    }


def nested_antithetic_subset(
    paths: np.ndarray,
    n_paths: int,
) -> np.ndarray:
    """Take a nested antithetic subset while preserving path pairs."""

    path_array = np.asarray(paths)
    total = len(path_array)

    if total % 2 or n_paths % 2:
        raise ValueError(
            "Both full and requested path counts must be even."
        )
    if n_paths <= 0 or n_paths > total:
        raise ValueError(
            "Requested path count is outside the available range."
        )

    full_half = total // 2
    requested_half = n_paths // 2
    indices = np.concatenate(
        [
            np.arange(requested_half),
            np.arange(
                full_half,
                full_half + requested_half,
            ),
        ]
    )
    return path_array[indices]


def conditional_exercise_policy_metrics(
    reference_exercise_indices: Sequence[int] | np.ndarray,
    predicted_exercise_indices: Sequence[int] | np.ndarray,
    *,
    maturity_index: int,
) -> dict[str, float | int]:
    """Compare paired stopping decisions with conditional error rates."""

    reference = np.asarray(
        reference_exercise_indices,
        dtype=np.int64,
    )
    predicted = np.asarray(
        predicted_exercise_indices,
        dtype=np.int64,
    )

    if reference.shape != predicted.shape or reference.size == 0:
        raise ValueError(
            "Exercise-index arrays need equal non-empty shapes."
        )
    if maturity_index <= 0:
        raise ValueError("maturity_index must be positive.")

    reference_early = reference < maturity_index
    predicted_early = predicted < maturity_index
    false_early = predicted_early & ~reference_early
    missed_early = ~predicted_early & reference_early

    continue_count = max(
        int((~reference_early).sum()),
        1,
    )
    exercise_count = max(
        int(reference_early.sum()),
        1,
    )

    return {
        "exact_step_agreement": float(
            np.mean(reference == predicted)
        ),
        "mean_absolute_step_error": float(
            np.mean(np.abs(reference - predicted))
        ),
        "early_exercise_agreement": float(
            np.mean(
                reference_early == predicted_early
            )
        ),
        "false_early_exercise_count": int(
            false_early.sum()
        ),
        "false_early_exercise_rate": float(
            false_early.sum() / continue_count
        ),
        "missed_early_exercise_count": int(
            missed_early.sum()
        ),
        "missed_early_exercise_rate": float(
            missed_early.sum() / exercise_count
        ),
    }


def paired_mae_bootstrap(
    reference: Sequence[float] | np.ndarray,
    classical: Sequence[float] | np.ndarray,
    neural: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap the paired neural-minus-classical MAE difference."""

    reference_array = np.asarray(
        reference,
        dtype=np.float64,
    )
    classical_array = np.asarray(
        classical,
        dtype=np.float64,
    )
    neural_array = np.asarray(
        neural,
        dtype=np.float64,
    )

    if not (
        reference_array.shape
        == classical_array.shape
        == neural_array.shape
        and reference_array.size > 1
    ):
        raise ValueError(
            "Bootstrap arrays need equal shapes and multiple rows."
        )
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    paired_difference = (
        np.abs(neural_array - reference_array)
        - np.abs(classical_array - reference_array)
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(paired_difference),
        size=(
            n_resamples,
            len(paired_difference),
        ),
    )
    bootstrap_means = (
        paired_difference[sampled].mean(axis=1)
    )

    return {
        "neural_minus_classical_mae": float(
            paired_difference.mean()
        ),
        "ci_low": float(
            np.quantile(bootstrap_means, 0.025)
        ),
        "ci_high": float(
            np.quantile(bootstrap_means, 0.975)
        ),
        "probability_neural_is_better": float(
            np.mean(bootstrap_means < 0.0)
        ),
        "resamples": int(n_resamples),
    }


def error_distribution_table(
    frame: pd.DataFrame,
    *,
    benchmark_column: str,
    method_columns: Sequence[str],
    group_column: str | None = None,
) -> pd.DataFrame:
    """Summarize error direction and quantiles by method and group."""

    required = [
        benchmark_column,
        *method_columns,
    ]
    if group_column is not None:
        required.append(group_column)

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Frame is missing columns: {missing}"
        )

    groups = (
        frame.groupby(
            group_column,
            observed=True,
        )
        if group_column is not None
        else [("all", frame)]
    )

    rows: list[dict[str, Any]] = []

    for group_name, subset in groups:
        reference = subset[
            benchmark_column
        ].to_numpy(dtype=np.float64)

        for method in method_columns:
            errors = (
                subset[method].to_numpy(
                    dtype=np.float64
                )
                - reference
            )
            rows.append(
                {
                    "group": group_name,
                    "method": method,
                    "observations": int(
                        len(errors)
                    ),
                    "negative_error_rate": float(
                        np.mean(errors < 0.0)
                    ),
                    "positive_error_rate": float(
                        np.mean(errors > 0.0)
                    ),
                    "mean_error": float(
                        errors.mean()
                    ),
                    "q05_error": float(
                        np.quantile(
                            errors,
                            0.05,
                        )
                    ),
                    "median_error": float(
                        np.median(errors)
                    ),
                    "q95_error": float(
                        np.quantile(
                            errors,
                            0.95,
                        )
                    ),
                    "mae": float(
                        np.mean(
                            np.abs(errors)
                        )
                    ),
                    "maximum_absolute_error": float(
                        np.max(
                            np.abs(errors)
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def pricing_bound_report(
    frame: pd.DataFrame,
    *,
    method_columns: Sequence[str],
    intrinsic_column: str = "intrinsic_value",
    european_column: str = "european_price",
) -> pd.DataFrame:
    """Report negative prices and financial lower-bound violations."""

    required = [
        intrinsic_column,
        european_column,
        *method_columns,
    ]
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Frame is missing columns: {missing}"
        )

    rows: list[dict[str, Any]] = []
    intrinsic = frame[
        intrinsic_column
    ].to_numpy(dtype=np.float64)
    european = frame[
        european_column
    ].to_numpy(dtype=np.float64)

    for method in method_columns:
        prices = frame[
            method
        ].to_numpy(dtype=np.float64)

        for check, floor in (
            (
                "negative_price",
                np.zeros_like(prices),
            ),
            (
                "below_intrinsic",
                intrinsic,
            ),
            (
                "below_european",
                european,
            ),
        ):
            violation = prices < floor
            magnitude = np.maximum(
                floor - prices,
                0.0,
            )
            rows.append(
                {
                    "method": method,
                    "check": check,
                    "violations": int(
                        violation.sum()
                    ),
                    "violation_rate": float(
                        violation.mean()
                    ),
                    "maximum_violation": float(
                        magnitude.max()
                    ),
                    "mean_positive_violation": (
                        float(
                            magnitude[
                                violation
                            ].mean()
                        )
                        if violation.any()
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def segmented_pricing_errors(
    frame: pd.DataFrame,
    *,
    segment_columns: Sequence[str],
    benchmark_column: str,
    method_columns: Sequence[str],
) -> pd.DataFrame:
    """Return pricing errors across predefined contract segments."""

    required = [
        benchmark_column,
        *segment_columns,
        *method_columns,
    ]
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Frame is missing columns: {missing}"
        )

    rows: list[dict[str, Any]] = []

    for segment_column in segment_columns:
        for segment, subset in frame.groupby(
            segment_column,
            observed=True,
        ):
            reference = subset[
                benchmark_column
            ].to_numpy(dtype=np.float64)

            for method in method_columns:
                metrics = pricing_error_metrics(
                    reference,
                    subset[method].to_numpy(
                        dtype=np.float64
                    ),
                )
                rows.append(
                    {
                        "segment_type": segment_column,
                        "segment": str(segment),
                        "method": method,
                        "observations": int(
                            len(subset)
                        ),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def build_contract_path_batches(
    contracts: Sequence[GBMContract],
    *,
    n_paths: int,
    n_steps: int,
    base_seed: int,
    seed_offset: int,
    description: str,
    show_progress: bool = True,
) -> list[ContractPathBatch]:
    """Simulate deterministic path batches for a contract collection."""

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError(
            "n_paths and n_steps must be positive."
        )

    batches: list[ContractPathBatch] = []

    progress = tqdm(
        enumerate(contracts),
        total=len(contracts),
        desc=description,
        unit="contract",
        disable=not show_progress,
    )

    for index, contract in progress:
        paths = simulate_contract_paths(
            contract,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=(
                int(base_seed)
                + int(seed_offset)
                + index
            ),
        )
        batches.append(
            ContractPathBatch(
                contract=contract,
                paths=paths,
            )
        )

    return batches


def save_dataframe(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save Parquet when available and fall back to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        frame.to_parquet(
            output_path,
            index=False,
        )
        return output_path
    except ImportError:
        fallback = output_path.with_suffix(
            ".csv"
        )
        frame.to_csv(
            fallback,
            index=False,
        )
        return fallback


__all__ = [
    "antithetic_pair_summary",
    "build_contract_path_batches",
    "conditional_exercise_policy_metrics",
    "error_distribution_table",
    "inspect_training_artifacts",
    "nested_antithetic_subset",
    "paired_mae_bootstrap",
    "pricing_bound_report",
    "save_dataframe",
    "segmented_pricing_errors",
    "write_training_manifest",
]
