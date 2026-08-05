"""Authoritative common-test comparison for the static models in Notebooks 04-08.

The upstream notebooks export aligned prediction tables.  This module joins
those tables on ``sample_id`` and recomputes every common pricing metric from
one evidence matrix.  It does not train, load model weights, or trust aggregate
summary tables when row-level predictions are available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.final_artifact_adapters import FinalNotebookPackage


DEFAULT_ERROR_BANDS = (0.001, 0.005, 0.010, 0.050)
DEFAULT_BOUNDARY_BANDS = (0.001, 0.005, 0.010)
DEFAULT_FINANCIAL_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class StaticModelSpec:
    """One prediction column participating in the common static comparison."""

    model_id: str
    model: str
    source_notebook: str
    prediction_column: str
    evaluation_role: str
    source_selected: bool
    financially_constrained: bool


MODEL_SPECS: tuple[StaticModelSpec, ...] = (
    StaticModelSpec(
        "black_scholes_proxy",
        "Black–Scholes proxy",
        "04",
        "normalized_european_price",
        "analytical proxy",
        False,
        False,
    ),
    StaticModelSpec(
        "direct_mlp",
        "Direct MLP",
        "04",
        "direct_mlp_prediction",
        "direct neural baseline",
        True,
        False,
    ),
    StaticModelSpec(
        "zero_premium_baseline",
        "Zero-premium baseline",
        "05",
        "zero_premium_baseline",
        "naive residual baseline",
        False,
        False,
    ),
    StaticModelSpec(
        "mean_premium_baseline",
        "Mean-premium baseline",
        "05",
        "mean_premium_baseline",
        "naive residual baseline",
        False,
        False,
    ),
    StaticModelSpec(
        "unconstrained_premium_mlp",
        "Unconstrained premium MLP",
        "05",
        "unconstrained_premium",
        "residual candidate",
        False,
        False,
    ),
    StaticModelSpec(
        "nonnegative_premium_mlp",
        "Non-negative premium MLP",
        "05",
        "nonnegative_premium",
        "residual candidate",
        False,
        False,
    ),
    StaticModelSpec(
        "constrained_floor_residual_mlp",
        "Constrained floor residual MLP",
        "05",
        "constrained_floor_prediction",
        "selected static price model",
        True,
        True,
    ),
    StaticModelSpec(
        "price_only_constrained_residual_mlp",
        "Price-only constrained residual MLP",
        "06",
        "price_only_normalized_price",
        "price-only control",
        False,
        True,
    ),
    StaticModelSpec(
        "multitask_constrained_residual_mlp",
        "Multi-task constrained residual MLP",
        "06",
        "predicted_normalized_american_price",
        "selected joint price-decision model",
        True,
        True,
    ),
    StaticModelSpec(
        "integrated_warm_start_constrained_price",
        "Integrated warm-start constrained price",
        "08",
        "predicted_normalized_american_price",
        "preferred in-domain integrated deployment output",
        True,
        True,
    ),
    StaticModelSpec(
        "integrated_warm_start_direct_price_head",
        "Integrated warm-start direct price head",
        "08",
        "predicted_direct_normalized_american_price",
        "auxiliary warm-start integrated output",
        False,
        False,
    ),
    StaticModelSpec(
        "integrated_scratch_constrained_price",
        "Integrated balanced-scratch constrained price",
        "08_scratch",
        "predicted_normalized_american_price",
        "scratch experiment winner and robustness benchmark",
        False,
        True,
    ),
    StaticModelSpec(
        "integrated_scratch_direct_price_head",
        "Integrated balanced-scratch direct price head",
        "08_scratch",
        "predicted_direct_normalized_american_price",
        "auxiliary scratch benchmark output",
        False,
        False,
    ),
)

MODEL_SPEC_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}


TARGET_COLUMNS = {
    "04": "normalized_american_price",
    "05": "normalized_american_price",
    "06": "normalized_american_price",
    "08": "true_normalized_american_price",
    "08_scratch": "true_normalized_american_price",
}

REFERENCE_CONTEXT_COLUMNS = (
    "sample_id",
    "split",
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
    "exercise_now",
    "normalized_european_price",
    "normalized_intrinsic_value",
    "normalized_american_price",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _numeric_array(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        invalid = int((~np.isfinite(values)).sum())
        raise ValueError(f"{label} contains {invalid} non-finite values")
    return values


def _read_fixed_strike(project_root: Path) -> float:
    manifest_path = (
        Path(project_root).resolve()
        / "data"
        / "manifests"
        / "production_dataset_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Production manifest is required to recover the fixed strike: "
            f"{manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        strike = float(payload["generation_config"]["strike"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Production manifest must contain generation_config.strike"
        ) from exc
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"Invalid production strike: {strike}")
    return strike


def _load_prediction_frames(
    packages: Mapping[str, FinalNotebookPackage],
) -> dict[str, pd.DataFrame]:
    required = ("04", "05", "06", "08")
    missing = [notebook for notebook in required if notebook not in packages]
    if missing:
        raise KeyError(f"Missing static packages: {missing}")

    frames: dict[str, pd.DataFrame] = {}
    for notebook in required:
        frame = packages[notebook].load_test_predictions().copy()
        _require_columns(
            frame,
            ("sample_id", TARGET_COLUMNS[notebook]),
            label=f"Notebook {notebook} predictions",
        )
        if frame["sample_id"].isna().any():
            raise ValueError(f"Notebook {notebook} contains missing sample_id values")
        duplicates = int(frame["sample_id"].duplicated().sum())
        if duplicates:
            raise ValueError(
                f"Notebook {notebook} contains {duplicates} duplicate sample IDs"
            )
        frames[notebook] = frame.sort_values("sample_id").reset_index(drop=True)

    scratch = packages["08"].load_benchmark_test_predictions().copy()
    _require_columns(
        scratch,
        ("sample_id", TARGET_COLUMNS["08_scratch"]),
        label="Notebook 08 scratch benchmark predictions",
    )
    if scratch["sample_id"].isna().any():
        raise ValueError("Notebook 08 scratch predictions contain missing sample IDs")
    duplicates = int(scratch["sample_id"].duplicated().sum())
    if duplicates:
        raise ValueError(
            f"Notebook 08 scratch predictions contain {duplicates} duplicate sample IDs"
        )
    frames["08_scratch"] = scratch.sort_values("sample_id").reset_index(drop=True)
    return frames


def _attach_boundary_distance(
    matrix: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> pd.DataFrame:
    candidates: list[tuple[str, pd.DataFrame]] = []
    for notebook in ("06", "05"):
        frame = frames[notebook]
        if "boundary_distance_normalized" in frame.columns:
            candidates.append(
                (
                    notebook,
                    frame[["sample_id", "boundary_distance_normalized"]].copy(),
                )
            )

    if not candidates:
        return matrix

    source_notebook, boundary = candidates[0]
    result = matrix.merge(
        boundary,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if result["boundary_distance_normalized"].isna().any():
        raise ValueError(
            f"Notebook {source_notebook} boundary distance did not align to all rows"
        )

    if len(candidates) > 1:
        other_notebook, other = candidates[1]
        check = boundary.merge(
            other,
            on="sample_id",
            validate="one_to_one",
            suffixes=("_primary", "_secondary"),
        )
        left = _numeric_array(
            check["boundary_distance_normalized_primary"],
            label=f"Notebook {source_notebook} boundary distance",
        )
        right = _numeric_array(
            check["boundary_distance_normalized_secondary"],
            label=f"Notebook {other_notebook} boundary distance",
        )
        if not np.allclose(left, right, atol=atol, rtol=rtol):
            maximum = float(np.max(np.abs(left - right)))
            raise ValueError(
                "Boundary-distance exports disagree between Notebooks "
                f"{source_notebook} and {other_notebook}; max difference={maximum}"
            )
    return result



def _aligned_context_values(
    reference_ids: pd.Series,
    source: pd.DataFrame,
    column: str,
    *,
    source_notebook: str,
) -> pd.Series:
    """Align one context column from an upstream prediction table."""

    lookup = source.set_index("sample_id")[column]
    values = reference_ids.map(lookup)
    if values.isna().any():
        missing = int(values.isna().sum())
        raise ValueError(
            f"Notebook {source_notebook} context column {column!r} "
            f"did not align to {missing} reference samples"
        )
    return values


def _context_values_match(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> tuple[bool, float | None]:
    """Compare context exports while allowing normal float32/float64 rounding."""

    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if left_numeric.notna().all() and right_numeric.notna().all():
        left_values = left_numeric.to_numpy(dtype=np.float64)
        right_values = right_numeric.to_numpy(dtype=np.float64)
        difference = np.abs(left_values - right_values)
        return (
            bool(np.allclose(left_values, right_values, atol=atol, rtol=rtol)),
            float(difference.max()) if len(difference) else 0.0,
        )

    left_values = left.astype(str).to_numpy()
    right_values = right.astype(str).to_numpy()
    return bool(np.array_equal(left_values, right_values)), None


def _build_common_context(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one shared test context from the aligned upstream exports.

    Notebook 04 is the authoritative source for sample IDs and the true price
    target. Context fields are then taken from the first notebook that exports
    them, with cross-source validation where the same field appears more than
    once. This avoids requiring every prediction package to duplicate all
    contract fields.
    """

    reference = frames["04"]
    _require_columns(
        reference,
        (
            "sample_id",
            "normalized_american_price",
        ),
        label="Notebook 04 predictions",
    )

    matrix = reference[
        ["sample_id", "normalized_american_price"]
    ].copy()
    matrix = matrix.rename(
        columns={
            "normalized_american_price": (
                "true_normalized_american_price"
            )
        }
    )

    source_priority = ("04", "05", "06", "08", "08_scratch")
    context_columns = tuple(
        column
        for column in REFERENCE_CONTEXT_COLUMNS
        if column not in {
            "sample_id",
            "normalized_american_price",
        }
    )

    for column in context_columns:
        candidates: list[tuple[str, pd.Series]] = []
        for notebook in source_priority:
            source = frames[notebook]
            if column not in source.columns:
                continue
            values = _aligned_context_values(
                matrix["sample_id"],
                source,
                column,
                source_notebook=notebook,
            )
            candidates.append((notebook, values))

        if not candidates:
            continue

        primary_notebook, primary_values = candidates[0]
        matrix[column] = primary_values.to_numpy()

        for other_notebook, other_values in candidates[1:]:
            matches, maximum_difference = _context_values_match(
                primary_values,
                other_values,
            )
            if not matches:
                difference_text = (
                    ""
                    if maximum_difference is None
                    else f"; max difference={maximum_difference}"
                )
                raise ValueError(
                    f"Context field {column!r} disagrees between "
                    f"Notebooks {primary_notebook} and {other_notebook}"
                    f"{difference_text}"
                )

    # Older Notebook 04 exports may not contain the normalized intrinsic value.
    # Notebook 05 and 06 normally provide it. As a final deterministic fallback,
    # moneyness=S/K implies normalized intrinsic=max(1-S/K, 0).
    if (
        "normalized_intrinsic_value" not in matrix.columns
        and "moneyness" in matrix.columns
    ):
        moneyness = _numeric_array(
            matrix["moneyness"],
            label="moneyness",
        )
        matrix["normalized_intrinsic_value"] = np.maximum(
            1.0 - moneyness,
            0.0,
        )

    required_context = (
        "normalized_european_price",
        "normalized_intrinsic_value",
    )
    _require_columns(
        matrix,
        required_context,
        label="Common static test context",
    )
    return matrix


def build_static_prediction_matrix(
    project_root: Path,
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Join all static-model predictions on the audited common test sample."""

    frames = _load_prediction_frames(packages)
    matrix = _build_common_context(frames)

    matrix["strike"] = _read_fixed_strike(project_root)
    matrix["true_price"] = (
        _numeric_array(
            matrix["true_normalized_american_price"],
            label="true normalized price",
        )
        * matrix["strike"].to_numpy(dtype=np.float64)
    )
    matrix["normalized_financial_floor"] = np.maximum(
        _numeric_array(
            matrix["normalized_european_price"],
            label="normalized European price",
        ),
        _numeric_array(
            matrix["normalized_intrinsic_value"],
            label="normalized intrinsic value",
        ),
    )

    matrix = _attach_boundary_distance(matrix, frames)

    for spec in MODEL_SPECS:
        source = frames[spec.source_notebook]
        _require_columns(
            source,
            ("sample_id", spec.prediction_column),
            label=f"Notebook {spec.source_notebook} predictions for {spec.model}",
        )
        prediction = source[["sample_id", spec.prediction_column]].copy()
        prediction = prediction.rename(
            columns={spec.prediction_column: f"prediction__{spec.model_id}"}
        )
        matrix = matrix.merge(
            prediction,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

    prediction_columns = [f"prediction__{spec.model_id}" for spec in MODEL_SPECS]
    for column in prediction_columns:
        _numeric_array(matrix[column], label=column)

    if matrix["sample_id"].duplicated().any():
        raise RuntimeError("Common static prediction matrix contains duplicate IDs")
    return matrix.sort_values("sample_id").reset_index(drop=True)


def _regression_record(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    error_bands: Sequence[float],
) -> dict[str, float | int]:
    error = predicted - actual
    absolute_error = np.abs(error)
    squared_error = error**2
    total_variation = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_variation = float(np.sum(squared_error))
    r2 = 1.0 - residual_variation / total_variation if total_variation > 0 else np.nan

    result: dict[str, float | int] = {
        "observations": int(len(actual)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_error": float(np.mean(error)),
        "r2": float(r2),
    }
    for band in error_bands:
        label = str(float(band)).rstrip("0").rstrip(".").replace(".", "_")
        result[f"within_{label}"] = float(np.mean(absolute_error <= float(band)))
    return result


def _financial_consistency_record(
    prediction: np.ndarray,
    european: np.ndarray,
    intrinsic: np.ndarray,
    *,
    tolerance: float = DEFAULT_FINANCIAL_TOLERANCE,
) -> dict[str, float | int]:
    observations = len(prediction)
    floor = np.maximum(european, intrinsic)
    checks = {
        "negative": prediction < -tolerance,
        "below_european": prediction < european - tolerance,
        "below_intrinsic": prediction < intrinsic - tolerance,
        "below_financial_floor": prediction < floor - tolerance,
    }
    result: dict[str, float | int] = {"observations": int(observations)}
    for name, mask in checks.items():
        count = int(np.sum(mask))
        result[f"{name}_count"] = count
        result[f"{name}_rate"] = float(count / observations) if observations else np.nan
    return result


def build_static_model_metrics(
    prediction_matrix: pd.DataFrame,
    *,
    error_bands: Sequence[float] = DEFAULT_ERROR_BANDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute common pricing and financial-consistency metrics."""

    actual_normalized = _numeric_array(
        prediction_matrix["true_normalized_american_price"],
        label="true normalized price",
    )
    strike = _numeric_array(prediction_matrix["strike"], label="strike")
    actual_price = actual_normalized * strike
    european = _numeric_array(
        prediction_matrix["normalized_european_price"],
        label="normalized European price",
    )
    intrinsic = _numeric_array(
        prediction_matrix["normalized_intrinsic_value"],
        label="normalized intrinsic value",
    )

    pricing_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        prediction = _numeric_array(
            prediction_matrix[f"prediction__{spec.model_id}"],
            label=spec.model,
        )
        normalized_metrics = _regression_record(
            actual_normalized,
            prediction,
            error_bands=error_bands,
        )
        price_metrics = _regression_record(
            actual_price,
            prediction * strike,
            error_bands=tuple(float(band) * float(strike[0]) for band in error_bands),
        )
        pricing_rows.append(
            {
                **asdict(spec),
                **{f"normalized_{key}": value for key, value in normalized_metrics.items()},
                **{f"price_{key}": value for key, value in price_metrics.items()},
            }
        )
        consistency_rows.append(
            {
                **asdict(spec),
                **_financial_consistency_record(
                    prediction,
                    european,
                    intrinsic,
                ),
            }
        )

    pricing = pd.DataFrame(pricing_rows).sort_values(
        ["normalized_mae", "model"],
        kind="stable",
    ).reset_index(drop=True)
    pricing.insert(0, "pricing_rank", np.arange(1, len(pricing) + 1))
    consistency = pd.DataFrame(consistency_rows).sort_values(
        ["below_financial_floor_rate", "model"],
        kind="stable",
    ).reset_index(drop=True)
    return pricing, consistency


def build_pairwise_absolute_error_comparison(
    prediction_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build paired row-level comparisons for every pair of static models."""

    actual = _numeric_array(
        prediction_matrix["true_normalized_american_price"],
        label="true normalized price",
    )
    absolute_errors = {
        spec.model_id: np.abs(
            _numeric_array(
                prediction_matrix[f"prediction__{spec.model_id}"],
                label=spec.model,
            )
            - actual
        )
        for spec in MODEL_SPECS
    }

    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(MODEL_SPECS):
        for right in MODEL_SPECS[left_index + 1 :]:
            left_error = absolute_errors[left.model_id]
            right_error = absolute_errors[right.model_id]
            difference = left_error - right_error
            tolerance = 1e-15
            rows.append(
                {
                    "model_a_id": left.model_id,
                    "model_a": left.model,
                    "model_b_id": right.model_id,
                    "model_b": right.model,
                    "observations": int(len(difference)),
                    "mean_absolute_error_difference_a_minus_b": float(
                        np.mean(difference)
                    ),
                    "median_absolute_error_difference_a_minus_b": float(
                        np.median(difference)
                    ),
                    "model_a_win_rate": float(np.mean(difference < -tolerance)),
                    "model_b_win_rate": float(np.mean(difference > tolerance)),
                    "tie_rate": float(np.mean(np.abs(difference) <= tolerance)),
                    "model_a_lower_mean_absolute_error": bool(
                        np.mean(left_error) < np.mean(right_error)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _segment_labels(matrix: pd.DataFrame) -> dict[str, pd.Series]:
    labels: dict[str, pd.Series] = {}
    if "moneyness" in matrix.columns:
        labels["moneyness"] = pd.cut(
            pd.to_numeric(matrix["moneyness"], errors="coerce"),
            bins=[-np.inf, 0.80, 0.95, 1.05, 1.20, np.inf],
            labels=["deep ITM", "ITM", "near ATM", "OTM", "deep OTM"],
        )
    if "time_to_maturity" in matrix.columns:
        labels["maturity"] = pd.cut(
            pd.to_numeric(matrix["time_to_maturity"], errors="coerce"),
            bins=[-np.inf, 0.50, 1.00, np.inf],
            labels=["short", "medium", "long"],
        )
    if "volatility" in matrix.columns:
        labels["volatility"] = pd.cut(
            pd.to_numeric(matrix["volatility"], errors="coerce"),
            bins=[-np.inf, 0.20, 0.50, np.inf],
            labels=["low", "medium", "high"],
        )
    if "exercise_now" in matrix.columns:
        labels["exercise_region"] = pd.Series(
            np.where(matrix["exercise_now"].astype(bool), "exercise", "continue"),
            index=matrix.index,
            dtype="object",
        )
    return labels


def build_segmented_static_pricing(
    prediction_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute static pricing error by common economic segments."""

    segment_labels = _segment_labels(prediction_matrix)
    if not segment_labels:
        return pd.DataFrame()

    actual = _numeric_array(
        prediction_matrix["true_normalized_american_price"],
        label="true normalized price",
    )
    strike = _numeric_array(prediction_matrix["strike"], label="strike")
    rows: list[dict[str, Any]] = []

    for segment_type, labels in segment_labels.items():
        temporary = pd.DataFrame({"segment": labels})
        for segment, index in temporary.groupby("segment", observed=True).groups.items():
            positions = np.asarray(list(index), dtype=int)
            if len(positions) == 0:
                continue
            for spec in MODEL_SPECS:
                prediction = _numeric_array(
                    prediction_matrix[f"prediction__{spec.model_id}"],
                    label=spec.model,
                )
                error = prediction[positions] - actual[positions]
                price_error = error * strike[positions]
                rows.append(
                    {
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "model_id": spec.model_id,
                        "model": spec.model,
                        "source_notebook": spec.source_notebook,
                        "observations": int(len(positions)),
                        "normalized_mae": float(np.mean(np.abs(error))),
                        "normalized_rmse": float(np.sqrt(np.mean(error**2))),
                        "price_mae": float(np.mean(np.abs(price_error))),
                        "price_rmse": float(np.sqrt(np.mean(price_error**2))),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["segment_type", "segment", "normalized_mae", "model"],
        kind="stable",
    ).reset_index(drop=True)


def build_boundary_static_pricing(
    prediction_matrix: pd.DataFrame,
    *,
    bands: Sequence[float] = DEFAULT_BOUNDARY_BANDS,
) -> pd.DataFrame:
    """Recompute pricing errors in cumulative bands around the exercise boundary."""

    if "boundary_distance_normalized" not in prediction_matrix.columns:
        return pd.DataFrame()

    boundary_distance = _numeric_array(
        prediction_matrix["boundary_distance_normalized"],
        label="boundary distance",
    )
    actual = _numeric_array(
        prediction_matrix["true_normalized_american_price"],
        label="true normalized price",
    )
    strike = _numeric_array(prediction_matrix["strike"], label="strike")
    exercise = (
        prediction_matrix["exercise_now"].astype(bool).to_numpy()
        if "exercise_now" in prediction_matrix.columns
        else np.zeros(len(prediction_matrix), dtype=bool)
    )

    rows: list[dict[str, Any]] = []
    for band in bands:
        limit = float(band)
        if limit <= 0.0:
            raise ValueError("Boundary bands must be positive")
        positions = np.flatnonzero(boundary_distance <= limit)
        if len(positions) == 0:
            continue
        for spec in MODEL_SPECS:
            prediction = _numeric_array(
                prediction_matrix[f"prediction__{spec.model_id}"],
                label=spec.model,
            )
            error = prediction[positions] - actual[positions]
            price_error = error * strike[positions]
            rows.append(
                {
                    "boundary_limit": limit,
                    "boundary_band": f"≤{limit:.3f}",
                    "model_id": spec.model_id,
                    "model": spec.model,
                    "source_notebook": spec.source_notebook,
                    "observations": int(len(positions)),
                    "exercise_observations": int(exercise[positions].sum()),
                    "continuation_observations": int((~exercise[positions]).sum()),
                    "normalized_mae": float(np.mean(np.abs(error))),
                    "normalized_rmse": float(np.sqrt(np.mean(error**2))),
                    "price_mae": float(np.mean(np.abs(price_error))),
                    "price_rmse": float(np.sqrt(np.mean(price_error**2))),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["boundary_limit", "normalized_mae", "model"],
        kind="stable",
    ).reset_index(drop=True)



def build_static_model_registry() -> pd.DataFrame:
    """Return the explicit model contract used by the Phase 4 comparison."""

    return pd.DataFrame([asdict(spec) for spec in MODEL_SPECS])

def run_phase_4_static_comparison(
    project_root: Path,
    packages: Mapping[str, FinalNotebookPackage],
) -> dict[str, pd.DataFrame]:
    """Build the full common-test evidence package for Phase 4."""

    matrix = build_static_prediction_matrix(project_root, packages)
    pricing, consistency = build_static_model_metrics(matrix)
    pairwise = build_pairwise_absolute_error_comparison(matrix)
    segmented = build_segmented_static_pricing(matrix)
    boundary = build_boundary_static_pricing(matrix)
    return {
        "static_prediction_matrix": matrix,
        "static_model_metrics": pricing,
        "static_financial_consistency": consistency,
        "static_pairwise_error_comparison": pairwise,
        "static_segmented_pricing": segmented,
        "static_boundary_pricing": boundary,
    }


def assert_phase_4_ready(results: Mapping[str, pd.DataFrame]) -> None:
    """Raise when the authoritative static comparison is incomplete."""

    required_tables = (
        "static_prediction_matrix",
        "static_model_metrics",
        "static_financial_consistency",
        "static_pairwise_error_comparison",
        "static_segmented_pricing",
        "static_boundary_pricing",
    )
    for name in required_tables:
        table = results.get(name)
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise RuntimeError(f"Phase 4 table is missing or empty: {name}")

    matrix = results["static_prediction_matrix"]
    if matrix["sample_id"].duplicated().any():
        raise RuntimeError("Phase 4 matrix contains duplicate sample IDs")

    prediction_columns = [f"prediction__{spec.model_id}" for spec in MODEL_SPECS]
    missing_prediction_columns = [
        column for column in prediction_columns if column not in matrix.columns
    ]
    if missing_prediction_columns:
        raise RuntimeError(
            "Phase 4 matrix is missing prediction columns: "
            f"{missing_prediction_columns}"
        )
    values = matrix[prediction_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError("Phase 4 matrix contains non-finite predictions")

    metrics = results["static_model_metrics"]
    expected_ids = {spec.model_id for spec in MODEL_SPECS}
    actual_ids = set(metrics["model_id"].astype(str))
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Phase 4 metric model set mismatch: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    if metrics["model_id"].duplicated().any():
        raise RuntimeError("Phase 4 metric table contains duplicate model IDs")

    pairwise = results["static_pairwise_error_comparison"]
    expected_pairs = len(MODEL_SPECS) * (len(MODEL_SPECS) - 1) // 2
    if len(pairwise) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairwise comparisons, found {len(pairwise)}"
        )


__all__ = [
    "DEFAULT_BOUNDARY_BANDS",
    "DEFAULT_ERROR_BANDS",
    "DEFAULT_FINANCIAL_TOLERANCE",
    "MODEL_SPECS",
    "MODEL_SPEC_BY_ID",
    "StaticModelSpec",
    "assert_phase_4_ready",
    "build_boundary_static_pricing",
    "build_pairwise_absolute_error_comparison",
    "build_segmented_static_pricing",
    "build_static_model_registry",
    "build_static_model_metrics",
    "build_static_prediction_matrix",
    "run_phase_4_static_comparison",
]
