"""Cross-model aggregation utilities for Notebook 09."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested metric dictionaries using dot-separated keys."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, prefix=name))
        else:
            result[name] = value
    return result


def metric_inventory_from_json_files(
    named_paths: Mapping[str, Path],
) -> pd.DataFrame:
    """Build a long-form inventory from available JSON metric files."""

    rows: list[dict[str, Any]] = []
    for source_name, path in named_paths.items():
        path = Path(path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            rows.append(
                {
                    "source": source_name,
                    "metric": "payload",
                    "value": str(payload),
                }
            )
            continue
        for metric, value in flatten_mapping(payload).items():
            rows.append(
                {
                    "source": source_name,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=["source", "metric", "value"])


def align_prediction_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    id_column: str,
) -> dict[str, pd.DataFrame]:
    """Sort and verify identical identifiers across model prediction frames."""

    if not frames:
        raise ValueError("frames cannot be empty.")
    aligned: dict[str, pd.DataFrame] = {}
    reference: np.ndarray | None = None
    for name, frame in frames.items():
        if id_column not in frame:
            raise ValueError(f"Frame {name!r} lacks {id_column!r}.")
        ordered = frame.sort_values(id_column).reset_index(drop=True)
        identifiers = ordered[id_column].to_numpy()
        if pd.Series(identifiers).duplicated().any():
            raise ValueError(f"Frame {name!r} contains duplicate identifiers.")
        if reference is None:
            reference = identifiers
        elif not np.array_equal(reference, identifiers):
            raise ValueError("Prediction frames do not contain identical identifiers.")
        aligned[name] = ordered
    return aligned


def build_consolidated_pricing_table(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    *,
    metric_names: tuple[str, ...] = (
        "mae",
        "rmse",
        "median_absolute_error",
        "max_absolute_error",
        "r2",
        "within_0.001",
        "within_0.005",
        "within_0.01",
    ),
) -> pd.DataFrame:
    """Create one model-by-metric table from normalized metric dictionaries."""

    rows: list[dict[str, Any]] = []
    for model, metrics in metrics_by_model.items():
        flat = flatten_mapping(metrics)
        row: dict[str, Any] = {"model": model}
        for metric in metric_names:
            direct = flat.get(metric)
            if direct is None:
                candidates = [
                    value
                    for key, value in flat.items()
                    if key.endswith(f".{metric}") or key == metric
                ]
                direct = candidates[0] if len(candidates) == 1 else np.nan
            row[metric] = direct
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=("model", *metric_names)).set_index("model")
    return pd.DataFrame(rows).set_index("model")


def calculate_ood_deterioration(
    in_domain: pd.DataFrame,
    out_of_domain: pd.DataFrame,
    *,
    model_column: str = "model",
    metric_column: str = "mae",
    regime_column: str = "regime",
) -> pd.DataFrame:
    """Calculate relative OOD error deterioration for each model and regime."""

    for frame_name, frame in (("in_domain", in_domain), ("out_of_domain", out_of_domain)):
        required = {model_column, metric_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} frame is missing columns: {sorted(missing)}")
    if regime_column not in out_of_domain:
        raise ValueError(f"out_of_domain frame lacks {regime_column!r}.")

    base = in_domain[[model_column, metric_column]].rename(
        columns={metric_column: "in_domain_metric"}
    )
    merged = out_of_domain.merge(base, on=model_column, how="left", validate="many_to_one")
    if merged["in_domain_metric"].isna().any():
        raise ValueError("Some OOD models do not have in-domain metrics.")
    if (merged["in_domain_metric"] <= 0).any():
        raise ValueError("In-domain metrics must be strictly positive.")
    merged["ood_deterioration"] = (
        merged[metric_column] - merged["in_domain_metric"]
    ) / merged["in_domain_metric"]
    return merged


def build_runtime_comparison(
    records: pd.DataFrame,
    *,
    model_column: str = "model",
    observations_column: str = "observations",
    seconds_column: str = "seconds",
) -> pd.DataFrame:
    """Normalize runtime records to seconds per observation and throughput."""

    required = {model_column, observations_column, seconds_column}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Runtime records are missing columns: {sorted(missing)}")
    result = records.copy()
    if (result[observations_column] <= 0).any() or (result[seconds_column] <= 0).any():
        raise ValueError("Runtime observations and seconds must be positive.")
    result["seconds_per_observation"] = (
        result[seconds_column] / result[observations_column]
    )
    result["observations_per_second"] = (
        result[observations_column] / result[seconds_column]
    )
    return result


def financial_consistency_table(
    records: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a consistent table of model financial-violation metrics."""

    expected = (
        "negative_price_violations",
        "below_intrinsic_violations",
        "below_european_violations",
        "spot_monotonicity_violation_rate",
        "volatility_monotonicity_violation_rate",
        "decision_disagreement_rate",
    )
    rows = []
    for model, values in records.items():
        flat = flatten_mapping(values)
        rows.append(
            {
                "model": model,
                **{
                    metric: flat.get(metric, np.nan)
                    for metric in expected
                },
            }
        )
    if not rows:
        return pd.DataFrame(columns=("model", *expected)).set_index("model")
    return pd.DataFrame(rows).set_index("model")


__all__ = [
    "align_prediction_frames",
    "build_consolidated_pricing_table",
    "build_runtime_comparison",
    "calculate_ood_deterioration",
    "financial_consistency_table",
    "flatten_mapping",
    "metric_inventory_from_json_files",
]
