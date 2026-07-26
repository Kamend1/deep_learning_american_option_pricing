"""Validation selection and ablation tables for integrated static models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import binary_classification_metrics
from src.evaluation.internal_consistency import internal_consistency_metrics
from src.evaluation.regression_metrics import regression_metrics


def evaluate_integrated_prediction_frame(
    frame: pd.DataFrame,
    *,
    classification_threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate pricing, continuation, classification, and consistency metrics."""

    required = [
        "true_normalized_american_price",
        "predicted_normalized_american_price",
        "predicted_direct_normalized_american_price",
        "true_normalized_continuation_value",
        "predicted_normalized_continuation_value",
        "exercise_target",
        "exercise_probability",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Integrated prediction frame is missing columns: {missing}")

    constrained = regression_metrics(
        frame["true_normalized_american_price"],
        frame["predicted_normalized_american_price"],
    )
    direct = regression_metrics(
        frame["true_normalized_american_price"],
        frame["predicted_direct_normalized_american_price"],
    )
    continuation = regression_metrics(
        frame["true_normalized_continuation_value"],
        frame["predicted_normalized_continuation_value"],
    )
    classification = binary_classification_metrics(
        frame["exercise_target"],
        frame["exercise_probability"],
        threshold=classification_threshold,
    )
    consistency = internal_consistency_metrics(
        frame,
        classification_threshold=classification_threshold,
    )

    result: dict[str, float] = {}
    result.update({f"constrained_{key}": value for key, value in constrained.items()})
    result.update({f"direct_{key}": value for key, value in direct.items()})
    result.update(
        {f"continuation_{key}": value for key, value in continuation.items()}
    )
    result.update(
        {f"exercise_{key}": value for key, value in classification.items()}
    )
    result.update(
        {f"consistency_{key}": value for key, value in consistency.items()}
    )
    return result


def build_integrated_ablation_table(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    classification_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build one aligned validation/test table for integrated configurations."""

    if not prediction_frames:
        raise ValueError("prediction_frames cannot be empty.")
    rows = []
    reference_ids: np.ndarray | None = None
    for name, frame in prediction_frames.items():
        if "sample_id" not in frame:
            raise ValueError(f"Prediction frame {name!r} lacks sample_id.")
        sorted_frame = frame.sort_values("sample_id").reset_index(drop=True)
        identifiers = sorted_frame["sample_id"].to_numpy()
        if reference_ids is None:
            reference_ids = identifiers
        elif not np.array_equal(reference_ids, identifiers):
            raise ValueError("All compared prediction frames must contain identical IDs.")
        rows.append(
            {
                "configuration": name,
                **evaluate_integrated_prediction_frame(
                    sorted_frame,
                    classification_threshold=classification_threshold,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("configuration").sort_index()


def select_validation_configuration(
    table: pd.DataFrame,
    *,
    primary_metric: str = "constrained_rmse",
    secondary_metric: str = "exercise_f1",
    tertiary_metric: str = "consistency_decision_disagreement_rate",
) -> dict[str, object]:
    """Select one configuration using a predefined lexicographic rule.

    Lower primary and tertiary metrics are better; higher secondary metric is
    better. The function must only be called with validation results.
    """

    if table.empty:
        raise ValueError("Validation table cannot be empty.")
    missing = [
        column
        for column in (primary_metric, secondary_metric, tertiary_metric)
        if column not in table
    ]
    if missing:
        raise ValueError(f"Validation table is missing metrics: {missing}")
    numeric = table.loc[
        :,
        [primary_metric, secondary_metric, tertiary_metric],
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Selection metrics must be finite.")

    ranked = table.reset_index().sort_values(
        [primary_metric, secondary_metric, tertiary_metric, "configuration"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    selected = ranked.iloc[0]
    return {
        "configuration": str(selected["configuration"]),
        "selection_rule": (
            f"min {primary_metric}, then max {secondary_metric}, "
            f"then min {tertiary_metric}"
        ),
        primary_metric: float(selected[primary_metric]),
        secondary_metric: float(selected[secondary_metric]),
        tertiary_metric: float(selected[tertiary_metric]),
    }


def merge_static_model_summaries(
    summaries: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Create a consolidated table from Step 4–8 metric dictionaries."""

    if not summaries:
        raise ValueError("summaries cannot be empty.")
    rows = [{"model": name, **dict(values)} for name, values in summaries.items()]
    return pd.DataFrame(rows).set_index("model")


__all__ = [
    "build_integrated_ablation_table",
    "evaluate_integrated_prediction_frame",
    "merge_static_model_summaries",
    "select_validation_configuration",
]
