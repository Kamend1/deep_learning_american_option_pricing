"""Tests for final integrated-model validation selection and comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.integrated_model_comparison import (
    build_integrated_ablation_table,
    evaluate_integrated_prediction_frame,
    select_validation_configuration,
)


def _prediction_frame(offset: float = 0.0) -> pd.DataFrame:
    truth = np.array([0.10, 0.20, 0.05, 0.15])
    continuation = np.array([0.12, 0.10, 0.06, 0.08])
    exercise = np.array([0, 1, 0, 1])
    predicted = truth + offset
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "true_normalized_american_price": truth,
            "predicted_normalized_american_price": predicted,
            "predicted_direct_normalized_american_price": predicted + 0.001,
            "true_normalized_continuation_value": continuation,
            "predicted_normalized_continuation_value": continuation + offset,
            "exercise_target": exercise,
            "exercise_probability": [0.1, 0.9, 0.2, 0.8],
            "continuation_exercise_probability": [0.1, 0.9, 0.2, 0.8],
            "normalized_european_price": [0.08, 0.15, 0.04, 0.10],
            "normalized_intrinsic_value": [0.05, 0.20, 0.03, 0.15],
            "predicted_floor_residual": [0.02, 0.0, 0.01, 0.0],
        }
    )


def test_integrated_metrics_cover_all_objectives() -> None:
    metrics = evaluate_integrated_prediction_frame(_prediction_frame())

    assert metrics["constrained_rmse"] == pytest.approx(0.0)
    assert metrics["exercise_f1"] == pytest.approx(1.0)
    assert metrics["continuation_rmse"] == pytest.approx(0.0)
    assert metrics["consistency_decision_disagreement_rate"] == 0.0


def test_ablation_requires_identical_ids() -> None:
    first = _prediction_frame()
    second = _prediction_frame(0.01)
    second.loc[0, "sample_id"] = 999

    with pytest.raises(ValueError):
        build_integrated_ablation_table({"a": first, "b": second})


def test_validation_selection_uses_predefined_lexicographic_rule() -> None:
    table = pd.DataFrame(
        {
            "constrained_rmse": [0.01, 0.01, 0.02],
            "exercise_f1": [0.80, 0.90, 0.99],
            "consistency_decision_disagreement_rate": [0.01, 0.02, 0.0],
        },
        index=pd.Index(["balanced", "decision", "pricing"], name="configuration"),
    )

    selected = select_validation_configuration(table)

    assert selected["configuration"] == "decision"
