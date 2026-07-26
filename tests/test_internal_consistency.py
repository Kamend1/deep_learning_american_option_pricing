"""Tests for integrated-model internal consistency evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.internal_consistency import (
    contradictory_output_flags,
    internal_consistency_metrics,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_normalized_american_price": [0.12, 0.20, 0.08],
            "predicted_direct_normalized_american_price": [0.11, 0.19, 0.04],
            "predicted_normalized_continuation_value": [0.10, 0.12, 0.05],
            "predicted_floor_residual": [0.02, 0.00, 0.03],
            "exercise_probability": [0.10, 0.80, 0.60],
            "continuation_exercise_probability": [0.12, 0.75, 0.40],
            "normalized_european_price": [0.10, 0.15, 0.05],
            "normalized_intrinsic_value": [0.08, 0.20, 0.05],
        }
    )


def test_consistency_metrics_detect_decision_disagreement_and_direct_violation() -> None:
    metrics = internal_consistency_metrics(_frame())

    assert metrics["decision_disagreement_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["constrained_below_european_rate"] == 0.0
    assert metrics["constrained_below_intrinsic_rate"] == 0.0
    assert metrics["direct_below_european_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["residual_reconstruction_mae"] == pytest.approx(0.0)


def test_row_level_flags_are_returned() -> None:
    flags = contradictory_output_flags(_frame())

    assert list(flags.columns) == [
        "constrained_negative",
        "constrained_below_european",
        "constrained_below_intrinsic",
        "direct_negative",
        "direct_below_european",
        "direct_below_intrinsic",
        "decision_disagreement",
        "any_contradiction",
    ]
    assert flags["decision_disagreement"].sum() == 1
    assert flags["any_contradiction"].sum() >= 1


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError):
        internal_consistency_metrics(_frame().drop(columns=["exercise_probability"]))
