import pandas as pd
import pytest

from src.evaluation.model_comparison import (
    align_prediction_frames,
    build_model_comparison_table,
    decide_h2_premium_decomposition,
    decide_h3_financial_constraints,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "actual": [0.20, 0.10, 0.05],
            "strike": [100.0, 100.0, 100.0],
            "intrinsic_value": [15.0, 5.0, 0.0],
            "european_price": [18.0, 8.0, 4.0],
        }
    )


def test_alignment_uses_identifiers_not_row_order() -> None:
    base = _base_frame()
    prediction = pd.DataFrame(
        {"sample_id": [3, 1, 2], "prediction": [0.05, 0.20, 0.10]}
    )
    aligned = align_prediction_frames(base, {"model": prediction})
    assert aligned["model"].tolist() == [0.20, 0.10, 0.05]


def test_alignment_rejects_missing_identifiers() -> None:
    base = _base_frame()
    prediction = pd.DataFrame(
        {"sample_id": [1, 2], "prediction": [0.20, 0.10]}
    )
    with pytest.raises(ValueError):
        align_prediction_frames(base, {"model": prediction})


def test_comparison_contains_accuracy_and_violation_columns() -> None:
    frame = _base_frame().assign(
        direct=[0.19, 0.09, 0.04],
        constrained=[0.20, 0.10, 0.05],
    )
    comparison = build_model_comparison_table(
        frame,
        actual_column="actual",
        prediction_columns={"Direct": "direct", "Constrained": "constrained"},
    )
    assert "mae" in comparison.columns
    assert "below_european_count" in comparison.columns
    assert "total_bound_violations" in comparison.columns


def test_h2_supported_when_threshold_is_met() -> None:
    comparison = pd.DataFrame(
        {"mae": [0.10, 0.08]}, index=["Direct", "Premium"]
    )
    decision = decide_h2_premium_decomposition(
        comparison,
        direct_model="Direct",
        premium_model="Premium",
        minimum_relative_mae_improvement=0.10,
    )
    assert decision.decision == "supported"


def test_h3_supported_when_violations_fall_without_large_degradation() -> None:
    comparison = pd.DataFrame(
        {
            "mae": [0.10, 0.101],
            "total_bound_violations": [10, 0],
        },
        index=["Unconstrained", "Constrained"],
    )
    decision = decide_h3_financial_constraints(
        comparison,
        unconstrained_model="Unconstrained",
        constrained_model="Constrained",
        maximum_relative_mae_degradation=0.02,
    )
    assert decision.decision == "supported"
