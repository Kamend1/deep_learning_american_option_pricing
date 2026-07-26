import numpy as np
import pandas as pd
import pytest

from src.evaluation.exercise_boundary import (
    boundary_band_metrics,
    boundary_location_error,
    calculate_boundary_distance_normalized,
    decide_h4_multitask_learning,
    extract_probability_boundary,
    validate_exercise_labels,
)


def test_boundary_distance_sign() -> None:
    distance = calculate_boundary_distance_normalized(
        [12.0, 8.0],
        [10.0, 10.0],
        100.0,
    )
    assert distance[0] > 0.0
    assert distance[1] < 0.0


def test_label_validation_detects_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "intrinsic_value": [12.0, 8.0],
            "continuation_value": [10.0, 10.0],
            "exercise_now": [True, True],
        }
    )
    report = validate_exercise_labels(frame)
    assert report["mismatches"] == 1.0


def test_probability_boundary_interpolates_crossing() -> None:
    estimate = extract_probability_boundary(
        [0.7, 0.8, 0.9, 1.0],
        [0.9, 0.7, 0.3, 0.1],
    )
    assert estimate.crossing_found
    assert estimate.boundary_moneyness == pytest.approx(0.85)


def test_boundary_error_ignores_missing_slices() -> None:
    metrics = boundary_location_error(
        [0.8, 0.9, np.nan],
        [0.82, 0.87, 1.0],
    )
    assert metrics["valid_slices"] == 2.0
    assert metrics["boundary_mae"] == pytest.approx(0.025)


def test_boundary_band_metrics_reports_price_mae() -> None:
    frame = pd.DataFrame(
        {
            "exercise_now": [0, 1, 1, 0],
            "probability": [0.1, 0.8, 0.6, 0.4],
            "boundary_distance_normalized": [-0.002, 0.002, 0.02, -0.02],
            "actual": [0.1, 0.2, 0.3, 0.4],
            "predicted": [0.11, 0.18, 0.29, 0.45],
        }
    )
    result = boundary_band_metrics(
        frame,
        probability_column="probability",
        bands=(0.005,),
        actual_price_column="actual",
        predicted_price_column="predicted",
    )
    assert result.loc[0, "observations"] == 2.0
    assert result.loc[0, "price_mae"] == pytest.approx(0.015)


def test_h4_decision_supported_when_both_thresholds_pass() -> None:
    decision = decide_h4_multitask_learning(
        classifier_boundary_f1=0.80,
        multitask_boundary_f1=0.83,
        price_only_boundary_mae=0.020,
        multitask_boundary_mae=0.018,
    )
    assert decision.decision == "supported"
