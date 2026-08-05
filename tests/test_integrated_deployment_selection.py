import numpy as np
import pandas as pd

from src.evaluation.integrated_deployment_selection import (
    IntegratedDeploymentSelectionConfig,
    assess_integrated_model_domain,
    assert_deployment_selection_integrity,
    build_union_domain_bounds,
    paired_exercise_decision_evidence,
    paired_price_error_evidence,
    select_integrated_deployment_candidate,
)


class RangeSpec:
    def __init__(self, *, moneyness, time_to_maturity, volatility, risk_free_rate, dividend_yield):
        self.moneyness = moneyness
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate": ["selected_scratch", "warm_start"],
            "parameter_count": [76_324, 34_500],
            "validation_constrained_rmse": [0.00091, 0.00060],
            "validation_exercise_f1": [0.9961, 0.9966],
            "validation_disagreement_rate": [0.0065, 0.0058],
            "validation_protected_price_violations": [0, 0],
            "validation_median_inference_seconds": [0.44, 0.30],
            "test_constrained_mae": [0.00042, 0.00029],
            "weighted_ood_constrained_mae": [0.0034, 0.0056],
        }
    ).set_index("candidate")


def test_selection_uses_validation_and_operational_evidence_only():
    frame = candidate_frame()
    first = select_integrated_deployment_candidate(frame)
    frame.loc["warm_start", "test_constrained_mae"] = 999.0
    frame.loc["warm_start", "weighted_ood_constrained_mae"] = 999.0
    second = select_integrated_deployment_candidate(frame)

    assert first == second
    assert first["preferred_integrated_candidate"] == "warm_start"
    assert first["test_metrics_used_for_selection"] is False
    assert first["ood_metrics_used_for_selection"] is False
    assert_deployment_selection_integrity(first)


def test_selection_rejects_warm_start_when_validation_quality_fails():
    frame = candidate_frame()
    frame.loc["warm_start", "validation_constrained_rmse"] = 0.002
    result = select_integrated_deployment_candidate(
        frame,
        config=IntegratedDeploymentSelectionConfig(
            validation_price_relative_tolerance=0.05,
        ),
    )
    assert result["preferred_integrated_candidate"] == "selected_scratch"
    assert result["checks"]["validation_price_within_tolerance"] is False


def test_domain_guard_routes_outside_contracts_to_crr():
    core = RangeSpec(
        moneyness=(0.5, 1.5),
        time_to_maturity=(0.02, 2.0),
        volatility=(0.05, 0.8),
        risk_free_rate=(0.0, 0.1),
        dividend_yield=(0.0, 0.08),
    )
    boundary = RangeSpec(
        moneyness=(0.45, 1.1),
        time_to_maturity=(0.02, 1.5),
        volatility=(0.05, 0.6),
        risk_free_rate=(0.01, 0.15),
        dividend_yield=(0.0, 0.06),
    )
    bounds = build_union_domain_bounds(core, boundary)
    frame = pd.DataFrame(
        {
            "moneyness": [1.0, 1.0],
            "time_to_maturity": [1.0, 3.0],
            "volatility": [0.2, 0.2],
            "risk_free_rate": [0.03, 0.03],
            "dividend_yield": [0.01, 0.01],
        }
    )
    result = assess_integrated_model_domain(frame, bounds)
    assert bool(result.loc[0, "in_domain"]) is True
    assert bool(result.loc[1, "in_domain"]) is False
    assert result.loc[1, "recommended_path"] == "high_resolution_crr"
    assert "time_to_maturity" in result.loc[1, "out_of_domain_fields"]


def test_paired_evidence_reports_warm_start_improvement():
    truth = np.array([0.1, 0.2, 0.3, 0.4])
    scratch = np.array([0.11, 0.22, 0.33, 0.44])
    warm = np.array([0.105, 0.21, 0.31, 0.42])
    pricing = paired_price_error_evidence(
        truth,
        scratch,
        warm,
        bootstrap_samples=100,
        seed=1,
    )
    assert pricing["mean_absolute_error_improvement"] > 0.0
    assert pricing["warm_start_win_rate"] == 1.0

    labels = np.array([False, True, True, False])
    scratch_prob = np.array([0.1, 0.4, 0.8, 0.7])
    warm_prob = np.array([0.1, 0.7, 0.9, 0.2])
    exercise = paired_exercise_decision_evidence(
        labels,
        scratch_prob,
        warm_prob,
        scratch_threshold=0.5,
        warm_threshold=0.5,
    )
    assert exercise["warm_start_only_correct"] == 2
    assert exercise["scratch_only_correct"] == 0
