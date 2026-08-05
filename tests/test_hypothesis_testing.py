from __future__ import annotations

from src.evaluation.hypothesis_testing import decide_all_hypotheses


def test_hypothesis_rules_resolve_all_six_decisions():
    evidence = {
        "black_scholes_mae": 0.10,
        "direct_mlp_mae": 0.02,
        "selected_residual_mae": 0.01,
        "selected_residual_model": "Constrained floor residual",
        "direct_violation_rate": 0.15,
        "constrained_violation_rate": 0.0,
        "classifier_boundary_f1": 0.96,
        "multitask_boundary_f1": 0.94,
        "price_only_boundary_mae": 0.001,
        "multitask_boundary_mae": 0.0015,
        "allowed_f1_degradation": 0.001,
        "required_mae_improvement": 0.01,
        "integrated_exercise_f1": 0.95,
        "specialist_exercise_f1": 0.97,
        "static_seconds_per_option": 2.5e-6,
        "crr_seconds_per_option": 0.5,
        "h5_static_model": "Constrained floor residual",
        "h6_eligible_models": 6,
        "h6_models_at_or_above_1_25": 6,
        "h6_models_above_1_0": 6,
        "h6_minimum_aggregate_ratio": 1.5,
        "h6_model_ratios": "all >= 1.5",
    }
    decisions = decide_all_hypotheses(evidence).set_index("hypothesis")

    assert decisions.loc["H1", "decision"] == "Supported"
    assert decisions.loc["H2", "decision"] == "Supported"
    assert decisions.loc["H3", "decision"] == "Supported"
    assert decisions.loc["H4", "decision"] == "Not supported"
    assert decisions.loc["H5", "decision"] == "Supported"
    assert decisions.loc["H6", "decision"] == "Supported"
