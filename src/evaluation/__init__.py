"""Pricing, classification, financial, and boundary evaluation utilities."""

from src.evaluation.classification_metrics import (
    binary_classification_metrics,
    calibration_frame,
    choose_f1_threshold,
    confusion_matrix_frame,
)
from src.evaluation.exercise_boundary import (
    BoundaryEstimate,
    H4Decision,
    boundary_band_metrics,
    boundary_location_error,
    boundary_monotonicity_report,
    calculate_boundary_distance_normalized,
    decide_h4_multitask_learning,
    extract_label_boundary,
    extract_probability_boundary,
    validate_exercise_labels,
)
from src.evaluation.financial_checks import (
    financial_bound_report,
    monotonicity_violation_rate,
)
from src.evaluation.model_comparison import (
    HypothesisDecision,
    align_prediction_frames,
    build_model_comparison_table,
    decide_h2_premium_decomposition,
    decide_h3_financial_constraints,
    premium_error_metrics,
)
from src.evaluation.regression_metrics import (
    compare_models,
    regression_metrics,
    segmented_regression_metrics,
)

__all__ = [
    "BoundaryEstimate",
    "H4Decision",
    "HypothesisDecision",
    "align_prediction_frames",
    "binary_classification_metrics",
    "boundary_band_metrics",
    "boundary_location_error",
    "boundary_monotonicity_report",
    "build_model_comparison_table",
    "calculate_boundary_distance_normalized",
    "calibration_frame",
    "choose_f1_threshold",
    "compare_models",
    "confusion_matrix_frame",
    "decide_h2_premium_decomposition",
    "decide_h3_financial_constraints",
    "decide_h4_multitask_learning",
    "extract_label_boundary",
    "extract_probability_boundary",
    "financial_bound_report",
    "monotonicity_violation_rate",
    "premium_error_metrics",
    "regression_metrics",
    "segmented_regression_metrics",
    "validate_exercise_labels",
]

from src.evaluation.lsm_comparison import (
    aggregate_seed_results,
    align_contract_results,
    compare_lsm_methods,
    confidence_interval_coverage,
    exercise_policy_metrics,
    pricing_error_metrics,
    runtime_summary,
    stopping_time_total_variation,
)

__all__ += [
    "aggregate_seed_results",
    "align_contract_results",
    "compare_lsm_methods",
    "confidence_interval_coverage",
    "exercise_policy_metrics",
    "pricing_error_metrics",
    "runtime_summary",
    "stopping_time_total_variation",
]

