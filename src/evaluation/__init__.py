"""Regression and financial-consistency evaluation utilities."""

from src.evaluation.financial_checks import (
    financial_bound_report,
    monotonicity_violation_rate,
)
from src.evaluation.regression_metrics import (
    compare_models,
    regression_metrics,
    segmented_regression_metrics,
)

__all__ = [
    "compare_models",
    "financial_bound_report",
    "monotonicity_violation_rate",
    "regression_metrics",
    "segmented_regression_metrics",
]
