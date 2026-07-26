"""Aligned model comparison and formal hypothesis-decision utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.evaluation.financial_checks import financial_bound_report
from src.evaluation.regression_metrics import regression_metrics


@dataclass(frozen=True, slots=True)
class HypothesisDecision:
    hypothesis: str
    decision: str
    rationale: str
    evidence: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def align_prediction_frames(
    base_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    id_column: str = "sample_id",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Merge model predictions by identifier and reject incomplete alignment."""

    if id_column not in base_frame.columns:
        raise ValueError(f"Base frame is missing {id_column!r}.")
    if base_frame[id_column].duplicated().any():
        raise ValueError("Base frame identifiers must be unique.")

    result = base_frame.copy()
    expected_ids = set(result[id_column].tolist())
    for name, frame in predictions.items():
        required = [id_column, prediction_column]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction frame {name!r} is missing: {missing}")
        if frame[id_column].duplicated().any():
            raise ValueError(f"Prediction frame {name!r} has duplicate identifiers.")
        observed_ids = set(frame[id_column].tolist())
        if observed_ids != expected_ids:
            raise ValueError(f"Prediction identifiers do not align for model {name!r}.")
        renamed = frame[[id_column, prediction_column]].rename(
            columns={prediction_column: name}
        )
        result = result.merge(renamed, on=id_column, validate="one_to_one")
    return result


def build_model_comparison_table(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Combine regression accuracy and financial-bound violation metrics."""

    rows: list[dict[str, object]] = []
    for model_name, column in prediction_columns.items():
        metrics = regression_metrics(frame[actual_column], frame[column])
        bounds = financial_bound_report(
            frame,
            normalized_prediction_column=column,
        ).set_index("check")
        row: dict[str, object] = {"model": model_name, **metrics}
        for check in ("negative_price", "below_intrinsic", "below_european"):
            row[f"{check}_count"] = int(bounds.loc[check, "violations"])
            row[f"{check}_rate"] = float(bounds.loc[check, "violation_rate"])
        row["total_bound_violations"] = int(
            row["negative_price_count"]
            + row["below_intrinsic_count"]
            + row["below_european_count"]
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("model").sort_values(
        ["mae", "total_bound_violations"]
    )


def premium_error_metrics(
    actual_premium: np.ndarray | pd.Series,
    predicted_premium: np.ndarray | pd.Series,
    *,
    material_threshold: float,
) -> dict[str, float]:
    """Report all-premium and material-premium errors."""

    actual = np.asarray(actual_premium, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_premium, dtype=np.float64).reshape(-1)
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Premium arrays must have equal non-zero length.")
    result = regression_metrics(actual, predicted)
    material = actual >= material_threshold
    result["material_threshold"] = float(material_threshold)
    result["material_observations"] = float(material.sum())
    result["negative_prediction_rate"] = float((predicted < 0.0).mean())
    if material.any():
        errors = np.abs(predicted[material] - actual[material])
        result["material_mae"] = float(errors.mean())
        denominator = np.maximum(actual[material], np.finfo(np.float64).eps)
        result["material_mean_relative_error"] = float(
            np.mean(errors / denominator)
        )
    else:
        result["material_mae"] = float("nan")
        result["material_mean_relative_error"] = float("nan")
    return result


def decide_h2_premium_decomposition(
    comparison: pd.DataFrame,
    *,
    direct_model: str,
    premium_model: str,
    minimum_relative_mae_improvement: float = 0.01,
) -> HypothesisDecision:
    """Decide H2 using a predefined relative MAE improvement threshold."""

    direct_mae = float(comparison.loc[direct_model, "mae"])
    premium_mae = float(comparison.loc[premium_model, "mae"])
    improvement = (direct_mae - premium_mae) / max(
        direct_mae, np.finfo(np.float64).eps
    )

    if improvement >= minimum_relative_mae_improvement:
        decision = "supported"
        rationale = "The premium model exceeds the predefined MAE improvement threshold."
    elif improvement > 0.0:
        decision = "partially supported"
        rationale = "The premium model improves MAE, but not by the predefined threshold."
    else:
        decision = "not supported"
        rationale = "The premium model does not improve MAE relative to the direct model."

    return HypothesisDecision(
        hypothesis="H2 — Premium decomposition",
        decision=decision,
        rationale=rationale,
        evidence={
            "direct_mae": direct_mae,
            "premium_mae": premium_mae,
            "relative_mae_improvement": float(improvement),
            "required_improvement": float(minimum_relative_mae_improvement),
        },
    )


def decide_h3_financial_constraints(
    comparison: pd.DataFrame,
    *,
    unconstrained_model: str,
    constrained_model: str,
    maximum_relative_mae_degradation: float = 0.02,
) -> HypothesisDecision:
    """Decide H3 from bound violations and an accuracy-degradation tolerance."""

    unconstrained_violations = int(
        comparison.loc[unconstrained_model, "total_bound_violations"]
    )
    constrained_violations = int(
        comparison.loc[constrained_model, "total_bound_violations"]
    )
    unconstrained_mae = float(comparison.loc[unconstrained_model, "mae"])
    constrained_mae = float(comparison.loc[constrained_model, "mae"])
    degradation = (constrained_mae - unconstrained_mae) / max(
        unconstrained_mae, np.finfo(np.float64).eps
    )

    fewer_violations = constrained_violations < unconstrained_violations
    zero_violations = constrained_violations == 0
    acceptable_accuracy = degradation <= maximum_relative_mae_degradation

    if (fewer_violations or zero_violations) and acceptable_accuracy:
        decision = "supported"
        rationale = (
            "The constrained model reduces financial-bound violations without "
            "exceeding the predefined MAE degradation tolerance."
        )
    elif fewer_violations or zero_violations:
        decision = "partially supported"
        rationale = (
            "The constrained model reduces violations, but its MAE degradation "
            "exceeds the predefined tolerance."
        )
    else:
        decision = "not supported"
        rationale = "The constrained model does not reduce financial-bound violations."

    return HypothesisDecision(
        hypothesis="H3 — Financial constraints",
        decision=decision,
        rationale=rationale,
        evidence={
            "unconstrained_violations": float(unconstrained_violations),
            "constrained_violations": float(constrained_violations),
            "unconstrained_mae": unconstrained_mae,
            "constrained_mae": constrained_mae,
            "relative_mae_degradation": float(degradation),
            "allowed_degradation": float(maximum_relative_mae_degradation),
        },
    )


__all__ = [
    "HypothesisDecision",
    "align_prediction_frames",
    "build_model_comparison_table",
    "decide_h2_premium_decomposition",
    "decide_h3_financial_constraints",
    "premium_error_metrics",
]
