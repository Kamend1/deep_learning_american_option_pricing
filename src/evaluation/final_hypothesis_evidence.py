"""Derive H1-H6 evidence from the validated Phase 4 and Phase 5 tables."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation.final_artifact_adapters import FinalNotebookPackage
from src.evaluation.hypothesis_testing import decide_all_hypotheses


NB05_SELECTED_MODEL_ALIASES = {
    "Constrained floor residual": "constrained_floor_residual_mlp",
    "Non-negative premium": "nonnegative_premium_mlp",
    "Unconstrained premium": "unconstrained_premium_mlp",
}


def _row_value(
    table: pd.DataFrame,
    *,
    key_column: str,
    key: str,
    value_column: str,
) -> float:
    if not {key_column, value_column}.issubset(table.columns):
        return float("nan")
    match = table.loc[table[key_column].astype(str).eq(key), value_column]
    if match.empty:
        return float("nan")
    value = pd.to_numeric(match, errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _maximum_violation_rate(
    consistency: pd.DataFrame,
    model_id: str,
) -> float:
    match = consistency.loc[consistency["model_id"].astype(str).eq(model_id)]
    if match.empty:
        return float("nan")
    rate_columns = [
        column
        for column in match.columns
        if column.endswith("_rate")
        and column
        in {
            "negative_rate",
            "below_european_rate",
            "below_intrinsic_rate",
            "below_financial_floor_rate",
        }
    ]
    if not rate_columns:
        return float("nan")
    values = pd.to_numeric(match.iloc[0][rate_columns], errors="coerce")
    return float(values.max()) if values.notna().any() else float("nan")


def build_hypothesis_evidence(
    packages: Mapping[str, FinalNotebookPackage],
    *,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Build the single explicit evidence mapping consumed by H1-H6 rules."""

    evidence: dict[str, Any] = {
        "black_scholes_mae": _row_value(
            static_model_metrics,
            key_column="model_id",
            key="black_scholes_proxy",
            value_column="normalized_mae",
        ),
        "direct_mlp_mae": _row_value(
            static_model_metrics,
            key_column="model_id",
            key="direct_mlp",
            value_column="normalized_mae",
        ),
    }

    selected_name = str(packages["05"].final_metrics.get("selected_model", ""))
    selected_id = NB05_SELECTED_MODEL_ALIASES.get(selected_name)
    if selected_id is None:
        raise ValueError(
            f"Unsupported Notebook 05 selected_model for H2: {selected_name!r}"
        )
    evidence.update(
        {
            "selected_residual_model": selected_name,
            "selected_residual_model_id": selected_id,
            "selected_residual_mae": _row_value(
                static_model_metrics,
                key_column="model_id",
                key=selected_id,
                value_column="normalized_mae",
            ),
            "direct_violation_rate": _maximum_violation_rate(
                static_financial_consistency, "direct_mlp"
            ),
            "constrained_violation_rate": _maximum_violation_rate(
                static_financial_consistency, selected_id
            ),
        }
    )

    h4 = (packages["06"].final_metrics.get("hypothesis") or {}).get(
        "evidence", {}
    )
    evidence.update(
        {
            "classifier_boundary_f1": h4.get("classifier_boundary_f1"),
            "multitask_boundary_f1": h4.get("multitask_boundary_f1"),
            "price_only_boundary_mae": h4.get("price_only_boundary_mae"),
            "multitask_boundary_mae": h4.get("multitask_boundary_mae"),
            "allowed_f1_degradation": h4.get(
                "maximum_allowed_f1_degradation", 0.001
            ),
            "required_mae_improvement": h4.get(
                "required_boundary_mae_improvement", 0.01
            ),
            "specialist_exercise_f1": _row_value(
                exercise_model_metrics,
                key_column="model_id",
                key="exercise_only_classifier",
                value_column="f1",
            ),
            "integrated_exercise_f1": _row_value(
                exercise_model_metrics,
                key_column="model_id",
                key="integrated_warm_start_exercise_head",
                value_column="f1",
            ),
            "integrated_continuation_f1": _row_value(
                exercise_model_metrics,
                key_column="model_id",
                key="integrated_warm_start_continuation_path",
                value_column="f1",
            ),
        }
    )

    static_runtime = runtime_comparison.loc[
        runtime_comparison["method_id"].eq(selected_id)
    ]
    crr_runtime = runtime_comparison.loc[runtime_comparison["method_id"].eq("crr")]
    evidence.update(
        {
            "h5_static_model": selected_name,
            "static_seconds_per_option": (
                float(static_runtime["seconds_per_observation"].iloc[0])
                if not static_runtime.empty
                else float("nan")
            ),
            "crr_seconds_per_option": (
                float(crr_runtime["seconds_per_observation"].iloc[0])
                if not crr_runtime.empty
                else float("nan")
            ),
        }
    )

    eligible = static_ood_model_summary.loc[
        static_ood_model_summary["h6_eligible"].astype(bool)
    ].copy()
    ratios = pd.to_numeric(
        eligible.get("aggregate_ood_to_in_domain_ratio"), errors="coerce"
    )
    valid = ratios.notna()
    eligible = eligible.loc[valid].copy()
    ratios = ratios.loc[valid]
    evidence.update(
        {
            "h6_eligible_models": int(len(eligible)),
            "h6_models_at_or_above_1_25": int((ratios >= 1.25).sum()),
            "h6_models_above_1_0": int((ratios > 1.0).sum()),
            "h6_minimum_aggregate_ratio": (
                float(ratios.min()) if not ratios.empty else float("nan")
            ),
            "h6_maximum_aggregate_ratio": (
                float(ratios.max()) if not ratios.empty else float("nan")
            ),
            "h6_model_ratios": "; ".join(
                f"{row.model_id}={float(row.aggregate_ood_to_in_domain_ratio):.4f}"
                for row in eligible.itertuples(index=False)
            ),
        }
    )
    return evidence


def build_hypothesis_evidence_table(evidence: Mapping[str, Any]) -> pd.DataFrame:
    """Return a readable long-form inventory of the exact decision inputs."""

    return pd.DataFrame(
        [
            {"evidence_key": key, "value": value}
            for key, value in sorted(evidence.items())
        ]
    )


def run_phase_6_hypothesis_decisions(
    packages: Mapping[str, FinalNotebookPackage],
    *,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Build evidence and deterministic H1-H6 decisions."""

    evidence = build_hypothesis_evidence(
        packages,
        static_model_metrics=static_model_metrics,
        static_financial_consistency=static_financial_consistency,
        exercise_model_metrics=exercise_model_metrics,
        static_ood_model_summary=static_ood_model_summary,
        runtime_comparison=runtime_comparison,
    )
    return {
        "hypothesis_evidence": evidence,
        "hypothesis_evidence_table": build_hypothesis_evidence_table(evidence),
        "hypothesis_decisions": decide_all_hypotheses(evidence),
    }


def assert_phase_6_ready(results: Mapping[str, Any]) -> None:
    """Require six resolved and valid hypothesis decisions."""

    decisions = results.get("hypothesis_decisions")
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise RuntimeError("Hypothesis decisions are missing")
    expected = {"H1", "H2", "H3", "H4", "H5", "H6"}
    if set(decisions["hypothesis"]) != expected:
        raise RuntimeError("Hypothesis decision table does not contain H1-H6 exactly once")
    if decisions["decision"].eq("Inconclusive").any():
        unresolved = ", ".join(
            decisions.loc[decisions["decision"].eq("Inconclusive"), "hypothesis"]
        )
        raise RuntimeError(f"Unresolved hypotheses: {unresolved}")


__all__ = [
    "assert_phase_6_ready",
    "build_hypothesis_evidence",
    "build_hypothesis_evidence_table",
    "run_phase_6_hypothesis_decisions",
]
