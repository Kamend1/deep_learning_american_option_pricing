"""Orchestration for Notebook 09 final interpretation and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.final_project_conclusions import run_phase_7_conclusions
from src.evaluation.final_validation import build_pre_export_readiness_audit


def run_phase_7_8_pre_export(
    *,
    artifact_audit: pd.DataFrame,
    package_coherence: pd.DataFrame,
    static_prediction_alignment: pd.DataFrame,
    static_field_alignment: pd.DataFrame,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    exercise_boundary_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    lsm_heldout_pricing: pd.DataFrame,
    lsm_coverage: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
    chart_paths: Mapping[str, Path],
) -> dict[str, Any]:
    conclusions = run_phase_7_conclusions(
        static_model_metrics=static_model_metrics,
        static_financial_consistency=static_financial_consistency,
        exercise_model_metrics=exercise_model_metrics,
        static_ood_model_summary=static_ood_model_summary,
        lsm_heldout_pricing=lsm_heldout_pricing,
        lsm_coverage=lsm_coverage,
        runtime_comparison=runtime_comparison,
        hypothesis_decisions=hypothesis_decisions,
    )
    audit = build_pre_export_readiness_audit(
        artifact_audit=artifact_audit,
        package_coherence=package_coherence,
        static_prediction_alignment=static_prediction_alignment,
        static_field_alignment=static_field_alignment,
        static_model_metrics=static_model_metrics,
        static_financial_consistency=static_financial_consistency,
        exercise_model_metrics=exercise_model_metrics,
        static_ood_model_summary=static_ood_model_summary,
        runtime_comparison=runtime_comparison,
        hypothesis_decisions=hypothesis_decisions,
        task_recommendations=conclusions["task_recommendations"],
        integrated_model_tradeoff=conclusions["integrated_model_tradeoff"],
        project_findings=conclusions["project_findings"],
        project_limitations=conclusions["project_limitations"],
        final_results_summary=conclusions["final_results_summary"],
        chart_paths=chart_paths,
    )
    return {
        **conclusions,
        "pre_export_readiness_audit": audit,
        "exercise_boundary_metrics": exercise_boundary_metrics,
    }


__all__ = ["run_phase_7_8_pre_export"]
