import pandas as pd
import pytest

from src.evaluation.final_project_conclusions import run_phase_7_conclusions
from src.evaluation.final_validation import (
    assert_phase_7_8_ready,
    build_pre_export_readiness_audit,
)

from phase7_8_fixtures import audit_tables, chart_paths, phase7_8_tables


def test_pre_export_readiness_passes(tmp_path):
    static, consistency, exercise, ood, lsm, coverage, runtime, hypotheses = (
        phase7_8_tables()
    )
    conclusions = run_phase_7_conclusions(
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        lsm_heldout_pricing=lsm,
        lsm_coverage=coverage,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
    )
    artifact, coherence, alignment, fields = audit_tables()
    audit = build_pre_export_readiness_audit(
        artifact_audit=artifact,
        package_coherence=coherence,
        static_prediction_alignment=alignment,
        static_field_alignment=fields,
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
        task_recommendations=conclusions["task_recommendations"],
        integrated_model_tradeoff=conclusions["integrated_model_tradeoff"],
        project_findings=conclusions["project_findings"],
        project_limitations=conclusions["project_limitations"],
        final_results_summary=conclusions["final_results_summary"],
        chart_paths=chart_paths(tmp_path),
    )
    assert_phase_7_8_ready(audit)


def test_final_gate_rejects_failed_check():
    with pytest.raises(RuntimeError):
        assert_phase_7_8_ready(
            pd.DataFrame([{"check": "broken", "valid": False, "details": "x"}])
        )


def test_pre_export_readiness_accepts_production_field_alignment_schema(tmp_path):
    static, consistency, exercise, ood, lsm, coverage, runtime, hypotheses = (
        phase7_8_tables()
    )
    conclusions = run_phase_7_conclusions(
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        lsm_heldout_pricing=lsm,
        lsm_coverage=coverage,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
    )
    artifact, coherence, alignment, _ = audit_tables()
    fields = pd.DataFrame(
        [
            {
                "notebook": "04",
                "field": "moneyness",
                "matches": True,
            },
            {
                "notebook": "08",
                "field": "<no shared state fields exported>",
                "matches": True,
            },
        ]
    )
    audit = build_pre_export_readiness_audit(
        artifact_audit=artifact,
        package_coherence=coherence,
        static_prediction_alignment=alignment,
        static_field_alignment=fields,
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
        task_recommendations=conclusions["task_recommendations"],
        integrated_model_tradeoff=conclusions["integrated_model_tradeoff"],
        project_findings=conclusions["project_findings"],
        project_limitations=conclusions["project_limitations"],
        final_results_summary=conclusions["final_results_summary"],
        chart_paths=chart_paths(tmp_path),
    )
    row = audit.loc[
        audit["check"].eq("static_field_alignment_valid")
    ].iloc[0]
    assert bool(row["valid"])
    assert "status_column=matches" in row["details"]


def test_pre_export_readiness_rejects_mismatched_field(tmp_path):
    static, consistency, exercise, ood, lsm, coverage, runtime, hypotheses = (
        phase7_8_tables()
    )
    conclusions = run_phase_7_conclusions(
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        lsm_heldout_pricing=lsm,
        lsm_coverage=coverage,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
    )
    artifact, coherence, alignment, _ = audit_tables()
    fields = pd.DataFrame([{"field": "volatility", "matches": False}])
    audit = build_pre_export_readiness_audit(
        artifact_audit=artifact,
        package_coherence=coherence,
        static_prediction_alignment=alignment,
        static_field_alignment=fields,
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
        task_recommendations=conclusions["task_recommendations"],
        integrated_model_tradeoff=conclusions["integrated_model_tradeoff"],
        project_findings=conclusions["project_findings"],
        project_limitations=conclusions["project_limitations"],
        final_results_summary=conclusions["final_results_summary"],
        chart_paths=chart_paths(tmp_path),
    )
    with pytest.raises(RuntimeError, match="static_field_alignment_valid"):
        assert_phase_7_8_ready(audit)
