from pathlib import Path

import pandas as pd

from src.evaluation.final_project_conclusions import run_phase_7_conclusions
from src.evaluation.final_reporting import (
    export_final_project,
    verify_export_manifest,
    write_export_manifest,
)
from src.evaluation.final_validation import build_post_export_readiness_audit

from phase7_8_fixtures import chart_paths, phase7_8_tables


def test_final_export_package_is_complete(tmp_path):
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
    tables = {
        **{
            key: conclusions[key]
            for key in (
                "task_recommendations",
                "integrated_model_tradeoff",
                "project_findings",
                "project_limitations",
            )
        },
        "hypothesis_decisions": hypotheses,
        "static_model_metrics": static,
        "static_financial_consistency": consistency,
        "exercise_model_metrics": exercise,
        "exercise_boundary_metrics": exercise.copy(),
        "static_ood_model_summary": ood,
        "lsm_heldout_pricing": lsm,
        "runtime_comparison": runtime,
    }
    for name in (
        "runtime_scaling",
        "accuracy_speed_tradeoff",
        "runtime_curves",
        "operational_crossover",
        "upfront_cost_inventory",
        "upfront_cost_scenarios",
        "lifecycle_break_even",
        "business_case_scenarios",
        "business_case_recommendations",
        "business_case_readiness_audit",
    ):
        tables[name] = pd.DataFrame([{"name": name, "value": 1.0}])
    audit = static.iloc[[0], :1].rename(columns={"model_id": "check"})
    audit["valid"] = True
    audit["details"] = "ok"
    charts = chart_paths(tmp_path / "charts")
    output = tmp_path / "final"
    export_final_project(
        output,
        tables=tables,
        final_results_summary=conclusions["final_results_summary"],
        final_conclusion_markdown=conclusions["final_conclusion_markdown"],
        chart_paths=charts,
        readiness_audit=audit,
    )
    manifest = write_export_manifest(output)
    verification = verify_export_manifest(output, manifest)
    post = build_post_export_readiness_audit(
        manifest,
        required_relative_paths=[
            "task_recommendations.csv",
            "final_results_summary.json",
            "final_project_conclusion.md",
            "final_readiness_audit.csv",
        ],
        export_verification=verification,
    )
    assert post["valid"].all()
    assert (output / "final_export_manifest.csv").is_file()


def test_manifest_verification_detects_tampering(tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    target = output / "result.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    manifest = write_export_manifest(output)
    target.write_text("a\n2\n", encoding="utf-8")
    verification = verify_export_manifest(output, manifest)
    assert not verification["valid"].all()
