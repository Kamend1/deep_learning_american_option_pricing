from __future__ import annotations

from src.evaluation.final_cross_family_evaluation import (
    assert_cross_family_evidence_ready,
    run_phase_5_cross_family_evaluation,
)

from phase5_6_fixtures import build_phase5_6_packages, phase4_tables


def test_cross_family_evidence_keeps_ood_runtime_and_lsm_separate(tmp_path):
    packages = build_phase5_6_packages(tmp_path)
    static_metrics, _ = phase4_tables()
    results = run_phase_5_cross_family_evaluation(packages, static_metrics)
    assert_cross_family_evidence_ready(results)

    ood = results["static_ood_comparison"]
    summary = results["static_ood_model_summary"]
    runtime = results["runtime_comparison"]

    eligible = summary.loc[summary["h6_eligible"]]
    assert not eligible.empty
    assert eligible["regimes"].eq(4).all()
    assert {"crr", "constrained_floor_residual_mlp"}.issubset(
        set(runtime["method_id"])
    )
    assert set(runtime["benchmark_family"]).issuperset(
        {"static neural inference", "numerical valuation", "path-based valuation"}
    )
    assert not results["lsm_heldout_pricing"].empty
    assert not results["lsm_ood_pricing"].empty
    assert not ood[["model_id", "ood_set"]].duplicated().any()

    # Upstream notebooks repeat benchmark runtime rows.  The final comparison
    # must retain only the authoritative owner for each logical method.
    assert not runtime["method_id"].duplicated().any()
    owners = runtime.set_index("method_id")["source_notebook"].to_dict()
    assert owners["direct_mlp"] == "04"
    assert owners["constrained_floor_residual_mlp"] == "05"
    assert owners["exercise_only_classifier"] == "06"
