from __future__ import annotations

from src.evaluation.final_cross_family_evaluation import (
    run_phase_5_cross_family_evaluation,
)
from src.evaluation.final_exercise_comparison import (
    run_phase_5_exercise_comparison,
)
from src.evaluation.final_hypothesis_evidence import (
    assert_phase_6_ready,
    run_phase_6_hypothesis_decisions,
)

from phase5_6_fixtures import build_phase5_6_packages, phase4_tables


def test_phase_6_uses_selected_static_runtime_and_all_ood_models(tmp_path):
    packages = build_phase5_6_packages(tmp_path)
    static_metrics, consistency = phase4_tables()
    exercise = run_phase_5_exercise_comparison(packages)
    cross = run_phase_5_cross_family_evaluation(packages, static_metrics)

    results = run_phase_6_hypothesis_decisions(
        packages,
        static_model_metrics=static_metrics,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise["exercise_model_metrics"],
        static_ood_model_summary=cross["static_ood_model_summary"],
        runtime_comparison=cross["runtime_comparison"],
    )
    assert_phase_6_ready(results)

    evidence = results["hypothesis_evidence"]
    assert evidence["h5_static_model"] == "Constrained floor residual"
    assert evidence["static_seconds_per_option"] < evidence["crr_seconds_per_option"]
    assert evidence["h6_eligible_models"] == 7
    assert len(results["hypothesis_decisions"]) == 6
