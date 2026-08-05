from __future__ import annotations

from src.evaluation.final_exercise_comparison import (
    assert_exercise_evidence_ready,
    run_phase_5_exercise_comparison,
)

from phase5_6_fixtures import build_phase5_6_packages


def test_exercise_comparison_aligns_four_decision_paths(tmp_path):
    packages = build_phase5_6_packages(tmp_path)
    results = run_phase_5_exercise_comparison(packages)
    assert_exercise_evidence_ready(results)

    matrix = results["exercise_prediction_matrix"]
    metrics = results["exercise_model_metrics"]
    boundary = results["exercise_boundary_metrics"]
    ood = results["exercise_ood_comparison"]

    assert len(matrix) == 6
    assert metrics["model_id"].nunique() == 6
    assert set(boundary["boundary_limit"]) == {0.001, 0.005, 0.010}
    assert ood.groupby("model_id")["ood_set"].nunique().eq(4).all()
    assert metrics.iloc[0]["model_id"] == "exercise_only_classifier"
