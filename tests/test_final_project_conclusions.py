from src.evaluation.final_project_conclusions import run_phase_7_conclusions

from phase7_8_fixtures import phase7_8_tables


def test_phase_7_conclusions_are_task_specific():
    static, consistency, exercise, ood, lsm, coverage, runtime, hypotheses = (
        phase7_8_tables()
    )
    results = run_phase_7_conclusions(
        static_model_metrics=static,
        static_financial_consistency=consistency,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        lsm_heldout_pricing=lsm,
        lsm_coverage=coverage,
        runtime_comparison=runtime,
        hypothesis_decisions=hypotheses,
    )
    recommendations = results["task_recommendations"].set_index("task")
    assert recommendations.loc[
        "Most accurate static price", "recommended_model"
    ] == "Constrained floor residual MLP"
    assert recommendations.loc[
        "Most accurate exercise decision", "recommended_model"
    ] == "Exercise-only classifier"
    assert results["final_results_summary"]["universal_preferred_model"] is None
    assert "does not produce one model" in results["final_conclusion_markdown"]
