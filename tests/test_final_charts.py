from src.evaluation.final_charts import generate_final_charts

from phase7_8_fixtures import phase7_8_tables


def test_generate_final_charts(tmp_path):
    static, _, exercise, ood, lsm, _, runtime, _ = phase7_8_tables()
    paths = generate_final_charts(
        tmp_path,
        static_model_metrics=static,
        exercise_model_metrics=exercise,
        static_ood_model_summary=ood,
        runtime_comparison=runtime,
        lsm_heldout_pricing=lsm,
    )
    assert len(paths) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
