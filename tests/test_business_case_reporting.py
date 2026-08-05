from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.business_case_reporting import (
    build_accuracy_speed_tradeoff,
    build_business_case_recommendations,
    build_research_question_7_summary,
    create_business_case_charts,
    render_business_case_markdown,
)


def _static_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "constrained_floor_residual_mlp",
                "price_mae": 0.01,
                "price_rmse": 0.02,
                "price_max_absolute_error": 0.08,
                "normalized_mae": 0.0001,
            },
            {
                "model_id": "integrated_warm_start_constrained_price",
                "price_mae": 0.03,
                "price_rmse": 0.05,
                "price_max_absolute_error": 0.12,
                "normalized_mae": 0.0003,
            },
        ]
    )


def _curves() -> pd.DataFrame:
    rows = []
    for method_id, family, marginal in (
        ("project_numba_crr", "numerical valuation", 4e-5),
        ("notebook05_constrained_residual", "static neural inference", 1e-6),
        ("notebook08_warm_start_integrated", "static neural inference", 1.5e-6),
    ):
        rows.append(
            {
                "method_id": method_id,
                "method": method_id,
                "family": family,
                "output_scope": "price only" if "05" in method_id else "price and exercise decision",
                "timing_mode": "warm",
                "fixed_seconds": 0.01,
                "marginal_seconds_per_observation": marginal,
                "observations_per_second_at_scale": 1 / marginal,
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def _accuracy_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_id": "project_numba_crr",
                "method": "Project CRR",
                "family": "numerical valuation",
                "output_scope": "price and root exercise decision",
                "price_mae": 0.0,
                "price_rmse": 0.0,
                "maximum_absolute_error": 0.0,
                "status": "complete",
            }
        ]
    )


def _operational() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timing_mode": "warm",
                "neural_method_id": method,
                "numerical_method_id": "project_numba_crr",
                "curve_crossover_observations": threshold,
                "status": "complete",
            }
            for method, threshold in (
                ("notebook05_constrained_residual", 250),
                ("notebook08_warm_start_integrated", 350),
            )
        ]
    )


def _lifecycle() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "neural_method_id": method,
                "numerical_method_id": "project_numba_crr",
                "break_even_valuations": value,
                "status": "complete",
            }
            for method in (
                "notebook05_constrained_residual",
                "notebook08_warm_start_integrated",
            )
            for value in (100_000, 500_000)
        ]
    )


def _scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "Large grid",
                "valuations_per_run": 1_000_000,
                "neural_method_id": method,
                "neural_method": method,
                "numerical_method": "Project CRR",
                "numerical_seconds_per_run": 40.0,
                "neural_seconds_per_run": 1.5,
                "seconds_saved_per_run": 38.5,
                "annual_hours_saved": 10.0,
                "neural_faster_for_workload": True,
            }
            for method in (
                "notebook05_constrained_residual",
                "notebook08_warm_start_integrated",
            )
        ]
    )


def test_reporting_answers_the_business_question(tmp_path: Path) -> None:
    tradeoff = build_accuracy_speed_tradeoff(
        _static_metrics(),
        _curves(),
        _accuracy_sample(),
    )
    assert set(tradeoff["method_id"]) == {
        "project_numba_crr",
        "notebook05_constrained_residual",
        "notebook08_warm_start_integrated",
    }
    assert {
        "price_rmse",
        "maximum_absolute_error",
        "warm_seconds_per_observation_at_scale",
        "warm_observations_per_second_at_scale",
    }.issubset(tradeoff.columns)
    notebook05 = tradeoff.loc[
        tradeoff["method_id"].eq("notebook05_constrained_residual")
    ].iloc[0]
    assert notebook05["price_rmse"] == 0.02
    assert notebook05["maximum_absolute_error"] == 0.08
    assert notebook05["warm_seconds_per_observation_at_scale"] == 1e-6

    recommendations = build_business_case_recommendations(
        _operational(),
        _lifecycle(),
        _scenarios(),
    )
    assert "One-off valuation or small portfolio" in set(
        recommendations["situation"]
    )

    rq7 = build_research_question_7_summary(
        _curves(),
        _operational(),
        _lifecycle(),
        _scenarios(),
    )
    assert rq7["operational_crossover_by_method"][
        "notebook05_constrained_residual"
    ] == 250
    assert "not justified for isolated valuations" in rq7["business_answer"]
    markdown = render_business_case_markdown(rq7, recommendations)
    assert "Does deep learning make business sense here?" in markdown
    assert "250" in markdown


def test_business_charts_are_created(tmp_path: Path) -> None:
    runtime = []
    for method_id, method, multiplier in (
        ("project_numba_crr", "Project CRR", 0.00004),
        ("notebook05_constrained_residual", "Notebook 05", 0.000001),
        ("notebook08_warm_start_integrated", "Notebook 08", 0.0000015),
    ):
        for n in (10, 100, 1_000, 10_000):
            runtime.append(
                {
                    "method_id": method_id,
                    "method": method,
                    "timing_mode": "warm",
                    "status": "complete",
                    "requested_observations": n,
                    "median_seconds": 0.01 + multiplier * n,
                }
            )
    lifecycle = _lifecycle().assign(
        neural_method=lambda x: x["neural_method_id"],
        upfront_hours=[1.0, 4.0, 1.0, 4.0],
    )
    charts = create_business_case_charts(
        tmp_path,
        runtime_scaling=pd.DataFrame(runtime),
        operational_crossover=_operational(),
        lifecycle_break_even=lifecycle,
        business_case_scenarios=_scenarios(),
    )
    assert len(charts) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in charts.values())
