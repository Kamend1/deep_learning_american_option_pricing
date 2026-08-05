"""Orchestration and strict validation for the final business-case experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.business_case_benchmark import (
    RuntimeScalingConfig,
    run_business_case_benchmark,
)
from src.evaluation.business_case_break_even import run_business_case_analysis
from src.evaluation.business_case_reporting import (
    build_accuracy_speed_tradeoff,
    build_business_case_recommendations,
    build_research_question_7_summary,
    create_business_case_charts,
    render_business_case_markdown,
)


REQUIRED_CORE_METHODS = {
    "project_numba_crr",
    "notebook05_constrained_residual",
    "notebook08_warm_start_integrated",
}


def _check(name: str, valid: bool, details: str) -> dict[str, Any]:
    return {"check": name, "valid": bool(valid), "details": str(details)}


def build_business_case_readiness_audit(
    *,
    runtime_scaling: pd.DataFrame,
    accuracy_speed_tradeoff: pd.DataFrame,
    runtime_curves: pd.DataFrame,
    operational_crossover: pd.DataFrame,
    upfront_cost_scenarios: pd.DataFrame,
    lifecycle_break_even: pd.DataFrame,
    business_case_scenarios: pd.DataFrame,
    rq7_summary: Mapping[str, Any],
    chart_paths: Mapping[str, Path],
) -> pd.DataFrame:
    """Validate that the practical research question has complete evidence."""

    checks: list[dict[str, Any]] = []
    complete_runtime = runtime_scaling.loc[
        runtime_scaling.get("status", pd.Series(dtype=str)).astype(str).eq("complete")
    ]
    observed_methods = set(complete_runtime.get("method_id", pd.Series(dtype=str)).astype(str))
    checks.append(
        _check(
            "core_runtime_methods_complete",
            REQUIRED_CORE_METHODS.issubset(observed_methods),
            f"methods={sorted(observed_methods)}",
        )
    )

    timing_modes = set(complete_runtime.get("timing_mode", pd.Series(dtype=str)).astype(str))
    checks.append(
        _check(
            "cold_and_warm_timings_present",
            {"cold", "warm"}.issubset(timing_modes),
            f"timing_modes={sorted(timing_modes)}",
        )
    )

    accuracy_methods = set(
        accuracy_speed_tradeoff.loc[
            accuracy_speed_tradeoff.get("status", pd.Series(dtype=str)).astype(str).eq("complete"),
            "method_id",
        ].astype(str)
    ) if not accuracy_speed_tradeoff.empty else set()
    checks.append(
        _check(
            "accuracy_conditioning_complete",
            REQUIRED_CORE_METHODS.issubset(accuracy_methods),
            f"methods={sorted(accuracy_methods)}",
        )
    )

    complete_curves = runtime_curves.loc[
        runtime_curves.get("status", pd.Series(dtype=str)).astype(str).eq("complete")
    ]
    curve_pairs = set(
        zip(
            complete_curves.get("method_id", pd.Series(dtype=str)).astype(str),
            complete_curves.get("timing_mode", pd.Series(dtype=str)).astype(str),
        )
    )
    required_curve_pairs = {
        (method, timing)
        for method in REQUIRED_CORE_METHODS
        for timing in ("cold", "warm")
    }
    checks.append(
        _check(
            "runtime_curves_complete",
            required_curve_pairs.issubset(curve_pairs),
            f"pairs={sorted(curve_pairs)}",
        )
    )

    complete_operational = operational_crossover.loc[
        operational_crossover.get("status", pd.Series(dtype=str)).astype(str).eq("complete")
        & operational_crossover.get("numerical_method_id", pd.Series(dtype=str)).astype(str).eq("project_numba_crr")
        & operational_crossover.get("timing_mode", pd.Series(dtype=str)).astype(str).eq("warm")
    ]
    operational_models = set(
        complete_operational.get("neural_method_id", pd.Series(dtype=str)).astype(str)
    )
    checks.append(
        _check(
            "operational_crossover_complete",
            {
                "notebook05_constrained_residual",
                "notebook08_warm_start_integrated",
            }.issubset(operational_models),
            f"models={sorted(operational_models)}",
        )
    )

    complete_upfront = upfront_cost_scenarios.loc[
        upfront_cost_scenarios.get("status", pd.Series(dtype=str)).astype(str).eq("complete")
    ]
    deployments = set(
        complete_upfront.get("deployment_id", pd.Series(dtype=str)).astype(str)
    )
    checks.append(
        _check(
            "upfront_cost_scenarios_complete",
            {"notebook05_price_only", "notebook08_combined"}.issubset(deployments),
            f"deployments={sorted(deployments)}",
        )
    )

    lifecycle = lifecycle_break_even.loc[
        lifecycle_break_even.get("status", pd.Series(dtype=str)).astype(str).eq("complete")
        & lifecycle_break_even.get("numerical_method_id", pd.Series(dtype=str)).astype(str).eq("project_numba_crr")
    ]
    lifecycle_models = set(
        lifecycle.get("neural_method_id", pd.Series(dtype=str)).astype(str)
    )
    checks.append(
        _check(
            "lifecycle_break_even_complete",
            {
                "notebook05_constrained_residual",
                "notebook08_warm_start_integrated",
            }.issubset(lifecycle_models),
            f"models={sorted(lifecycle_models)}",
        )
    )

    workload_valid = (
        not business_case_scenarios.empty
        and pd.to_numeric(
            business_case_scenarios.get("valuations_per_run"), errors="coerce"
        ).notna().all()
        and pd.to_numeric(
            business_case_scenarios.get("seconds_saved_per_run"), errors="coerce"
        ).notna().all()
    )
    checks.append(
        _check(
            "business_workload_scenarios_complete",
            workload_valid,
            f"rows={len(business_case_scenarios)}",
        )
    )

    rq_valid = bool(rq7_summary.get("business_answer")) and bool(
        rq7_summary.get("operational_crossover_by_method")
    )
    checks.append(
        _check(
            "research_question_7_answer_complete",
            rq_valid,
            str(rq7_summary.get("research_question", "missing")),
        )
    )

    required_charts = {
        "business_runtime_scaling",
        "business_speedup_vs_crr",
        "business_lifecycle_break_even",
        "business_workload_scenarios",
    }
    valid_charts = {
        name
        for name, path in chart_paths.items()
        if Path(path).is_file() and Path(path).stat().st_size > 0
    }
    checks.append(
        _check(
            "business_case_charts_complete",
            required_charts.issubset(valid_charts),
            f"charts={sorted(valid_charts)}",
        )
    )

    return pd.DataFrame(checks)


def assert_business_case_ready(audit: pd.DataFrame) -> None:
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        raise RuntimeError("Business-case readiness audit is missing")
    invalid = audit.loc[~audit["valid"].astype(bool)]
    if not invalid.empty:
        raise RuntimeError(
            "Business-case readiness failed:\n" + invalid.to_string(index=False)
        )


def run_final_business_case(
    project_root: Path,
    *,
    static_model_metrics: pd.DataFrame,
    output_dir: Path,
    config: RuntimeScalingConfig | None = None,
    device: str | torch.device = "cpu",
    include_quantlib: bool = True,
    overrides_seconds: Mapping[str, float | None] | None = None,
    assumed_label_generation_hours: Sequence[float] = (0.5, 1, 2, 4, 8, 12, 24),
    assumed_total_build_hours: Sequence[float] = (1, 4, 8, 24, 48),
) -> dict[str, Any]:
    """Run the measured benchmark, economics, interpretation, charts, and gate."""

    benchmark = run_business_case_benchmark(
        Path(project_root),
        config=config,
        device=device,
        include_quantlib=include_quantlib,
    )
    analysis = run_business_case_analysis(
        Path(project_root),
        benchmark["runtime_scaling"],
        overrides_seconds=overrides_seconds,
        assumed_label_generation_hours=assumed_label_generation_hours,
        assumed_total_build_hours=assumed_total_build_hours,
    )
    accuracy_speed = build_accuracy_speed_tradeoff(
        static_model_metrics,
        analysis["runtime_curves"],
        benchmark["accuracy_speed_sample"],
    )
    recommendations = build_business_case_recommendations(
        analysis["operational_crossover"],
        analysis["lifecycle_break_even"],
        analysis["business_case_scenarios"],
    )
    rq7 = build_research_question_7_summary(
        analysis["runtime_curves"],
        analysis["operational_crossover"],
        analysis["lifecycle_break_even"],
        analysis["business_case_scenarios"],
    )
    markdown = render_business_case_markdown(rq7, recommendations)
    charts = create_business_case_charts(
        Path(output_dir),
        runtime_scaling=benchmark["runtime_scaling"],
        operational_crossover=analysis["operational_crossover"],
        lifecycle_break_even=analysis["lifecycle_break_even"],
        business_case_scenarios=analysis["business_case_scenarios"],
    )
    audit = build_business_case_readiness_audit(
        runtime_scaling=benchmark["runtime_scaling"],
        accuracy_speed_tradeoff=accuracy_speed,
        runtime_curves=analysis["runtime_curves"],
        operational_crossover=analysis["operational_crossover"],
        upfront_cost_scenarios=analysis["upfront_cost_scenarios"],
        lifecycle_break_even=analysis["lifecycle_break_even"],
        business_case_scenarios=analysis["business_case_scenarios"],
        rq7_summary=rq7,
        chart_paths=charts,
    )
    assert_business_case_ready(audit)
    return {
        **benchmark,
        **analysis,
        "accuracy_speed_tradeoff": accuracy_speed,
        "business_case_recommendations": recommendations,
        "research_question_7_summary": rq7,
        "business_case_markdown": markdown,
        "business_case_chart_paths": charts,
        "business_case_readiness_audit": audit,
    }


__all__ = [
    "REQUIRED_CORE_METHODS",
    "assert_business_case_ready",
    "build_business_case_readiness_audit",
    "run_final_business_case",
]
