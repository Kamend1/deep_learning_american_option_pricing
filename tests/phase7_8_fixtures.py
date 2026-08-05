from __future__ import annotations

from pathlib import Path

import pandas as pd


def phase7_8_tables():
    static = pd.DataFrame(
        [
            {
                "model_id": "black_scholes_proxy",
                "model": "Black–Scholes proxy",
                "source_notebook": "04",
                "source_selected": False,
                "pricing_rank": 5,
                "normalized_mae": 0.013,
                "price_mae": 1.30,
            },
            {
                "model_id": "direct_mlp",
                "model": "Direct MLP",
                "source_notebook": "04",
                "source_selected": True,
                "pricing_rank": 4,
                "normalized_mae": 0.0008,
                "price_mae": 0.08,
            },
            {
                "model_id": "constrained_floor_residual_mlp",
                "model": "Constrained floor residual MLP",
                "source_notebook": "05",
                "source_selected": True,
                "pricing_rank": 1,
                "normalized_mae": 0.0001,
                "price_mae": 0.01,
            },
            {
                "model_id": "price_only_constrained_residual_mlp",
                "model": "Price-only constrained residual MLP",
                "source_notebook": "06",
                "source_selected": False,
                "pricing_rank": 2,
                "normalized_mae": 0.0001,
                "price_mae": 0.01,
            },
            {
                "model_id": "integrated_warm_start_constrained_price",
                "model": "Integrated warm-start constrained price",
                "source_notebook": "08",
                "source_selected": True,
                "pricing_rank": 3,
                "normalized_mae": 0.0004,
                "price_mae": 0.04,
            },
            {
                "model_id": "integrated_scratch_constrained_price",
                "model": "Integrated balanced-scratch constrained price",
                "source_notebook": "08_scratch",
                "source_selected": False,
                "pricing_rank": 4,
                "normalized_mae": 0.0005,
                "price_mae": 0.05,
            },
        ]
    )
    consistency = pd.DataFrame(
        [
            {
                "model_id": "direct_mlp",
                "model": "Direct MLP",
                "financially_constrained": False,
                "below_financial_floor_rate": 0.30,
            },
            {
                "model_id": "constrained_floor_residual_mlp",
                "model": "Constrained floor residual MLP",
                "financially_constrained": True,
                "below_financial_floor_rate": 0.0,
            },
            {
                "model_id": "integrated_warm_start_constrained_price",
                "model": "Integrated warm-start constrained price",
                "financially_constrained": True,
                "below_financial_floor_rate": 0.0,
            },
            {
                "model_id": "integrated_scratch_constrained_price",
                "model": "Integrated balanced-scratch constrained price",
                "financially_constrained": True,
                "below_financial_floor_rate": 0.0,
            },
        ]
    )
    exercise = pd.DataFrame(
        [
            {
                "model_id": "exercise_only_classifier",
                "model": "Exercise-only classifier",
                "source_notebook": "06",
                "exercise_rank": 1,
                "f1": 0.996,
            },
            {
                "model_id": "multitask_exercise_head",
                "model": "Multi-task exercise head",
                "source_notebook": "06",
                "exercise_rank": 3,
                "f1": 0.994,
            },
            {
                "model_id": "integrated_warm_start_exercise_head",
                "model": "Integrated warm-start exercise head",
                "source_notebook": "08",
                "exercise_rank": 2,
                "f1": 0.9958,
            },
            {
                "model_id": "integrated_warm_start_continuation_path",
                "model": "Integrated warm-start continuation-implied decision",
                "source_notebook": "08",
                "exercise_rank": 4,
                "f1": 0.985,
            },
            {
                "model_id": "integrated_scratch_exercise_head",
                "model": "Integrated balanced-scratch exercise head",
                "source_notebook": "08_scratch",
                "exercise_rank": 5,
                "f1": 0.9955,
            },
            {
                "model_id": "integrated_scratch_continuation_path",
                "model": "Integrated balanced-scratch continuation-implied decision",
                "source_notebook": "08_scratch",
                "exercise_rank": 6,
                "f1": 0.984,
            },
        ]
    )
    ood = pd.DataFrame(
        [
            {
                "model_id": "direct_mlp",
                "model": "Direct MLP",
                "source_notebook": "04",
                "h6_eligible": True,
                "regimes": 4,
                "aggregate_ood_normalized_mae": 0.02,
                "aggregate_ood_to_in_domain_ratio": 25.0,
            },
            {
                "model_id": "constrained_floor_residual_mlp",
                "model": "Constrained floor residual MLP",
                "source_notebook": "05",
                "h6_eligible": True,
                "regimes": 4,
                "aggregate_ood_normalized_mae": 0.0018,
                "aggregate_ood_to_in_domain_ratio": 18.0,
            },
            {
                "model_id": "integrated_warm_start_constrained_price",
                "model": "Integrated warm-start constrained price",
                "source_notebook": "08",
                "h6_eligible": True,
                "regimes": 4,
                "aggregate_ood_normalized_mae": 0.0034,
                "aggregate_ood_to_in_domain_ratio": 8.0,
            },
            {
                "model_id": "integrated_scratch_constrained_price",
                "model": "Integrated balanced-scratch constrained price",
                "source_notebook": "08_scratch",
                "h6_eligible": True,
                "regimes": 4,
                "aggregate_ood_normalized_mae": 0.0030,
                "aggregate_ood_to_in_domain_ratio": 6.0,
            },
        ]
    )
    lsm = pd.DataFrame(
        [
            {"source_notebook": "07", "method": "classical_lsm_price", "mae": 0.04},
            {"source_notebook": "07", "method": "neural_lsm_price", "mae": 0.09},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"metric": "Classical LSM 95% CI coverage", "coverage": 0.68},
            {"metric": "Neural LSM 95% CI coverage", "coverage": 0.42},
        ]
    )
    runtime = pd.DataFrame(
        [
            {
                "benchmark_family": "static neural inference",
                "method": "Constrained floor residual MLP",
                "seconds_per_observation": 2e-6,
                "observations_per_second": 500000.0,
            },
            {
                "benchmark_family": "static neural inference",
                "method": "Integrated warm-start deployment model",
                "seconds_per_observation": 5e-6,
                "observations_per_second": 200000.0,
            },
            {
                "benchmark_family": "numerical valuation",
                "method": "High-resolution CRR",
                "seconds_per_observation": 0.5,
                "observations_per_second": 2.0,
            },
            {
                "benchmark_family": "path-based valuation",
                "method": "Classical LSM end-to-end",
                "seconds_per_observation": 0.35,
                "observations_per_second": 2.86,
            },
            {
                "benchmark_family": "up-front training",
                "method": "Neural LSM policy training",
                "seconds_per_observation": float("nan"),
                "observations_per_second": float("nan"),
            },
        ]
    )
    hypotheses = pd.DataFrame(
        [
            {
                "hypothesis": hypothesis,
                "decision": decision,
                "primary_evidence": "evidence",
                "secondary_evidence": "",
                "threshold": "threshold",
                "limitation": "limitation",
            }
            for hypothesis, decision in [
                ("H1", "Supported"),
                ("H2", "Supported"),
                ("H3", "Supported"),
                ("H4", "Not supported"),
                ("H5", "Supported"),
                ("H6", "Supported"),
            ]
        ]
    )
    return static, consistency, exercise, ood, lsm, coverage, runtime, hypotheses


def audit_tables():
    artifact_audit = pd.DataFrame(
        [{"required_for_final": True, "valid": True}]
    )
    package_coherence = pd.DataFrame([{"valid": True}])
    prediction_alignment = pd.DataFrame([{"valid": True}])
    field_alignment = pd.DataFrame([{"matches": True}])
    return artifact_audit, package_coherence, prediction_alignment, field_alignment


def chart_paths(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in (
        "static_pricing_mae",
        "exercise_f1",
        "ood_deterioration",
        "runtime_comparison",
        "lsm_heldout_mae",
        "business_runtime_scaling",
        "business_speedup_vs_crr",
        "business_lifecycle_break_even",
        "business_workload_scenarios",
    ):
        path = Path(root) / f"{name}.png"
        path.write_bytes(b"png")
        result[name] = path
    return result
