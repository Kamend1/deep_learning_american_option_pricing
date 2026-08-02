"""Regression tests for canonical Notebook 09 aggregation."""

from __future__ import annotations

import numpy as np

from src.evaluation.final_project_evaluation import (
    build_boundary_comparison,
    build_financial_consistency_table,
    build_hypothesis_evidence,
    build_ood_comparison,
    build_runtime_comparison_from_results,
    build_static_ablation_table,
    build_static_pricing_table,
    validate_final_evaluation_semantics,
)
from src.evaluation.hypothesis_testing import decide_all_hypotheses


def _synthetic_results() -> dict:
    return {
        "direct": {
            "pricing": {
                "black_scholes_proxy": {
                    "observations": 100,
                    "mae": 0.0134,
                    "rmse": 0.027,
                },
                "direct_mlp": {
                    "observations": 100,
                    "mae": 0.00078,
                    "rmse": 0.0011,
                },
            },
            "financial_consistency": [
                {
                    "check": "negative_price",
                    "violations": 0,
                    "violation_rate": 0.0,
                },
                {
                    "check": "below_intrinsic",
                    "violations": 18,
                    "violation_rate": 0.18,
                },
                {
                    "check": "below_european",
                    "violations": 14,
                    "violation_rate": 0.14,
                },
            ],
            "ood": [
                {
                    "ood_set": "extreme_moneyness",
                    "observations": 50,
                    "mae": 0.012,
                    "rmse": 0.020,
                },
                {
                    "ood_set": "high_volatility",
                    "observations": 50,
                    "mae": 0.009,
                    "rmse": 0.014,
                },
            ],
            "runtime": [
                {
                    "observations": 100_000,
                    "median_seconds": 0.0012,
                    "device": "cuda",
                }
            ],
        },
        "premium": {
            "pricing": [
                {
                    "model": "Constrained floor residual",
                    "observations": 100,
                    "mae": 0.00010,
                    "rmse": 0.00021,
                },
                {
                    "model": "Non-negative premium",
                    "observations": 100,
                    "mae": 0.00020,
                    "rmse": 0.00035,
                },
                {
                    "model": "Direct MLP",
                    "observations": 100,
                    "mae": 0.00079,
                    "rmse": 0.0012,
                },
            ],
            "financial_consistency": [
                {
                    "model": "Constrained floor residual",
                    "negative_price_count": 0,
                    "negative_price_rate": 0.0,
                    "below_intrinsic_count": 0,
                    "below_intrinsic_rate": 0.0,
                    "below_european_count": 0,
                    "below_european_rate": 0.0,
                    "total_bound_violations": 0,
                },
                {
                    "model": "Direct MLP",
                    "negative_price_count": 0,
                    "negative_price_rate": 0.0,
                    "below_intrinsic_count": 18,
                    "below_intrinsic_rate": 0.18,
                    "below_european_count": 14,
                    "below_european_rate": 0.14,
                    "total_bound_violations": 32,
                },
            ],
            "ood": [
                {
                    "model": "Direct MLP",
                    "ood_set": "extreme_moneyness",
                    "observations": 50,
                    "mae": 0.012,
                    "rmse": 0.020,
                },
                {
                    "model": "Constrained floor residual",
                    "ood_set": "extreme_moneyness",
                    "observations": 50,
                    "mae": 0.00012,
                    "rmse": 0.00030,
                },
            ],
            "runtime": [
                {
                    "model": "Direct MLP",
                    "observations": 100_000,
                    "median_seconds": 0.0013,
                    "device": "cuda",
                },
                {
                    "model": "Constrained floor residual",
                    "observations": 100_000,
                    "median_seconds": 0.0012,
                    "device": "cuda",
                },
            ],
        },
        "multitask": {
            "pricing": [
                {
                    "model": "Price-only constrained residual",
                    "observations": 100,
                    "mae": 0.000102,
                    "rmse": 0.00021,
                },
                {
                    "model": "Multi-task constrained residual",
                    "observations": 100,
                    "mae": 0.00024,
                    "rmse": 0.00050,
                },
            ],
            "financial_consistency": [
                {
                    "model": "Price-only constrained residual",
                    "negative_price_count": 0,
                    "negative_price_rate": 0.0,
                    "below_financial_floor_count": 0,
                    "below_financial_floor_rate": 0.0,
                    "observations": 100,
                },
                {
                    "model": "Multi-task constrained residual",
                    "negative_price_count": 0,
                    "negative_price_rate": 0.0,
                    "below_financial_floor_count": 0,
                    "below_financial_floor_rate": 0.0,
                    "observations": 100,
                },
            ],
            "classification": [
                {
                    "model": "Exercise-only classifier",
                    "f1": 0.9964,
                    "accuracy": 0.998,
                },
                {
                    "model": "Multi-task model",
                    "f1": 0.9963,
                    "accuracy": 0.998,
                },
            ],
            "boundary_location": [
                {
                    "model": "Exercise-only classifier",
                    "boundary_mae": 0.0011,
                },
                {
                    "model": "Multi-task model",
                    "boundary_mae": 0.0010,
                },
            ],
            "ood": [
                {
                    "model": "Multi-task price",
                    "ood_set": "extreme_moneyness",
                    "observations": 50,
                    "mae": 0.001,
                    "rmse": 0.002,
                }
            ],
            "hypothesis": {
                "evidence": {
                    "classifier_boundary_f1": 0.9964,
                    "multitask_boundary_f1": 0.9963,
                    "price_only_boundary_mae": 0.00008,
                    "multitask_boundary_mae": 0.00015,
                    "required_f1_improvement": 0.01,
                    "required_mae_improvement": 0.01,
                }
            },
        },
        "integrated": {
            "pricing_metrics": [
                {
                    "head": "Constrained residual",
                    "observations": 100,
                    "mae": 0.00056,
                    "rmse": 0.0011,
                },
                {
                    "head": "Direct price",
                    "observations": 100,
                    "mae": 0.0056,
                    "rmse": 0.0073,
                },
            ],
            "exercise_metrics": {
                "f1": 0.9868,
                "accuracy": 0.993,
            },
            "consistency_metrics": {
                "constrained_negative_rate": 0.0,
                "constrained_below_european_rate": 0.0,
                "constrained_below_intrinsic_rate": 0.0,
                "direct_negative_rate": 0.0,
                "direct_below_european_rate": 0.16,
                "direct_below_intrinsic_rate": 0.27,
                "decision_disagreement_rate": 0.0068,
                "any_contradiction_rate": 0.38,
                "residual_reconstruction_mae": 0.0,
            },
            "boundary_analysis": [
                {
                    "boundary_band": "≤0.001",
                    "observations": 100,
                    "price_mae": 0.00003,
                    "exercise_accuracy": 0.98,
                    "exercise_f1": 0.9868,
                    "balanced_accuracy": 0.959,
                }
            ],
            "ood_metrics": [
                {
                    "component": "american_put_ood_extreme_moneyness",
                    "observations": 50,
                    "constrained_mae": 0.0005,
                    "constrained_rmse": 0.0014,
                }
            ],
            "runtime": {
                "observations": 100_000,
                "seconds": 0.003,
                "device": "cuda",
            },
        },
        "lsm": {
            "runtime": [
                {
                    "method": "CRR",
                    "count": 100,
                    "median": 0.09,
                },
                {
                    "method": "Neural LSM evaluation",
                    "count": 100,
                    "median": 0.075,
                },
            ],
            "training": {"runtime_seconds": 628.0},
        },
    }


def test_canonical_final_evaluation_normalization() -> None:
    results = _synthetic_results()

    static = build_static_pricing_table(results)
    direct = static.loc[static["model"].eq("Direct MLP")]
    assert len(direct) == 1
    assert direct["source_notebook"].iloc[0] == "04"

    consistency = build_financial_consistency_table(results)
    direct_rate = consistency.loc[
        consistency["model"].eq("Direct MLP"),
        "below_intrinsic_violation_rate",
    ].iloc[0]
    constrained_rate = consistency.loc[
        consistency["model"].eq("Constrained floor residual"),
        "below_intrinsic_violation_rate",
    ].iloc[0]
    assert direct_rate == 0.18
    assert constrained_rate == 0.0

    boundary = build_boundary_comparison(results)
    assert "Multi-task constrained residual" in set(boundary["model"])
    assert "Final integrated constrained price" in set(boundary["model"])

    ood = build_ood_comparison(results, static)
    assert not ood.duplicated(["model", "regime"]).any()
    assert not ood["regime"].str.startswith("american_put_ood_").any()
    assert len(ood.loc[ood["model"].eq("Direct MLP")]) == 2

    runtime = build_runtime_comparison_from_results(results)
    duplicate_direct_runtime = runtime.loc[
        runtime["model"].eq("Direct MLP")
        & runtime["observations"].eq(100_000)
    ]
    assert len(duplicate_direct_runtime) == 1
    assert duplicate_direct_runtime["source_notebook"].iloc[0] == "04"

    ablation = build_static_ablation_table(
        static,
        consistency,
        boundary,
    )
    for model in (
        "Multi-task constrained residual",
        "Final integrated constrained price",
    ):
        value = ablation.loc[
            ablation["model"].eq(model),
            "exercise_f1",
        ].iloc[0]
        assert np.isfinite(value)

    evidence = build_hypothesis_evidence(
        results,
        static,
        consistency,
        ood,
        runtime,
    )
    assert evidence["direct_violation_rate"] == 0.18
    assert evidence["constrained_violation_rate"] == 0.0

    decisions = decide_all_hypotheses(evidence)
    h3 = decisions.loc[decisions["hypothesis"].eq("H3")].iloc[0]
    assert h3["decision"] == "Supported"

    semantic = validate_final_evaluation_semantics(
        static,
        consistency,
        ood,
        runtime,
        ablation,
    )
    assert semantic["valid"].all()
