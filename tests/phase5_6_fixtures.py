from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.final_artifact_adapters import FinalNotebookPackage


def _package(
    root: Path,
    notebook: str,
    *,
    metrics: dict[str, Any],
    predictions: pd.DataFrame | None = None,
    benchmark_predictions: pd.DataFrame | None = None,
) -> FinalNotebookPackage:
    root = Path(root)
    artifact = root / f"nb{notebook}"
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact / f"nb{notebook}.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics_path = artifact / "final_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    predictions_path = None
    if predictions is not None:
        predictions_path = artifact / "test_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
    benchmark_path = None
    if benchmark_predictions is not None:
        benchmark_path = artifact / "scratch_test_predictions.csv"
        benchmark_predictions.to_csv(benchmark_path, index=False)
    return FinalNotebookPackage(
        notebook_number=notebook,
        notebook_id=f"nb{notebook}",
        training_profile="final" if notebook == "07" else "full",
        selected_model=str(metrics.get("selected_model", f"model-{notebook}")),
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        final_metrics_path=metrics_path,
        final_metrics=metrics,
        training_manifests=(),
        test_predictions_path=predictions_path,
        benchmark_test_predictions_path=benchmark_path,
    )


def build_phase5_6_packages(root: Path) -> dict[str, FinalNotebookPackage]:
    sample_id = [1, 2, 3, 4, 5, 6]
    target = [0, 0, 1, 1, 1, 0]
    nb06_predictions = pd.DataFrame(
        {
            "sample_id": sample_id,
            "exercise_now": target,
            "classifier_probability": [0.05, 0.20, 0.85, 0.90, 0.75, 0.10],
            "multitask_probability": [0.10, 0.30, 0.80, 0.85, 0.70, 0.15],
            "boundary_distance_normalized": [0.0005, 0.002, 0.0008, 0.004, 0.008, 0.020],
            "signed_boundary_margin": [-0.0005, -0.002, 0.0008, 0.004, 0.008, -0.020],
        }
    )
    nb08_predictions = pd.DataFrame(
        {
            "sample_id": sample_id,
            "exercise_target": target,
            "exercise_probability": [0.10, 0.25, 0.82, 0.88, 0.72, 0.12],
            "continuation_exercise_probability": [0.15, 0.35, 0.78, 0.80, 0.65, 0.20],
        }
    )
    regimes = [
        "high_volatility",
        "extreme_moneyness",
        "long_maturity",
        "rate_dividend",
    ]

    nb04_ood = [
        {"ood_set": regime, "observations": 50, "mae": 0.04, "rmse": 0.05}
        for regime in regimes
    ]
    nb05_ood = []
    for regime in regimes:
        nb05_ood.extend(
            [
                {
                    "ood_set": regime,
                    "model": "Non-negative premium",
                    "observations": 50,
                    "mae": 0.02,
                    "rmse": 0.03,
                },
                {
                    "ood_set": regime,
                    "model": "Constrained floor residual",
                    "observations": 50,
                    "mae": 0.015,
                    "rmse": 0.02,
                },
            ]
        )
    nb06_ood_pricing = []
    nb06_ood_classification = []
    for regime in regimes:
        nb06_ood_pricing.extend(
            [
                {
                    "ood_set": regime,
                    "model": "Price-only constrained residual",
                    "observations": 50,
                    "mae": 0.018,
                    "rmse": 0.025,
                },
                {
                    "ood_set": regime,
                    "model": "Multi-task constrained residual",
                    "observations": 50,
                    "mae": 0.025,
                    "rmse": 0.035,
                },
            ]
        )
        nb06_ood_classification.extend(
            [
                {
                    "ood_set": regime,
                    "model": "Exercise-only classifier",
                    "observations": 50,
                    "positive_rate": 0.3,
                    "threshold": 0.5,
                    "accuracy": 0.90,
                    "balanced_accuracy": 0.88,
                    "precision": 0.85,
                    "recall": 0.80,
                    "f1": 0.825,
                    "brier_score": 0.08,
                    "roc_auc": 0.95,
                    "pr_auc": 0.90,
                },
                {
                    "ood_set": regime,
                    "model": "Multi-task model",
                    "observations": 50,
                    "positive_rate": 0.3,
                    "threshold": 0.5,
                    "accuracy": 0.88,
                    "balanced_accuracy": 0.86,
                    "precision": 0.82,
                    "recall": 0.78,
                    "f1": 0.80,
                    "brier_score": 0.09,
                    "roc_auc": 0.94,
                    "pr_auc": 0.89,
                },
            ]
        )

    nb08_ood = []
    for regime in regimes:
        nb08_ood.append(
            {
                "component": f"american_put_ood_{regime}",
                "observations": 50,
                "constrained_mae": 0.03,
                "constrained_rmse": 0.04,
                "direct_mae": 0.05,
                "direct_rmse": 0.06,
                "exercise_head_accuracy": 0.89,
                "exercise_head_balanced_accuracy": 0.87,
                "exercise_head_precision": 0.84,
                "exercise_head_recall": 0.81,
                "exercise_head_f1": 0.825,
                "exercise_head_brier_score": 0.08,
                "exercise_head_roc_auc": 0.95,
                "exercise_head_pr_auc": 0.90,
                "continuation_path_accuracy": 0.87,
                "continuation_path_balanced_accuracy": 0.85,
                "continuation_path_precision": 0.80,
                "continuation_path_recall": 0.79,
                "continuation_path_f1": 0.795,
                "continuation_path_brier_score": 0.10,
                "continuation_path_roc_auc": 0.93,
                "continuation_path_pr_auc": 0.87,
            }
        )

    packages = {
        "04": _package(
            root,
            "04",
            metrics={
                "selected_model": "Direct MLP",
                "ood": nb04_ood,
                "runtime": [
                    {
                        "model": "Direct MLP",
                        "observations": 100000,
                        "median_seconds": 0.20,
                        "observations_per_second": 500000.0,
                        "device": "cpu",
                    }
                ],
            },
        ),
        "05": _package(
            root,
            "05",
            metrics={
                "selected_model": "Constrained floor residual",
                "ood": nb05_ood,
                "runtime": [
                    {
                        "model": "Direct MLP",
                        "observations": 100000,
                        "median_seconds": 0.21,
                        "observations_per_second": 476190.0,
                        "device": "cpu",
                    },
                    {
                        "model": "Constrained floor residual",
                        "observations": 100000,
                        "median_seconds": 0.25,
                        "observations_per_second": 400000.0,
                        "device": "cpu",
                    }
                ],
            },
        ),
        "06": _package(
            root,
            "06",
            metrics={
                "thresholds": {"exercise_classifier": 0.5, "multitask": 0.5},
                "ood_classification": nb06_ood_classification,
                "ood_pricing": nb06_ood_pricing,
                "inference": [
                    {
                        "model": "Constrained floor residual",
                        "observations": 100000,
                        "median_seconds": 0.26,
                    },
                    {
                        "model": "Exercise-only classifier",
                        "observations": 100000,
                        "median_seconds": 0.20,
                    },
                    {
                        "model": "Price-only constrained residual",
                        "observations": 100000,
                        "median_seconds": 0.22,
                    },
                    {
                        "model": "Multi-task model",
                        "observations": 100000,
                        "median_seconds": 0.24,
                    },
                ],
                "hypothesis": {
                    "evidence": {
                        "classifier_boundary_f1": 0.96,
                        "multitask_boundary_f1": 0.94,
                        "maximum_allowed_f1_degradation": 0.001,
                        "price_only_boundary_mae": 0.001,
                        "multitask_boundary_mae": 0.0015,
                        "required_boundary_mae_improvement": 0.01,
                    }
                },
            },
            predictions=nb06_predictions,
        ),
        "07": _package(
            root,
            "07",
            metrics={
                "selected_model": "Classical LSM",
                "training": {"runtime_seconds": 600.0},
                "heldout_pricing": [
                    {"method": "classical_lsm_price", "mae": 0.04, "rmse": 0.05},
                    {"method": "neural_lsm_price", "mae": 0.09, "rmse": 0.12},
                ],
                "ood_pricing": [
                    {"ood_set": regime, "method": "classical_lsm_price", "mae": 0.06}
                    for regime in regimes
                ]
                + [
                    {"ood_set": regime, "method": "neural_lsm_price", "mae": 0.30}
                    for regime in regimes
                ],
                "financial_bounds": [
                    {"method": "classical_lsm_price", "check": "below_european", "violation_rate": 0.2},
                    {"method": "neural_lsm_price", "check": "below_european", "violation_rate": 0.3},
                ],
                "policy_summary": [{"metric": "exercise_rate", "mean": 0.4}],
                "coverage": {
                    "Classical LSM 95% CI coverage": 0.68,
                    "Neural LSM 95% CI coverage": 0.42,
                },
                "runtime": [
                    {"method": "CRR", "count": 100, "median": 0.50},
                    {"method": "Classical LSM end-to-end", "count": 100, "median": 0.35},
                    {"method": "Neural LSM evaluation", "count": 100, "median": 0.60},
                ],
                "runtime_break_even": {"estimated_break_even_contracts": float("inf")},
            },
        ),
        "08": _package(
            root,
            "08",
            metrics={
                "thresholds": {"exercise_head": 0.5, "continuation_path": 0.5},
                "selection": {
                    "scratch_selection": {
                        "exercise_threshold": 0.55,
                        "continuation_threshold": 0.60,
                    }
                },
                "ood": nb08_ood,
                "scratch_benchmark": {
                    "classification": {"threshold": 0.55},
                    "boundary": [
                        {"decision_path": "Exercise head", "threshold": 0.55},
                        {"decision_path": "Continuation-implied", "threshold": 0.60},
                    ],
                    "ood": [dict(record) for record in nb08_ood],
                },
                "runtime": [
                    {
                        "model": "Notebook 08 warm-start",
                        "observations": 100000,
                        "median_seconds": 0.30,
                        "seconds_per_observation": 0.000003,
                        "observations_per_second": 333333.33,
                        "device": "cpu",
                    },
                    {
                        "model": "Notebook 08 selected scratch",
                        "observations": 100000,
                        "median_seconds": 0.45,
                        "seconds_per_observation": 0.0000045,
                        "observations_per_second": 222222.22,
                        "device": "cpu",
                    },
                ],
            },
            predictions=nb08_predictions,
            benchmark_predictions=nb08_predictions.assign(
                exercise_probability=[0.12, 0.27, 0.80, 0.86, 0.70, 0.14],
                continuation_exercise_probability=[0.18, 0.37, 0.76, 0.79, 0.63, 0.22],
            ),
        ),
    }
    return packages


def phase4_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(
        [
            {"model_id": "black_scholes_proxy", "normalized_mae": 0.10},
            {"model_id": "direct_mlp", "normalized_mae": 0.02},
            {"model_id": "nonnegative_premium_mlp", "normalized_mae": 0.012},
            {"model_id": "constrained_floor_residual_mlp", "normalized_mae": 0.010},
            {"model_id": "price_only_constrained_residual_mlp", "normalized_mae": 0.011},
            {"model_id": "multitask_constrained_residual_mlp", "normalized_mae": 0.015},
            {"model_id": "integrated_warm_start_constrained_price", "normalized_mae": 0.020},
            {"model_id": "integrated_warm_start_direct_price_head", "normalized_mae": 0.030},
            {"model_id": "integrated_scratch_constrained_price", "normalized_mae": 0.025},
            {"model_id": "integrated_scratch_direct_price_head", "normalized_mae": 0.035},
        ]
    )
    consistency = pd.DataFrame(
        [
            {
                "model_id": "direct_mlp",
                "negative_rate": 0.0,
                "below_european_rate": 0.10,
                "below_intrinsic_rate": 0.08,
                "below_financial_floor_rate": 0.15,
            },
            {
                "model_id": "constrained_floor_residual_mlp",
                "negative_rate": 0.0,
                "below_european_rate": 0.0,
                "below_intrinsic_rate": 0.0,
                "below_financial_floor_rate": 0.0,
            },
        ]
    )
    return metrics, consistency
