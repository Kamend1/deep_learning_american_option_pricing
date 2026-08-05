from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_final_project(root: Path, *, target_shift: float = 0.0) -> Path:
    root = Path(root)
    _write_json(
        root / "data/manifests/production_dataset_manifest.json",
        {"components": [{"name": "core", "observations": 3}]},
    )

    sample_id = [10, 11, 12]
    target = [0.10, 0.20, 0.30]
    state = {
        "sample_id": sample_id,
        "moneyness": [0.9, 1.0, 1.1],
        "log_moneyness": [-0.1, 0.0, 0.1],
        "time_to_maturity": [0.5, 1.0, 1.5],
        "risk_free_rate": [0.02, 0.03, 0.04],
        "dividend_yield": [0.00, 0.01, 0.02],
        "volatility": [0.20, 0.25, 0.30],
    }

    # Notebook 04
    nb04 = root / "artifacts/direct_mlp"
    _write_json(
        nb04 / "final_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "notebook": "04_direct_mlp_pricer",
            "training_profile": "full",
            "selected_model": "Direct MLP",
            "checkpoint": "best_direct_mlp.pt",
            "pricing": {
                "black_scholes_proxy": {"mae": 0.02},
                "direct_mlp": {"mae": 0.01},
            },
            "financial_consistency": [],
            "ood": [],
            "runtime": [],
        },
    )
    _write_json(
        nb04 / "training_complete.json",
        {
            "status": "complete",
            "notebook": "04_direct_mlp_pricer",
            "training_profile": "full",
            "checkpoint": "best_direct_mlp.pt",
        },
    )
    _touch(nb04 / "best_direct_mlp.pt")
    _write_csv(
        nb04 / "test_predictions.csv",
        pd.DataFrame(
            {
                **state,
                "normalized_american_price": target,
                "direct_mlp_prediction": [0.11, 0.19, 0.31],
            }
        ),
    )

    # Notebook 05
    nb05 = root / "artifacts/premium_models"
    _write_json(
        nb05 / "final_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "notebook": "05_early_exercise_premium_model",
            "training_profile": "full",
            "selected_model": "Constrained floor residual",
            "selected_candidate": "balanced",
            "checkpoint": "best_premium_model.pt",
            "pricing": [],
            "premium_error": [],
            "financial_consistency": [],
            "segmented_results": [],
            "ood": [],
            "runtime": [],
            "hypotheses": {},
        },
    )
    _write_json(
        nb05 / "training_complete.json",
        {
            "status": "complete",
            "notebook": "05_early_exercise_premium_model",
            "training_profile": "full",
            "dependencies": {},
            "candidates": {"balanced": {}},
        },
    )
    _touch(nb05 / "best_premium_model.pt")
    _write_csv(
        nb05 / "test_predictions.csv",
        pd.DataFrame(
            {
                **state,
                "normalized_american_price": target,
                "constrained_floor_prediction": [0.10, 0.20, 0.30],
            }
        ),
    )

    # Notebook 06
    nb06 = root / "artifacts/multitask_model"
    _write_json(
        nb06 / "final_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "notebook": "06_exercise_boundary_analysis",
            "training_profile": "full",
            "selected_candidate": "balanced",
            "checkpoint": "best_multitask_pricer.pt",
            "dependencies": {},
            "thresholds": {"exercise_classifier": 0.5, "multitask": 0.5},
            "classification": [],
            "pricing": [],
            "boundary_bands": [],
            "boundary_pricing": [],
            "boundary_location": [],
            "financial_consistency": [],
            "ood_classification": [],
            "ood_pricing": [],
            "inference": [],
            "hypothesis": {},
        },
    )
    for name, notebook in (
        ("multitask_training_complete.json", "06_multitask_model"),
        ("exercise_classifier_complete.json", "06_exercise_classifier"),
    ):
        _write_json(
            nb06 / name,
            {
                "status": "complete",
                "notebook": notebook,
                "training_profile": "full",
                "dependencies": {},
                "candidates": {} if "multitask" in name else None,
                "checkpoint": (
                    "best_exercise_classifier.pt"
                    if "classifier" in name
                    else "best_multitask_pricer.pt"
                ),
            },
        )
    _touch(nb06 / "best_multitask_pricer.pt")
    _touch(nb06 / "best_exercise_classifier.pt")
    _write_csv(
        nb06 / "test_predictions.csv",
        pd.DataFrame(
            {
                **state,
                "normalized_american_price": target,
                "exercise_now": [0, 1, 1],
                "classifier_probability": [0.1, 0.8, 0.9],
                "multitask_probability": [0.2, 0.7, 0.8],
                "predicted_normalized_american_price": [0.10, 0.20, 0.30],
                "price_only_normalized_price": [0.10, 0.20, 0.30],
            }
        ),
    )

    # Notebook 07
    nb07 = root / "artifacts/neural_lsm"
    _write_json(
        nb07 / "final_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "notebook": "07_neural_longstaff_schwartz",
            "training_profile": "final",
            "selected_model": "Classical LSM",
            "neural_policy_checkpoint": "neural_lsm_policy.pt",
            "selected_classical_basis": "laguerre",
            "selected_classical_degree": 2,
            "training": {"dependencies": {}},
            "heldout_pricing": [],
            "paired_mae_bootstrap": [],
            "coverage": {},
            "financial_bounds": [],
            "policy_summary": [],
            "ood_pricing": [],
            "ood_policy": [],
            "runtime": [],
            "runtime_context": [],
            "runtime_break_even": {},
            "h5_decision": {},
        },
    )
    _write_json(
        nb07 / "training_complete.json",
        {
            "status": "complete",
            "notebook": "07_neural_longstaff_schwartz",
            "training_profile": "final",
            "dependencies": {},
            "checkpoint": "neural_lsm_policy.pt",
        },
    )
    _touch(nb07 / "neural_lsm_policy.pt")
    _write_csv(
        nb07 / "heldout_pricing_results.csv",
        pd.DataFrame(
            {
                "contract_id": [1, 2],
                "crr_price": [1.0, 2.0],
                "classical_lsm_price": [1.1, 2.1],
                "neural_lsm_price": [1.2, 2.2],
            }
        ),
    )

    # Notebook 08 — schema v2 with deployment and scratch roles
    nb08 = root / "artifacts/final_multihead"
    deployment_selection = {
        "schema_version": 1,
        "selection_scope": "in_domain_combined_deployment",
        "preferred_integrated_candidate": "warm_start",
        "selected_scratch_candidate": "balanced",
        "selection_rule": "validation and operational evidence only",
        "checks": {"validation_price": True, "runtime": True},
        "all_checks_passed": True,
        "test_metrics_used_for_selection": False,
        "ood_metrics_used_for_selection": False,
    }
    selection = {
        "schema_version": 2,
        "configuration": "balanced",
        "selected_scratch_configuration": "balanced",
        "selected_scratch_checkpoint": "best_integrated_scratch.pt",
        "preferred_integrated_candidate": "warm_start",
        "preferred_integrated_checkpoint": "best_integrated_deployment.pt",
        "canonical_checkpoint": "best_integrated_multihead.pt",
        "authoritative_price_output": "constrained_price",
        "selection_basis": "validation_and_operational_evidence_only",
        "test_metrics_used_for_selection": False,
        "ood_metrics_used_for_selection": False,
        "deployment_scope": "in_domain_combined_price_and_exercise",
        "fallback_method": "high_resolution_crr",
        "scratch_selection": {
            "exercise_threshold": 0.7,
            "continuation_threshold": 0.75,
        },
        "deployment_selection": deployment_selection,
        "dependencies": {},
    }
    deployment_policy = {
        "schema_version": 1,
        "preferred_integrated_candidate": "warm_start",
        "neural_scope": "contracts inside the validated in-domain ranges",
        "fallback_method": "high_resolution_crr",
        "fallback_trigger": "one or more inputs outside domain_bounds.json",
        "price_only_preference": "Notebook 05 constrained residual model",
        "exercise_only_preference": "Notebook 06 specialist classifier",
    }
    domain_bounds = {
        "moneyness": [0.45, 1.5],
        "time_to_maturity": [7.0 / 365.0, 2.0],
        "volatility": [0.05, 0.8],
        "risk_free_rate": [0.0, 0.15],
        "dividend_yield": [0.0, 0.08],
    }
    warm_boundary = [
        {
            "decision_path": "Exercise head", "boundary_band": "≤0.001",
            "boundary_limit": 0.001, "observations": 2, "threshold": 0.7,
            "accuracy": 1.0, "balanced_accuracy": 1.0, "f1": 1.0,
            "price_mae": 0.01, "decision_errors": 0, "total_regret": 0.0,
        },
        {
            "decision_path": "Continuation-implied", "boundary_band": "≤0.001",
            "boundary_limit": 0.001, "observations": 2, "threshold": 0.75,
            "accuracy": 0.5, "balanced_accuracy": 0.5, "f1": 0.5,
            "price_mae": 0.01, "decision_errors": 1, "total_regret": 0.1,
        },
    ]
    scratch_boundary = [dict(item) for item in warm_boundary]
    _write_json(nb08 / "selection.json", selection)
    _write_json(nb08 / "deployment_selection.json", deployment_selection)
    _write_json(nb08 / "deployment_policy.json", deployment_policy)
    _write_json(nb08 / "domain_bounds.json", domain_bounds)
    _write_json(
        nb08 / "final_metrics.json",
        {
            "schema_version": 2,
            "status": "complete",
            "notebook": "08_final_multihead_model",
            "training_profile": "full",
            "selected_scratch_configuration": "balanced",
            "preferred_integrated_candidate": "warm_start",
            "checkpoint": "best_integrated_multihead.pt",
            "deployment_checkpoint": "best_integrated_deployment.pt",
            "scratch_checkpoint": "best_integrated_scratch.pt",
            "authoritative_price_output": "constrained_price",
            "deployment_scope": deployment_policy["neural_scope"],
            "fallback_method": "high_resolution_crr",
            "thresholds": {"exercise_head": 0.7, "continuation_path": 0.75},
            "selection": selection,
            "pricing": [], "continuation": {},
            "classification": {"threshold": 0.7}, "consistency": {},
            "boundary": warm_boundary, "ood": [],
            "runtime": [
                {"model": "Notebook 08 warm-start", "observations": 1000,
                 "median_seconds": 0.01, "seconds_per_observation": 0.00001,
                 "observations_per_second": 100000.0, "device": "cpu"},
                {"model": "Notebook 08 selected scratch", "observations": 1000,
                 "median_seconds": 0.02, "seconds_per_observation": 0.00002,
                 "observations_per_second": 50000.0, "device": "cpu"},
            ],
            "scratch_benchmark": {
                "configuration": "balanced", "pricing": [], "continuation": {},
                "classification": {"threshold": 0.7}, "consistency": {},
                "boundary": scratch_boundary, "ood": [], "segmented_results": [],
            },
            "integrated_candidate_comparison": [],
            "dependencies": {},
        },
    )
    for name, notebook in (
        ("scratch_training_complete.json", "08_final_multihead_scratch"),
        ("warm_start_training_complete.json", "08_final_multihead_warm_start"),
    ):
        _write_json(
            nb08 / name,
            {"status": "complete", "notebook": notebook,
             "training_profile": "full", "dependencies": {},
             "checkpoint": "best_warm_start.pt" if "warm" in name else "best_balanced.pt"},
        )
    for checkpoint in (
        "best_integrated_multihead.pt", "best_integrated_deployment.pt",
        "best_integrated_scratch.pt",
    ):
        _touch(nb08 / checkpoint)
    warm_predictions = pd.DataFrame(
        {
            **state,
            "true_normalized_american_price": [value + target_shift for value in target],
            "predicted_normalized_american_price": [0.10, 0.20, 0.30],
            "predicted_direct_normalized_american_price": [0.11, 0.19, 0.31],
            "exercise_target": [0, 1, 1],
            "exercise_probability": [0.1, 0.8, 0.9],
            "continuation_exercise_probability": [0.2, 0.7, 0.8],
            "true_normalized_continuation_value": [0.12, 0.18, 0.25],
            "predicted_normalized_continuation_value": [0.11, 0.19, 0.24],
        }
    )
    _write_csv(nb08 / "test_predictions.csv", warm_predictions)
    scratch_predictions = warm_predictions.copy()
    scratch_predictions["predicted_normalized_american_price"] = [0.101, 0.201, 0.301]
    scratch_predictions["exercise_probability"] = [0.12, 0.79, 0.88]
    _write_csv(nb08 / "scratch_test_predictions.csv", scratch_predictions)
    _write_csv(nb08 / "boundary_analysis.csv", pd.DataFrame(warm_boundary))
    _write_json(
        nb08 / "runtime_summary.json",
        {"observations": 1000, "median_seconds": 0.01,
         "seconds_per_observation": 0.00001,
         "observations_per_second": 100000.0, "device": "cpu"},
    )

    return root


@pytest.fixture
def final_project(tmp_path: Path) -> Path:
    return build_final_project(tmp_path)
