"""Miniature end-to-end smoke test for final project aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.artifact_registry import ArtifactSpec, audit_artifacts
from src.evaluation.final_project_evaluation import (
    align_prediction_frames,
    build_consolidated_pricing_table,
)
from src.evaluation.final_reporting import export_final_evaluation
from src.evaluation.hypothesis_testing import decide_all_hypotheses


@pytest.mark.integration
def test_full_project_evaluation_pipeline(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    direct = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "truth": [0.10, 0.20, 0.30, 0.40],
            "prediction": [0.11, 0.19, 0.29, 0.39],
        }
    )
    residual = direct.copy()
    residual["prediction"] = [0.10, 0.20, 0.295, 0.405]

    aligned = align_prediction_frames(
        {"direct": direct, "residual": residual},
        id_column="sample_id",
    )
    assert len(aligned["direct"]) == 4

    metrics = {
        "direct": {"mae": 0.01, "rmse": 0.01, "r2": 0.99},
        "residual": {"mae": 0.0025, "rmse": 0.0035, "r2": 0.999},
    }
    table = build_consolidated_pricing_table(metrics)
    assert table.index.tolist() == ["direct", "residual"]

    evidence = {
        "black_scholes_mae": 0.05,
        "direct_mlp_mae": 0.01,
        "best_residual_mae": 0.0025,
        "direct_violation_rate": 0.01,
        "constrained_violation_rate": 0.0,
        "price_only_boundary_f1": 0.80,
        "multitask_boundary_f1": 0.84,
        "price_only_boundary_error": 0.02,
        "multitask_boundary_error": 0.01,
        "crr_seconds_per_option": 0.005,
        "neural_seconds_per_option": 0.00002,
        "in_domain_mae": 0.0025,
        "aggregate_ood_mae": 0.004,
    }
    decisions = decide_all_hypotheses(evidence)

    metric_path = artifacts / "metrics.json"
    metric_path.write_text(json.dumps(metrics), encoding="utf-8")
    registry = (
        ArtifactSpec("metrics", "evaluation", ("artifacts/metrics.json",), True, "json"),
    )
    audit = audit_artifacts(tmp_path, registry)
    assert audit["valid"].all()

    outputs = export_final_evaluation(
        tmp_path / "final",
        tables={"consolidated_model_metrics": table},
        hypothesis_decisions=decisions,
        artifact_audit=audit,
        summary={"status": "SMOKE_COMPLETE"},
    )
    assert Path(outputs["consolidated_model_metrics"]).exists()
    assert Path(outputs["final_results_summary"]).exists()
