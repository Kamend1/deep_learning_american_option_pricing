"""Tests for Step 9 final-evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.artifact_registry import (
    ArtifactSpec,
    audit_artifacts,
)
from src.evaluation.final_project_evaluation import (
    align_prediction_frames,
    build_consolidated_pricing_table,
    build_runtime_comparison,
    calculate_ood_deterioration,
)
from src.evaluation.final_reporting import export_final_evaluation
from src.evaluation.hypothesis_testing import decide_all_hypotheses


def test_artifact_audit_reports_found_and_pending(tmp_path: Path) -> None:
    existing = tmp_path / "metrics.json"
    existing.write_text('{"mae": 0.01}', encoding="utf-8")
    registry = (
        ArtifactSpec("found", "test", ("metrics.json",), True, "json"),
        ArtifactSpec("missing", "test", ("missing.json",), False, "json"),
    )
    audit = audit_artifacts(tmp_path, registry)
    assert audit.set_index("name").loc["found", "valid"]
    assert not audit.set_index("name").loc["missing", "found"]


def test_align_prediction_frames_rejects_misalignment() -> None:
    left = pd.DataFrame({"sample_id": [2, 1], "prediction": [0.2, 0.1]})
    right = pd.DataFrame({"sample_id": [1, 2], "prediction": [0.1, 0.2]})
    aligned = align_prediction_frames({"left": left, "right": right}, id_column="sample_id")
    assert aligned["left"]["sample_id"].tolist() == [1, 2]

    bad = pd.DataFrame({"sample_id": [1, 3], "prediction": [0.1, 0.3]})
    with pytest.raises(ValueError):
        align_prediction_frames({"left": left, "bad": bad}, id_column="sample_id")


def test_consolidated_pricing_table() -> None:
    table = build_consolidated_pricing_table(
        {
            "direct": {"mae": 0.02, "rmse": 0.03, "r2": 0.98},
            "residual": {"mae": 0.01, "rmse": 0.02, "r2": 0.99},
        }
    )
    assert table.loc["residual", "mae"] < table.loc["direct", "mae"]


def test_ood_and_runtime_tables() -> None:
    in_domain = pd.DataFrame({"model": ["a"], "mae": [0.01]})
    ood = pd.DataFrame({"model": ["a"], "regime": ["high_vol"], "mae": [0.015]})
    deterioration = calculate_ood_deterioration(in_domain, ood)
    assert deterioration.loc[0, "ood_deterioration"] == pytest.approx(0.5)

    runtime = build_runtime_comparison(
        pd.DataFrame({"model": ["a"], "observations": [1000], "seconds": [2.0]})
    )
    assert runtime.loc[0, "observations_per_second"] == pytest.approx(500.0)


def test_hypothesis_rules_and_export(tmp_path: Path) -> None:
    evidence = {
        "black_scholes_mae": 0.10,
        "direct_mlp_mae": 0.05,
        "best_residual_mae": 0.04,
        "direct_violation_rate": 0.02,
        "constrained_violation_rate": 0.0,
        "price_only_boundary_f1": 0.80,
        "multitask_boundary_f1": 0.85,
        "price_only_boundary_error": 0.03,
        "multitask_boundary_error": 0.02,
        "crr_seconds_per_option": 0.01,
        "neural_seconds_per_option": 0.0001,
        "in_domain_mae": 0.01,
        "aggregate_ood_mae": 0.02,
    }
    decisions = decide_all_hypotheses(evidence)
    assert set(decisions["decision"]) == {"Supported"}

    audit = pd.DataFrame(
        {
            "name": ["artifact"],
            "category": ["test"],
            "required_for_final": [True],
            "found": [True],
            "valid": [True],
            "resolved_path": ["x"],
            "loader": ["json"],
            "notes": ["ok"],
        }
    )
    paths = export_final_evaluation(
        tmp_path,
        tables={"pricing": pd.DataFrame({"mae": [0.01]}, index=["model"])},
        hypothesis_decisions=decisions,
        artifact_audit=audit,
        summary={"status": "READY"},
    )
    assert Path(paths["final_writeup_inputs"]).exists()
    payload = json.loads(Path(paths["hypothesis_decisions"]).read_text(encoding="utf-8"))
    assert len(payload) == 6
