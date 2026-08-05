from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.final_artifact_adapters import FinalNotebookPackage
from src.evaluation.final_static_comparison import (
    MODEL_SPECS,
    assert_phase_4_ready,
    build_static_prediction_matrix,
    run_phase_4_static_comparison,
)


def _package(
    tmp_path: Path,
    notebook: str,
    frame: pd.DataFrame,
    *,
    benchmark_frame: pd.DataFrame | None = None,
) -> FinalNotebookPackage:
    prediction_path = tmp_path / f"nb{notebook}_predictions.csv"
    frame.to_csv(prediction_path, index=False)
    checkpoint_path = tmp_path / f"nb{notebook}.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    metrics_path = tmp_path / f"nb{notebook}_metrics.json"
    metrics_path.write_text('{"status": "complete"}', encoding="utf-8")
    benchmark_path = None
    if benchmark_frame is not None:
        benchmark_path = tmp_path / f"nb{notebook}_benchmark_predictions.csv"
        benchmark_frame.to_csv(benchmark_path, index=False)
    return FinalNotebookPackage(
        notebook_number=notebook,
        notebook_id=f"nb{notebook}",
        training_profile="full",
        selected_model=f"model-{notebook}",
        checkpoint_name=checkpoint_path.name,
        checkpoint_path=checkpoint_path,
        final_metrics_path=metrics_path,
        final_metrics={"status": "complete"},
        training_manifests=(),
        test_predictions_path=prediction_path,
        benchmark_test_predictions_path=benchmark_path,
    )


@pytest.fixture()
def static_project(tmp_path: Path):
    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "production_dataset_manifest.json").write_text(
        json.dumps({"generation_config": {"strike": 100.0}}),
        encoding="utf-8",
    )

    sample_id = np.arange(6)
    true = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    european = np.array([0.08, 0.18, 0.25, 0.35, 0.45, 0.55])
    intrinsic = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    base = pd.DataFrame(
        {
            "sample_id": sample_id,
            "split": "test",
            "moneyness": [0.75, 0.90, 1.00, 1.10, 1.30, 0.82],
            "log_moneyness": np.log([0.75, 0.90, 1.00, 1.10, 1.30, 0.82]),
            "time_to_maturity": [0.2, 0.4, 0.7, 1.2, 1.8, 0.9],
            "risk_free_rate": 0.03,
            "dividend_yield": 0.01,
            "volatility": [0.15, 0.25, 0.35, 0.45, 0.60, 0.18],
            "exercise_now": [True, True, False, False, False, True],
            "normalized_european_price": european,
            "normalized_intrinsic_value": intrinsic,
            "normalized_american_price": true,
        }
    )

    nb04 = base.drop(columns=["normalized_intrinsic_value"]).copy()
    nb04["direct_mlp_prediction"] = true + 0.010

    nb05 = base.copy()
    nb05["boundary_distance_normalized"] = [0.0005, 0.002, 0.004, 0.008, 0.020, 0.0008]
    nb05["zero_premium_baseline"] = european
    nb05["mean_premium_baseline"] = true + 0.020
    nb05["unconstrained_premium"] = true + 0.006
    nb05["nonnegative_premium"] = true + 0.004
    nb05["constrained_floor_prediction"] = true + 0.001

    nb06 = base.copy()
    nb06["boundary_distance_normalized"] = nb05["boundary_distance_normalized"]
    nb06["price_only_normalized_price"] = true + 0.002
    nb06["predicted_normalized_american_price"] = true + 0.003

    nb08 = pd.DataFrame(
        {
            "sample_id": sample_id,
            "true_normalized_american_price": true.astype(np.float32),
            "predicted_normalized_american_price": true + 0.004,
            "predicted_direct_normalized_american_price": true - 0.050,
        }
    )

    packages = {
        "04": _package(tmp_path, "04", nb04),
        "05": _package(tmp_path, "05", nb05),
        "06": _package(tmp_path, "06", nb06),
        "08": _package(
            tmp_path,
            "08",
            nb08,
            benchmark_frame=nb08.assign(
                predicted_normalized_american_price=true + 0.005,
                predicted_direct_normalized_american_price=true - 0.040,
            ),
        ),
    }
    return tmp_path, packages


def test_static_prediction_matrix_joins_all_models(static_project):
    project_root, packages = static_project
    matrix = build_static_prediction_matrix(project_root, packages)

    assert len(matrix) == 6
    assert "normalized_intrinsic_value" in matrix.columns
    assert matrix["normalized_intrinsic_value"].tolist() == pytest.approx(
        [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    )
    assert matrix["strike"].eq(100.0).all()
    assert matrix["sample_id"].is_unique
    for spec in MODEL_SPECS:
        assert f"prediction__{spec.model_id}" in matrix.columns


def test_phase_4_recomputes_and_ranks_common_metrics(static_project):
    project_root, packages = static_project
    results = run_phase_4_static_comparison(project_root, packages)
    assert_phase_4_ready(results)

    metrics = results["static_model_metrics"]
    best = metrics.iloc[0]
    assert best["model_id"] == "constrained_floor_residual_mlp"
    assert best["normalized_mae"] == pytest.approx(0.001)
    assert best["price_mae"] == pytest.approx(0.1)
    assert len(results["static_pairwise_error_comparison"]) == 78
    assert not results["static_segmented_pricing"].empty
    assert not results["static_boundary_pricing"].empty


def test_financial_consistency_uses_common_floor(static_project):
    project_root, packages = static_project
    results = run_phase_4_static_comparison(project_root, packages)
    consistency = results["static_financial_consistency"].set_index("model_id")

    assert consistency.loc[
        "constrained_floor_residual_mlp", "below_financial_floor_count"
    ] == 0
    assert consistency.loc[
        "integrated_warm_start_direct_price_head", "below_financial_floor_count"
    ] > 0


def test_phase_4_gate_rejects_nonfinite_prediction(static_project):
    project_root, packages = static_project
    results = run_phase_4_static_comparison(project_root, packages)
    results["static_prediction_matrix"].loc[
        0, "prediction__direct_mlp"
    ] = np.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        assert_phase_4_ready(results)
