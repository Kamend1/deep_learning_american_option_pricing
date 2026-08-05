from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd


def _install_dependency_stubs(monkeypatch) -> None:
    data_pkg = types.ModuleType("src.data")
    data_pkg.__path__ = []
    production = types.ModuleType("src.data.production_generation")
    production.CORE_RANGES = object()
    production.sample_parameter_chunk = lambda **kwargs: {
        "spot": np.full(kwargs["n_samples"], 100.0),
        "strike": np.full(kwargs["n_samples"], 100.0),
        "moneyness": np.ones(kwargs["n_samples"]),
        "log_moneyness": np.zeros(kwargs["n_samples"]),
        "time_to_maturity": np.ones(kwargs["n_samples"]),
        "volatility": np.full(kwargs["n_samples"], 0.2),
        "risk_free_rate": np.full(kwargs["n_samples"], 0.03),
        "dividend_yield": np.full(kwargs["n_samples"], 0.01),
    }
    production.price_american_put_batch = lambda *args, **kwargs: (
        np.ones(len(args[0])),
        np.ones(len(args[0])),
        np.zeros(len(args[0])),
        np.ones(len(args[0])),
        np.zeros(len(args[0]), dtype=bool),
    )
    datasets = types.ModuleType("src.data.torch_datasets")
    datasets.FEATURE_COLUMNS = (
        "log_moneyness",
        "time_to_maturity",
        "risk_free_rate",
        "dividend_yield",
        "volatility",
    )
    datasets.load_feature_scaler = lambda path: None

    support = types.ModuleType("src.evaluation.integrated_experiment_support")
    support.load_integrated_model_package = lambda *args, **kwargs: None
    support.load_premium_model_package = lambda *args, **kwargs: None

    models_pkg = types.ModuleType("src.models")
    models_pkg.__path__ = []
    integrated = types.ModuleType("src.models.integrated_multihead_pricer")
    integrated.reconstruct_integrated_outputs = lambda *args, **kwargs: {}

    for name, module in {
        "src.data": data_pkg,
        "src.data.production_generation": production,
        "src.data.torch_datasets": datasets,
        "src.evaluation.integrated_experiment_support": support,
        "src.models": models_pkg,
        "src.models.integrated_multihead_pricer": integrated,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_extrapolated_basis_is_measured_only_once(monkeypatch) -> None:
    _install_dependency_stubs(monkeypatch)
    sys.modules.pop("src.evaluation.business_case_benchmark", None)
    module = importlib.import_module("src.evaluation.business_case_benchmark")

    calls = {"runs": 0}

    def setup():
        return object()

    def run(state, frame: pd.DataFrame):
        calls["runs"] += 1
        return {"price": np.ones(len(frame))}

    method = module.BenchmarkMethod(
        method_id="slow_method",
        method="Slow method",
        family="numerical valuation",
        output_scope="price only",
        setup=setup,
        run=run,
        exact_limit=100,
    )
    config = module.RuntimeScalingConfig(
        batch_sizes=(10, 100, 1_000),
        repeats=1,
        warmup_runs=1,
        cold_repeats=1,
        accuracy_sample_size=10,
    )
    inputs = pd.DataFrame({"x": np.arange(1_000)})
    result = module.benchmark_runtime_scaling(
        [method],
        inputs,
        config=config,
    )

    # Two unique measured bases (10 and 100), each with cold, warm-up, and warm.
    assert calls["runs"] == 6
    projected = result.loc[result["requested_observations"].eq(1_000)]
    assert set(projected["measurement_type"]) == {"extrapolated"}
    assert set(projected["basis_observations"]) == {100}


def test_final_business_case_orchestrates_the_full_evidence_chain(
    monkeypatch,
    tmp_path,
) -> None:
    _install_dependency_stubs(monkeypatch)
    sys.modules.pop("src.evaluation.business_case_benchmark", None)
    sys.modules.pop("src.evaluation.final_business_case", None)
    benchmark_module = importlib.import_module(
        "src.evaluation.business_case_benchmark"
    )
    final_module = importlib.import_module(
        "src.evaluation.final_business_case"
    )

    rows = []
    definitions = {
        "project_numba_crr": (
            "Project CRR",
            "numerical valuation",
            "price and root exercise decision",
            0.002,
            4.0e-5,
        ),
        "notebook05_constrained_residual": (
            "Notebook 05",
            "static neural inference",
            "price only",
            0.010,
            1.0e-6,
        ),
        "notebook08_warm_start_integrated": (
            "Notebook 08",
            "static neural inference",
            "price and exercise decision",
            0.012,
            1.5e-6,
        ),
    }
    for method_id, (method, family, scope, fixed, marginal) in definitions.items():
        for timing_mode in ("cold", "warm"):
            mode_fixed = fixed * (2 if timing_mode == "cold" else 1)
            for observations in (10, 100, 1_000, 10_000, 100_000):
                seconds = mode_fixed + marginal * observations
                rows.append(
                    {
                        "method_id": method_id,
                        "method": method,
                        "family": family,
                        "output_scope": scope,
                        "timing_mode": timing_mode,
                        "requested_observations": observations,
                        "basis_observations": observations,
                        "measurement_type": "measured",
                        "repeats": 5,
                        "median_seconds": seconds,
                        "mean_seconds": seconds,
                        "stdev_seconds": 0.0,
                        "seconds_per_observation": seconds / observations,
                        "observations_per_second": observations / seconds,
                        "optional_method": False,
                        "notes": "synthetic",
                        "status": "complete",
                    }
                )
    scaling = pd.DataFrame(rows)
    accuracy = pd.DataFrame(
        [
            {
                "method_id": "project_numba_crr",
                "method": "Project CRR",
                "family": "numerical valuation",
                "output_scope": "price and root exercise decision",
                "observations": 100,
                "reference_method_id": "project_numba_crr",
                "price_mae": 0.0,
                "price_rmse": 0.0,
                "maximum_absolute_error": 0.0,
                "mean_error": 0.0,
                "status": "complete",
                "notes": "reference",
            }
        ]
    )

    def fake_benchmark(*args, **kwargs):
        return {
            "runtime_scaling": scaling,
            "accuracy_speed_sample": accuracy,
            "runtime_environment": {
                "torch_device": "cpu",
                "quantlib_available": False,
            },
            "benchmark_inputs": pd.DataFrame({"sample_id": [0]}),
            "methods": (),
        }

    monkeypatch.setattr(
        final_module,
        "run_business_case_benchmark",
        fake_benchmark,
    )
    static_metrics = pd.DataFrame(
        [
            {
                "model_id": "constrained_floor_residual_mlp",
                "price_mae": 0.01,
                "normalized_mae": 0.0001,
            },
            {
                "model_id": "integrated_warm_start_constrained_price",
                "price_mae": 0.03,
                "normalized_mae": 0.0003,
            },
        ]
    )
    result = final_module.run_final_business_case(
        tmp_path,
        static_model_metrics=static_metrics,
        output_dir=tmp_path / "charts",
        config=benchmark_module.RuntimeScalingConfig(
            batch_sizes=(10, 100, 1_000),
            repeats=1,
            accuracy_sample_size=10,
        ),
        include_quantlib=False,
        overrides_seconds={"deployment_preparation": 0.0},
        assumed_total_build_hours=(1.0, 4.0),
    )
    assert result["business_case_readiness_audit"]["valid"].all()
    assert not result["lifecycle_break_even"].empty
    assert result["research_question_7_summary"]["business_answer"]
    assert len(result["business_case_chart_paths"]) == 4
