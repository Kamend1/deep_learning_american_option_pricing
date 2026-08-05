"""Measured runtime scaling for the American-put business-case analysis.

The benchmark compares the deployed neural surrogates with the numerical
methods that remain available in production.  It keeps cold-start and warm
execution separate, uses the same deterministic in-domain contracts for every
method, and labels extrapolated rows explicitly when an exact numerical run
would be unnecessarily expensive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from src.data.production_generation import (
    CORE_RANGES,
    price_american_put_batch,
    sample_parameter_chunk,
)
from src.data.torch_datasets import FEATURE_COLUMNS, load_feature_scaler
from src.evaluation.integrated_experiment_support import (
    load_integrated_model_package,
    load_premium_model_package,
)
from src.models.integrated_multihead_pricer import reconstruct_integrated_outputs


SetupFunction = Callable[[], Any]
RunFunction = Callable[[Any, pd.DataFrame], Mapping[str, np.ndarray]]


@dataclass(frozen=True, slots=True)
class RuntimeScalingConfig:
    """Configuration for a reproducible CPU/GPU scaling benchmark."""

    batch_sizes: tuple[int, ...] = (
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
    )
    repeats: int = 5
    warmup_runs: int = 1
    cold_repeats: int = 1
    optional_repeats: int = 3
    seed: int = 42
    strike: float = 100.0
    crr_steps: int = 250
    quantlib_crr_steps: int = 250
    quantlib_fd_time_steps: int = 250
    quantlib_fd_grid_points: int = 250
    project_crr_exact_limit: int = 100_000
    quantlib_exact_limit: int = 1_000
    accuracy_sample_size: int = 500

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(int(size) <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if tuple(sorted(set(self.batch_sizes))) != self.batch_sizes:
            raise ValueError("batch_sizes must be strictly increasing and unique")
        if (
            self.repeats <= 0
            or self.cold_repeats <= 0
            or self.optional_repeats <= 0
        ):
            raise ValueError("repeat counts must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.crr_steps <= 0:
            raise ValueError("crr_steps must be positive")
        if self.accuracy_sample_size <= 0:
            raise ValueError("accuracy_sample_size must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkMethod:
    """One pricing or inference method under a common benchmark contract."""

    method_id: str
    method: str
    family: str
    output_scope: str
    setup: SetupFunction
    run: RunFunction
    exact_limit: int | None = None
    optional: bool = False
    notes: str = ""


def _synchronize(device: torch.device | None = None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _black_scholes_put_vectorized(frame: pd.DataFrame) -> np.ndarray:
    spot = frame["spot"].to_numpy(dtype=np.float64)
    strike = frame["strike"].to_numpy(dtype=np.float64)
    maturity = frame["time_to_maturity"].to_numpy(dtype=np.float64)
    rate = frame["risk_free_rate"].to_numpy(dtype=np.float64)
    dividend = frame["dividend_yield"].to_numpy(dtype=np.float64)
    volatility = frame["volatility"].to_numpy(dtype=np.float64)

    root_t = np.sqrt(maturity)
    denominator = volatility * root_t
    d1 = (
        np.log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / denominator
    d2 = d1 - denominator
    return (
        strike * np.exp(-rate * maturity) * ndtr(-d2)
        - spot * np.exp(-dividend * maturity) * ndtr(-d1)
    )


def build_business_case_inputs(
    observations: int,
    *,
    seed: int = 42,
    strike: float = 100.0,
) -> pd.DataFrame:
    """Create deterministic in-domain contracts without pricing them by CRR."""

    if observations <= 0:
        raise ValueError("observations must be positive")
    parameters = sample_parameter_chunk(
        n_samples=int(observations),
        ranges=CORE_RANGES,
        seed=int(seed),
        strike=float(strike),
    )
    frame = pd.DataFrame(parameters)
    european = _black_scholes_put_vectorized(frame)
    intrinsic = np.maximum(
        frame["strike"].to_numpy(dtype=np.float64)
        - frame["spot"].to_numpy(dtype=np.float64),
        0.0,
    )
    strike_values = frame["strike"].to_numpy(dtype=np.float64)
    frame["normalized_european_price"] = european / strike_values
    frame["normalized_intrinsic_value"] = intrinsic / strike_values
    frame["normalized_financial_floor"] = np.maximum(
        frame["normalized_european_price"],
        frame["normalized_intrinsic_value"],
    )
    frame.insert(0, "sample_id", np.arange(observations, dtype=np.int64))
    return frame


def _scaled_feature_tensor(
    frame: pd.DataFrame,
    *,
    scaler: Any,
    device: torch.device,
) -> torch.Tensor:
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Benchmark inputs are missing feature columns: {missing}")
    transformed = scaler.transform(frame.loc[:, list(FEATURE_COLUMNS)])
    values = np.asarray(transformed, dtype=np.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _project_crr_method(config: RuntimeScalingConfig) -> BenchmarkMethod:
    def setup() -> dict[str, int]:
        return {"steps": int(config.crr_steps)}

    def run(state: Mapping[str, int], frame: pd.DataFrame) -> Mapping[str, np.ndarray]:
        european, raw_american, intrinsic, continuation, exercise_now = (
            price_american_put_batch(
                frame["spot"].to_numpy(dtype=np.float64),
                frame["strike"].to_numpy(dtype=np.float64),
                frame["time_to_maturity"].to_numpy(dtype=np.float64),
                frame["risk_free_rate"].to_numpy(dtype=np.float64),
                frame["dividend_yield"].to_numpy(dtype=np.float64),
                frame["volatility"].to_numpy(dtype=np.float64),
                int(state["steps"]),
            )
        )
        price = np.maximum.reduce((raw_american, european, intrinsic))
        return {
            "price": price,
            "exercise_probability": exercise_now.astype(np.float64),
            "continuation_value": continuation,
        }

    return BenchmarkMethod(
        method_id="project_numba_crr",
        method="Project high-resolution Numba CRR",
        family="numerical valuation",
        output_scope="price and root exercise decision",
        setup=setup,
        run=run,
        exact_limit=int(config.project_crr_exact_limit),
        notes="Production 250-step CRR label generator with no-arbitrage floor repair.",
    )


def _notebook05_method(project_root: Path, device: torch.device) -> BenchmarkMethod:
    checkpoint_path = project_root / "artifacts/premium_models/best_premium_model.pt"
    scaler_path = project_root / "artifacts/direct_mlp/feature_scaler.joblib"

    def setup() -> dict[str, Any]:
        model, checkpoint = load_premium_model_package(
            checkpoint_path,
            device=device,
        )
        scaler = load_feature_scaler(scaler_path)
        return {"model": model, "checkpoint": checkpoint, "scaler": scaler}

    def run(state: Mapping[str, Any], frame: pd.DataFrame) -> Mapping[str, np.ndarray]:
        features = _scaled_feature_tensor(
            frame,
            scaler=state["scaler"],
            device=device,
        )
        _synchronize(device)
        with torch.inference_mode():
            residual = state["model"](features).reshape(-1)
            floor = torch.as_tensor(
                frame["normalized_financial_floor"].to_numpy(dtype=np.float32),
                device=device,
            )
            price = floor + residual
            output = price.detach().cpu().numpy()
        _synchronize(device)
        return {"normalized_price": output}

    return BenchmarkMethod(
        method_id="notebook05_constrained_residual",
        method="Notebook 05 constrained residual model",
        family="static neural inference",
        output_scope="price only",
        setup=setup,
        run=run,
        notes="Canonical validation-selected Notebook 05 deployment checkpoint.",
    )


def _notebook08_method(project_root: Path, device: torch.device) -> BenchmarkMethod:
    checkpoint_path = project_root / "artifacts/final_multihead/best_integrated_deployment.pt"
    scaler_path = project_root / "artifacts/final_multihead/feature_scaler.joblib"

    def setup() -> dict[str, Any]:
        model, checkpoint, model_config, loss_config = load_integrated_model_package(
            checkpoint_path,
            device=device,
        )
        scaler = load_feature_scaler(scaler_path)
        return {
            "model": model,
            "checkpoint": checkpoint,
            "model_config": model_config,
            "loss_config": loss_config,
            "scaler": scaler,
        }

    def run(state: Mapping[str, Any], frame: pd.DataFrame) -> Mapping[str, np.ndarray]:
        features = _scaled_feature_tensor(
            frame,
            scaler=state["scaler"],
            device=device,
        )
        european = torch.as_tensor(
            frame["normalized_european_price"].to_numpy(dtype=np.float32),
            device=device,
        )
        intrinsic = torch.as_tensor(
            frame["normalized_intrinsic_value"].to_numpy(dtype=np.float32),
            device=device,
        )
        _synchronize(device)
        with torch.inference_mode():
            raw = state["model"](features)
            reconstructed = reconstruct_integrated_outputs(
                raw,
                normalized_european=european,
                normalized_intrinsic=intrinsic,
                decision_sharpness=float(
                    state["loss_config"].decision_sharpness
                ),
            )
            price = reconstructed["constrained_price"].detach().cpu().numpy()
            probability = (
                reconstructed["exercise_probability"].detach().cpu().numpy()
            )
        _synchronize(device)
        return {
            "normalized_price": price.reshape(-1),
            "exercise_probability": probability.reshape(-1),
        }

    return BenchmarkMethod(
        method_id="notebook08_warm_start_integrated",
        method="Notebook 08 warm-start integrated model",
        family="static neural inference",
        output_scope="price and exercise decision",
        setup=setup,
        run=run,
        notes="Preferred in-domain integrated deployment checkpoint from Notebook 08.",
    )


def _quantlib_available() -> tuple[bool, str | None]:
    try:
        import QuantLib as ql  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, repr(exc)
    return True, getattr(ql, "__version__", "unknown")


def _quantlib_price_one(
    ql: Any,
    row: Any,
    *,
    engine: str,
    steps: int,
    fd_time_steps: int,
    fd_grid_points: int,
) -> float:
    evaluation_date = ql.Date(2, 1, 2024)
    ql.Settings.instance().evaluationDate = evaluation_date
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    maturity_days = max(1, int(round(float(row.time_to_maturity) * 365.0)))
    maturity_date = evaluation_date + maturity_days

    spot_quote = ql.SimpleQuote(float(row.spot))
    spot_handle = ql.QuoteHandle(spot_quote)
    risk_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, float(row.risk_free_rate), day_count)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, float(row.dividend_yield), day_count)
    )
    volatility_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            evaluation_date,
            calendar,
            float(row.volatility),
            day_count,
        )
    )
    process = ql.BlackScholesMertonProcess(
        spot_handle,
        dividend_curve,
        risk_curve,
        volatility_curve,
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(row.strike))
    exercise = ql.AmericanExercise(evaluation_date, maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    if engine == "binomial_crr":
        pricing_engine = ql.BinomialVanillaEngine(process, "crr", int(steps))
    elif engine == "finite_difference":
        pricing_engine = ql.FdBlackScholesVanillaEngine(
            process,
            int(fd_time_steps),
            int(fd_grid_points),
        )
    else:  # pragma: no cover - internal contract
        raise ValueError(f"Unsupported QuantLib engine: {engine}")
    option.setPricingEngine(pricing_engine)
    return float(option.NPV())


def _quantlib_method(
    config: RuntimeScalingConfig,
    *,
    engine: str,
) -> BenchmarkMethod | None:
    available, version_or_error = _quantlib_available()
    if not available:
        return None

    label = (
        "QuantLib binomial CRR"
        if engine == "binomial_crr"
        else "QuantLib finite-difference American put"
    )

    def setup() -> dict[str, Any]:
        import QuantLib as ql  # type: ignore

        return {"ql": ql, "version": getattr(ql, "__version__", "unknown")}

    def run(state: Mapping[str, Any], frame: pd.DataFrame) -> Mapping[str, np.ndarray]:
        ql = state["ql"]
        prices = np.fromiter(
            (
                _quantlib_price_one(
                    ql,
                    row,
                    engine=engine,
                    steps=config.quantlib_crr_steps,
                    fd_time_steps=config.quantlib_fd_time_steps,
                    fd_grid_points=config.quantlib_fd_grid_points,
                )
                for row in frame.itertuples(index=False)
            ),
            dtype=np.float64,
            count=len(frame),
        )
        return {"price": prices}

    return BenchmarkMethod(
        method_id=(
            "quantlib_binomial_crr"
            if engine == "binomial_crr"
            else "quantlib_finite_difference"
        ),
        method=label,
        family="numerical valuation",
        output_scope="price only",
        setup=setup,
        run=run,
        exact_limit=int(config.quantlib_exact_limit),
        optional=True,
        notes=f"Optional QuantLib {version_or_error} benchmark.",
    )


def build_business_case_methods(
    project_root: Path,
    *,
    device: str | torch.device = "cpu",
    config: RuntimeScalingConfig | None = None,
    include_quantlib: bool = True,
) -> tuple[BenchmarkMethod, ...]:
    """Build the project, neural, and optional standard-library methods."""

    root = Path(project_root).resolve()
    runtime_config = config or RuntimeScalingConfig()
    torch_device = torch.device(device)
    methods: list[BenchmarkMethod] = [
        _project_crr_method(runtime_config),
        _notebook05_method(root, torch_device),
        _notebook08_method(root, torch_device),
    ]
    if include_quantlib:
        for engine in ("binomial_crr", "finite_difference"):
            method = _quantlib_method(runtime_config, engine=engine)
            if method is not None:
                methods.append(method)
    return tuple(methods)


def _time_call(
    function: Callable[[], Mapping[str, np.ndarray]],
    *,
    device: torch.device | None = None,
) -> tuple[float, Mapping[str, np.ndarray]]:
    _synchronize(device)
    started = time.perf_counter()
    output = function()
    _synchronize(device)
    return float(time.perf_counter() - started), output


def _validate_output(output: Mapping[str, np.ndarray], observations: int) -> None:
    if not output:
        raise ValueError("Benchmark method returned no outputs")
    for name, values in output.items():
        array = np.asarray(values)
        if array.reshape(-1).shape[0] != observations:
            raise ValueError(
                f"Output {name!r} has {array.reshape(-1).shape[0]} rows; "
                f"expected {observations}"
            )
        if not np.isfinite(array.astype(np.float64)).all():
            raise ValueError(f"Output {name!r} contains non-finite values")


def _measured_scaling_rows(
    method: BenchmarkMethod,
    frame: pd.DataFrame,
    *,
    requested_size: int,
    measured_size: int,
    config: RuntimeScalingConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    subset = frame.iloc[:measured_size].copy()
    rows: list[dict[str, Any]] = []

    cold_times: list[float] = []
    for _ in range(config.cold_repeats):
        elapsed, output = _time_call(
            lambda: method.run(method.setup(), subset),
            device=device,
        )
        _validate_output(output, measured_size)
        cold_times.append(elapsed)

    state = method.setup()
    for _ in range(config.warmup_runs):
        output = method.run(state, subset)
        _validate_output(output, measured_size)

    warm_times: list[float] = []
    repeat_count = (
        min(config.repeats, config.optional_repeats)
        if method.optional
        else config.repeats
    )
    for _ in range(repeat_count):
        elapsed, output = _time_call(
            lambda: method.run(state, subset),
            device=device,
        )
        _validate_output(output, measured_size)
        warm_times.append(elapsed)

    for timing_mode, values in (("cold", cold_times), ("warm", warm_times)):
        median_seconds = float(statistics.median(values))
        mean_seconds = float(statistics.fmean(values))
        stdev_seconds = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        measurement_type = "measured" if requested_size == measured_size else "extrapolated"
        if measurement_type == "extrapolated":
            scale = requested_size / measured_size
            median_seconds *= scale
            mean_seconds *= scale
            stdev_seconds *= scale
        rows.append(
            {
                "method_id": method.method_id,
                "method": method.method,
                "family": method.family,
                "output_scope": method.output_scope,
                "timing_mode": timing_mode,
                "requested_observations": int(requested_size),
                "basis_observations": int(measured_size),
                "measurement_type": measurement_type,
                "repeats": int(len(values)),
                "median_seconds": median_seconds,
                "mean_seconds": mean_seconds,
                "stdev_seconds": stdev_seconds,
                "seconds_per_observation": median_seconds / requested_size,
                "observations_per_second": (
                    requested_size / median_seconds if median_seconds > 0.0 else np.nan
                ),
                "optional_method": bool(method.optional),
                "notes": method.notes,
                "status": "complete",
            }
        )
    return rows


def _project_measurement_rows(
    basis_rows: Sequence[Mapping[str, Any]],
    *,
    requested_size: int,
    basis_size: int,
) -> list[dict[str, Any]]:
    """Reuse one measured basis for larger explicitly extrapolated workloads."""

    projected: list[dict[str, Any]] = []
    for raw in basis_rows:
        row = dict(raw)
        row["requested_observations"] = int(requested_size)
        row["basis_observations"] = int(basis_size)
        if requested_size == basis_size:
            row["measurement_type"] = "measured"
        else:
            row["measurement_type"] = "extrapolated"
            scale = float(requested_size) / float(basis_size)
            for column in (
                "median_seconds",
                "mean_seconds",
                "stdev_seconds",
            ):
                row[column] = float(row[column]) * scale
        median = float(row["median_seconds"])
        row["seconds_per_observation"] = median / float(requested_size)
        row["observations_per_second"] = (
            float(requested_size) / median if median > 0.0 else np.nan
        )
        projected.append(row)
    return projected


def benchmark_runtime_scaling(
    methods: Sequence[BenchmarkMethod],
    inputs: pd.DataFrame,
    *,
    config: RuntimeScalingConfig | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Measure each unique basis once and preserve optional-method failures."""

    runtime_config = config or RuntimeScalingConfig()
    torch_device = torch.device(device)
    maximum = max(runtime_config.batch_sizes)
    if len(inputs) < maximum:
        raise ValueError(
            f"Benchmark input has {len(inputs)} rows; at least {maximum} are required"
        )

    rows: list[dict[str, Any]] = []
    for method in methods:
        basis_cache: dict[int, list[dict[str, Any]]] = {}
        basis_errors: dict[int, Exception] = {}
        for requested_size in runtime_config.batch_sizes:
            exact_limit = method.exact_limit or requested_size
            basis_size = min(int(requested_size), int(exact_limit))
            if basis_size not in basis_cache and basis_size not in basis_errors:
                try:
                    basis_cache[basis_size] = _measured_scaling_rows(
                        method,
                        inputs,
                        requested_size=basis_size,
                        measured_size=basis_size,
                        config=runtime_config,
                        device=torch_device,
                    )
                except Exception as exc:
                    if not method.optional:
                        raise
                    basis_errors[basis_size] = exc

            if basis_size in basis_cache:
                rows.extend(
                    _project_measurement_rows(
                        basis_cache[basis_size],
                        requested_size=int(requested_size),
                        basis_size=basis_size,
                    )
                )
                continue

            exc = basis_errors[basis_size]
            for timing_mode in ("cold", "warm"):
                rows.append(
                    {
                        "method_id": method.method_id,
                        "method": method.method,
                        "family": method.family,
                        "output_scope": method.output_scope,
                        "timing_mode": timing_mode,
                        "requested_observations": int(requested_size),
                        "basis_observations": int(basis_size),
                        "measurement_type": "failed",
                        "repeats": 0,
                        "median_seconds": np.nan,
                        "mean_seconds": np.nan,
                        "stdev_seconds": np.nan,
                        "seconds_per_observation": np.nan,
                        "observations_per_second": np.nan,
                        "optional_method": True,
                        "notes": f"{method.notes} Error: {exc!r}",
                        "status": "failed",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["method_id", "timing_mode", "requested_observations"],
        kind="stable",
    ).reset_index(drop=True)


def benchmark_accuracy_sample(
    methods: Sequence[BenchmarkMethod],
    inputs: pd.DataFrame,
    *,
    observations: int,
    strike: float,
) -> pd.DataFrame:
    """Compare method prices on one small deterministic sample."""

    subset = inputs.iloc[: int(observations)].copy()
    method_by_id = {method.method_id: method for method in methods}
    if "project_numba_crr" not in method_by_id:
        raise KeyError("The project CRR method is required as the accuracy reference")

    reference_method = method_by_id["project_numba_crr"]
    reference = reference_method.run(reference_method.setup(), subset)
    reference_price = np.asarray(reference["price"], dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for method in methods:
        try:
            output = method.run(method.setup(), subset)
            if "price" in output:
                predicted_units = np.asarray(output["price"], dtype=np.float64)
            elif "normalized_price" in output:
                predicted_units = (
                    np.asarray(output["normalized_price"], dtype=np.float64)
                    * float(strike)
                )
            else:
                raise KeyError("Method does not return a price output")
            error = predicted_units - reference_price
            rows.append(
                {
                    "method_id": method.method_id,
                    "method": method.method,
                    "family": method.family,
                    "output_scope": method.output_scope,
                    "observations": int(len(subset)),
                    "reference_method_id": "project_numba_crr",
                    "price_mae": float(np.mean(np.abs(error))),
                    "price_rmse": float(np.sqrt(np.mean(error**2))),
                    "maximum_absolute_error": float(np.max(np.abs(error))),
                    "mean_error": float(np.mean(error)),
                    "status": "complete",
                    "notes": method.notes,
                }
            )
        except Exception as exc:
            if not method.optional:
                raise
            rows.append(
                {
                    "method_id": method.method_id,
                    "method": method.method,
                    "family": method.family,
                    "output_scope": method.output_scope,
                    "observations": int(len(subset)),
                    "reference_method_id": "project_numba_crr",
                    "price_mae": np.nan,
                    "price_rmse": np.nan,
                    "maximum_absolute_error": np.nan,
                    "mean_error": np.nan,
                    "status": "failed",
                    "notes": f"{method.notes} Error: {exc!r}",
                }
            )
    return pd.DataFrame(rows)


def build_runtime_environment(
    *,
    config: RuntimeScalingConfig,
    device: str | torch.device,
) -> dict[str, Any]:
    available, quantlib_version_or_error = _quantlib_available()
    torch_device = torch.device(device)
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "torch_device": str(torch_device),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "quantlib_available": bool(available),
        "quantlib_version_or_error": quantlib_version_or_error,
        "cpu_count": os.cpu_count(),
        "benchmark_config": asdict(config),
    }


def run_business_case_benchmark(
    project_root: Path,
    *,
    config: RuntimeScalingConfig | None = None,
    device: str | torch.device = "cpu",
    include_quantlib: bool = True,
) -> dict[str, Any]:
    """Run the complete measured benchmark without changing model artifacts."""

    runtime_config = config or RuntimeScalingConfig()
    inputs = build_business_case_inputs(
        max(runtime_config.batch_sizes),
        seed=runtime_config.seed,
        strike=runtime_config.strike,
    )
    methods = build_business_case_methods(
        Path(project_root),
        device=device,
        config=runtime_config,
        include_quantlib=include_quantlib,
    )
    scaling = benchmark_runtime_scaling(
        methods,
        inputs,
        config=runtime_config,
        device=device,
    )
    accuracy = benchmark_accuracy_sample(
        methods,
        inputs,
        observations=min(runtime_config.accuracy_sample_size, len(inputs)),
        strike=runtime_config.strike,
    )
    return {
        "runtime_scaling": scaling,
        "accuracy_speed_sample": accuracy,
        "runtime_environment": build_runtime_environment(
            config=runtime_config,
            device=device,
        ),
        "benchmark_inputs": inputs,
        "methods": methods,
    }


__all__ = [
    "BenchmarkMethod",
    "RuntimeScalingConfig",
    "benchmark_accuracy_sample",
    "benchmark_runtime_scaling",
    "build_business_case_inputs",
    "build_business_case_methods",
    "build_runtime_environment",
    "run_business_case_benchmark",
]
