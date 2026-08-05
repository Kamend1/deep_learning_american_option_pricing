"""Operational crossover and lifecycle break-even analysis.

The functions in this module distinguish the marginal question (when one
valuation job is faster) from the lifecycle question (when data generation and
training have been repaid).  Missing historical timing data is never invented:
measured manifest values are used where available and clearly labelled
scenarios are generated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RuntimeCurveConfig:
    """Rules for fitting simple fixed-plus-marginal runtime curves."""

    minimum_observations: int = 10
    minimum_points: int = 3
    maximum_points: int = 5

    def __post_init__(self) -> None:
        if self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be positive")
        if self.minimum_points < 2:
            raise ValueError("minimum_points must be at least two")
        if self.maximum_points < self.minimum_points:
            raise ValueError("maximum_points cannot be below minimum_points")


DEFAULT_LABEL_GENERATION_HOUR_SCENARIOS = (
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    12.0,
    24.0,
)

DEFAULT_TOTAL_BUILD_HOUR_SCENARIOS = (
    1.0,
    4.0,
    8.0,
    24.0,
    48.0,
)


METHOD_DEPLOYMENT_IDS = {
    "notebook05_constrained_residual": "notebook05_price_only",
    "notebook08_warm_start_integrated": "notebook08_combined",
}


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def fit_runtime_curves(
    runtime_scaling: pd.DataFrame,
    *,
    config: RuntimeCurveConfig | None = None,
) -> pd.DataFrame:
    """Fit ``total_seconds = fixed_seconds + marginal_seconds * N`` curves."""

    curve_config = config or RuntimeCurveConfig()
    required = {
        "method_id",
        "method",
        "family",
        "output_scope",
        "timing_mode",
        "requested_observations",
        "median_seconds",
        "measurement_type",
        "status",
    }
    missing = sorted(required.difference(runtime_scaling.columns))
    if missing:
        raise ValueError(f"Runtime scaling table is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (method_id, timing_mode), group in runtime_scaling.groupby(
        ["method_id", "timing_mode"],
        sort=False,
    ):
        usable = group.loc[
            group["status"].astype(str).eq("complete")
            & group["measurement_type"].astype(str).eq("measured")
        ].copy()
        usable["requested_observations"] = pd.to_numeric(
            usable["requested_observations"], errors="coerce"
        )
        usable["median_seconds"] = pd.to_numeric(
            usable["median_seconds"], errors="coerce"
        )
        usable = usable.loc[
            usable["requested_observations"].ge(curve_config.minimum_observations)
            & usable["median_seconds"].gt(0.0)
        ].sort_values("requested_observations")
        usable = usable.tail(curve_config.maximum_points)

        representative = group.iloc[0]
        if len(usable) < curve_config.minimum_points:
            rows.append(
                {
                    "method_id": str(method_id),
                    "method": representative["method"],
                    "family": representative["family"],
                    "output_scope": representative["output_scope"],
                    "timing_mode": str(timing_mode),
                    "fixed_seconds": np.nan,
                    "marginal_seconds_per_observation": np.nan,
                    "observations_per_second_at_scale": np.nan,
                    "fit_points": int(len(usable)),
                    "fit_min_observations": np.nan,
                    "fit_max_observations": np.nan,
                    "r_squared": np.nan,
                    "status": "insufficient_data",
                }
            )
            continue

        x = usable["requested_observations"].to_numpy(dtype=np.float64)
        y = usable["median_seconds"].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        slope = max(float(slope), 0.0)
        intercept = max(float(intercept), 0.0)
        fitted = intercept + slope * x
        residual = float(np.sum((y - fitted) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - residual / total if total > 0.0 else 1.0
        rows.append(
            {
                "method_id": str(method_id),
                "method": representative["method"],
                "family": representative["family"],
                "output_scope": representative["output_scope"],
                "timing_mode": str(timing_mode),
                "fixed_seconds": intercept,
                "marginal_seconds_per_observation": slope,
                "observations_per_second_at_scale": (
                    1.0 / slope if slope > 0.0 else np.inf
                ),
                "fit_points": int(len(usable)),
                "fit_min_observations": int(x.min()),
                "fit_max_observations": int(x.max()),
                "r_squared": float(r_squared),
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def _curve_row(
    curves: pd.DataFrame,
    method_id: str,
    *,
    timing_mode: str = "warm",
) -> pd.Series:
    match = curves.loc[
        curves["method_id"].astype(str).eq(str(method_id))
        & curves["timing_mode"].astype(str).eq(str(timing_mode))
        & curves["status"].astype(str).eq("complete")
    ]
    if match.empty:
        raise KeyError(f"No complete {timing_mode} runtime curve for {method_id}")
    return match.iloc[0]


def predict_runtime_seconds(
    curve: pd.Series | Mapping[str, Any],
    observations: int | float,
) -> float:
    n = float(observations)
    if n < 0.0:
        raise ValueError("observations cannot be negative")
    fixed = _numeric(curve["fixed_seconds"])
    marginal = _numeric(curve["marginal_seconds_per_observation"])
    if not np.isfinite(fixed) or not np.isfinite(marginal):
        return float("nan")
    return max(fixed + marginal * n, 0.0)


def build_operational_crossover(
    runtime_curves: pd.DataFrame,
    runtime_scaling: pd.DataFrame,
    *,
    neural_method_ids: Sequence[str] = tuple(METHOD_DEPLOYMENT_IDS),
    numerical_method_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate one-job crossover sizes for every neural/numerical pair."""

    if numerical_method_ids is None:
        numerical_method_ids = tuple(
            runtime_curves.loc[
                runtime_curves["family"].astype(str).eq("numerical valuation"),
                "method_id",
            ].astype(str).unique()
        )

    rows: list[dict[str, Any]] = []
    for timing_mode in ("cold", "warm"):
        for neural_id in neural_method_ids:
            try:
                neural = _curve_row(
                    runtime_curves,
                    neural_id,
                    timing_mode=timing_mode,
                )
            except KeyError:
                continue
            for numerical_id in numerical_method_ids:
                try:
                    numerical = _curve_row(
                        runtime_curves,
                        numerical_id,
                        timing_mode=timing_mode,
                    )
                except KeyError:
                    continue

                neural_fixed = _numeric(neural["fixed_seconds"])
                numerical_fixed = _numeric(numerical["fixed_seconds"])
                neural_marginal = _numeric(
                    neural["marginal_seconds_per_observation"]
                )
                numerical_marginal = _numeric(
                    numerical["marginal_seconds_per_observation"]
                )
                marginal_saving = numerical_marginal - neural_marginal
                if marginal_saving <= 0.0:
                    crossover = np.nan
                    status = "no_crossover_neural_not_faster_at_scale"
                else:
                    raw = (neural_fixed - numerical_fixed) / marginal_saving
                    crossover = max(int(math.ceil(raw)), 1)
                    status = "complete"

                pair = runtime_scaling.loc[
                    runtime_scaling["method_id"].astype(str).isin(
                        [neural_id, numerical_id]
                    )
                    & runtime_scaling["timing_mode"].astype(str).eq(timing_mode)
                    & runtime_scaling["status"].astype(str).eq("complete")
                ]
                pivot = pair.pivot_table(
                    index="requested_observations",
                    columns="method_id",
                    values="median_seconds",
                    aggfunc="first",
                )
                measured_crossover = np.nan
                if {neural_id, numerical_id}.issubset(pivot.columns):
                    faster = pivot[neural_id] < pivot[numerical_id]
                    if faster.any():
                        measured_crossover = int(pivot.index[faster][0])

                rows.append(
                    {
                        "timing_mode": timing_mode,
                        "neural_method_id": neural_id,
                        "neural_method": neural["method"],
                        "neural_output_scope": neural["output_scope"],
                        "numerical_method_id": numerical_id,
                        "numerical_method": numerical["method"],
                        "curve_crossover_observations": crossover,
                        "smallest_benchmark_batch_neural_faster": measured_crossover,
                        "neural_marginal_seconds": neural_marginal,
                        "numerical_marginal_seconds": numerical_marginal,
                        "marginal_seconds_saved_per_valuation": marginal_saving,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_numeric_key(value: Any, candidate_keys: set[str]) -> float | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in candidate_keys:
                number = _numeric(item)
                if np.isfinite(number) and number >= 0.0:
                    return number
        for item in value.values():
            found = _find_numeric_key(item, candidate_keys)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found = _find_numeric_key(item, candidate_keys)
            if found is not None:
                return found
    return None


def _component_record(
    component_id: str,
    component: str,
    path: Path | None,
    keys: Iterable[str],
    *,
    override_seconds: float | None,
    required_for: str,
) -> dict[str, Any]:
    if override_seconds is not None:
        value = float(override_seconds)
        if value < 0.0:
            raise ValueError(f"Negative override for {component_id}")
        return {
            "component_id": component_id,
            "component": component,
            "seconds": value,
            "source": "user_override",
            "source_path": None,
            "status": "complete",
            "required_for": required_for,
        }

    payload = _read_json(path) if path is not None and path.is_file() else None
    value = _find_numeric_key(payload, set(keys)) if payload is not None else None
    return {
        "component_id": component_id,
        "component": component,
        "seconds": value if value is not None else np.nan,
        "source": "saved_manifest" if value is not None else "missing",
        "source_path": str(path) if path is not None else None,
        "status": "complete" if value is not None else "missing",
        "required_for": required_for,
    }


def build_upfront_cost_inventory(
    project_root: Path,
    *,
    overrides_seconds: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Read measured build times and retain missing values transparently."""

    root = Path(project_root).resolve()
    overrides = dict(overrides_seconds or {})
    timing_keys = {
        "training_runtime_seconds",
        "runtime_seconds",
        "elapsed_seconds",
        "generation_runtime_seconds",
        "total_runtime_seconds",
    }

    data_candidates = [
        root / "data/manifests/production_dataset_manifest.json",
        root / "data/manifests/production_generation_manifest.json",
        root / "artifacts/data_generation/generation_complete.json",
        root / "artifacts/data_generation/runtime_summary.json",
    ]
    data_path = next((path for path in data_candidates if path.is_file()), data_candidates[0])

    specifications = [
        (
            "production_label_generation",
            "Generate the production CRR labels",
            data_path,
            timing_keys,
            "all neural deployments",
        ),
        (
            "notebook04_direct_training",
            "Train the direct pricing baseline",
            root / "artifacts/direct_mlp/training_complete.json",
            timing_keys,
            "full research programme",
        ),
        (
            "notebook05_training",
            "Train the selected Notebook 05 price model candidates",
            root / "artifacts/premium_models/training_complete.json",
            timing_keys,
            "Notebook 05 deployment and full research programme",
        ),
        (
            "notebook06_classifier_training",
            "Train the Notebook 06 exercise classifier",
            root / "artifacts/multitask_model/exercise_classifier_complete.json",
            timing_keys,
            "full research programme",
        ),
        (
            "notebook06_multitask_training",
            "Train the Notebook 06 multi-task predecessor",
            root / "artifacts/multitask_model/multitask_training_complete.json",
            timing_keys,
            "Notebook 08 warm-start deployment and full research programme",
        ),
        (
            "notebook07_training",
            "Train the neural Longstaff–Schwartz policy",
            root / "artifacts/neural_lsm/training_complete.json",
            timing_keys,
            "full research programme",
        ),
        (
            "notebook08_scratch_training",
            "Train the three Notebook 08 scratch candidates",
            root / "artifacts/final_multihead/scratch_training_complete.json",
            timing_keys,
            "full research programme",
        ),
        (
            "notebook08_warm_start_training",
            "Train the preferred Notebook 08 warm-start deployment model",
            root / "artifacts/final_multihead/warm_start_training_complete.json",
            timing_keys,
            "Notebook 08 warm-start deployment and full research programme",
        ),
        (
            "deployment_preparation",
            "Serialize, validate, and prepare the deployment package",
            None,
            timing_keys,
            "all neural deployments",
        ),
    ]

    rows = [
        _component_record(
            component_id,
            component,
            path,
            keys,
            override_seconds=overrides.get(component_id),
            required_for=required_for,
        )
        for component_id, component, path, keys, required_for in specifications
    ]
    return pd.DataFrame(rows)


def _sum_components(
    inventory: pd.DataFrame,
    component_ids: Sequence[str],
) -> tuple[float, list[str]]:
    subset = inventory.set_index("component_id").reindex(component_ids)
    missing = subset.index[subset["seconds"].isna()].astype(str).tolist()
    total = float(pd.to_numeric(subset["seconds"], errors="coerce").sum(min_count=1))
    return total, missing


def build_upfront_cost_scenarios(
    inventory: pd.DataFrame,
    *,
    assumed_label_generation_hours: Sequence[float] = DEFAULT_LABEL_GENERATION_HOUR_SCENARIOS,
    assumed_total_build_hours: Sequence[float] = DEFAULT_TOTAL_BUILD_HOUR_SCENARIOS,
) -> pd.DataFrame:
    """Build measured and scenario-based deployment cost packages."""

    required_columns = {"component_id", "seconds", "status"}
    missing_columns = sorted(required_columns.difference(inventory.columns))
    if missing_columns:
        raise ValueError(f"Upfront inventory is missing columns: {missing_columns}")

    packages = {
        "notebook05_price_only": (
            "Minimum reproducible Notebook 05 deployment",
            [
                "production_label_generation",
                "notebook05_training",
                "deployment_preparation",
            ],
        ),
        "notebook08_combined": (
            "Minimum reproducible Notebook 08 combined deployment",
            [
                "production_label_generation",
                "notebook06_multitask_training",
                "notebook08_warm_start_training",
                "deployment_preparation",
            ],
        ),
        "full_research_programme": (
            "Full research programme including rejected candidates",
            [
                "production_label_generation",
                "notebook04_direct_training",
                "notebook05_training",
                "notebook06_classifier_training",
                "notebook06_multitask_training",
                "notebook07_training",
                "notebook08_scratch_training",
                "notebook08_warm_start_training",
                "deployment_preparation",
            ],
        ),
    }

    rows: list[dict[str, Any]] = []
    label_row = inventory.loc[
        inventory["component_id"].astype(str).eq("production_label_generation")
    ]
    label_measured = (
        not label_row.empty and np.isfinite(_numeric(label_row.iloc[0]["seconds"]))
    )

    for deployment_id, (label, component_ids) in packages.items():
        total, missing = _sum_components(inventory, component_ids)
        non_label_missing = [
            item for item in missing if item != "production_label_generation"
        ]
        if not missing:
            rows.append(
                {
                    "deployment_id": deployment_id,
                    "deployment_scope": label,
                    "scenario_id": "measured_saved_manifests",
                    "upfront_seconds": total,
                    "upfront_hours": total / 3600.0,
                    "label_generation_hours": (
                        _numeric(label_row.iloc[0]["seconds"]) / 3600.0
                        if label_measured
                        else np.nan
                    ),
                    "evidence_type": "measured",
                    "missing_components": "",
                    "status": "complete",
                }
            )
            continue

        if non_label_missing:
            # Historical research runs do not always retain every wall-clock
            # component. Preserve the gap and provide an explicit total-build
            # scenario range rather than silently treating missing time as zero.
            rows.append(
                {
                    "deployment_id": deployment_id,
                    "deployment_scope": label,
                    "scenario_id": "incomplete_saved_manifests",
                    "upfront_seconds": np.nan,
                    "upfront_hours": np.nan,
                    "label_generation_hours": np.nan,
                    "evidence_type": "incomplete",
                    "missing_components": ", ".join(missing),
                    "status": "missing_components",
                }
            )
            for hours in assumed_total_build_hours:
                total_seconds = float(hours) * 3600.0
                rows.append(
                    {
                        "deployment_id": deployment_id,
                        "deployment_scope": label,
                        "scenario_id": f"assumed_total_build_{float(hours):g}h",
                        "upfront_seconds": total_seconds,
                        "upfront_hours": float(hours),
                        "label_generation_hours": np.nan,
                        "evidence_type": "total_build_scenario",
                        "missing_components": ", ".join(missing),
                        "status": "complete",
                    }
                )
            continue

        base_ids = [
            item for item in component_ids if item != "production_label_generation"
        ]
        base_total, base_missing = _sum_components(inventory, base_ids)
        if base_missing:
            continue
        for hours in assumed_label_generation_hours:
            label_seconds = float(hours) * 3600.0
            scenario_total = base_total + label_seconds
            rows.append(
                {
                    "deployment_id": deployment_id,
                    "deployment_scope": label,
                    "scenario_id": f"assumed_label_generation_{float(hours):g}h",
                    "upfront_seconds": scenario_total,
                    "upfront_hours": scenario_total / 3600.0,
                    "label_generation_hours": float(hours),
                    "evidence_type": "scenario",
                    "missing_components": "production_label_generation",
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows)


def build_lifecycle_break_even(
    runtime_curves: pd.DataFrame,
    upfront_scenarios: pd.DataFrame,
    *,
    numerical_method_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate cumulative valuations required to repay neural build cost."""

    if numerical_method_ids is None:
        numerical_method_ids = tuple(
            runtime_curves.loc[
                runtime_curves["family"].astype(str).eq("numerical valuation"),
                "method_id",
            ].astype(str).unique()
        )

    rows: list[dict[str, Any]] = []
    for neural_method_id, deployment_id in METHOD_DEPLOYMENT_IDS.items():
        try:
            neural = _curve_row(runtime_curves, neural_method_id, timing_mode="warm")
        except KeyError:
            continue
        scenarios = upfront_scenarios.loc[
            upfront_scenarios["deployment_id"].astype(str).eq(deployment_id)
            & upfront_scenarios["status"].astype(str).eq("complete")
        ]
        for numerical_method_id in numerical_method_ids:
            try:
                numerical = _curve_row(
                    runtime_curves,
                    numerical_method_id,
                    timing_mode="warm",
                )
            except KeyError:
                continue
            marginal_saving = _numeric(
                numerical["marginal_seconds_per_observation"]
            ) - _numeric(neural["marginal_seconds_per_observation"])
            for _, scenario in scenarios.iterrows():
                upfront = _numeric(scenario["upfront_seconds"])
                numerator = (
                    upfront
                    + _numeric(neural["fixed_seconds"])
                    - _numeric(numerical["fixed_seconds"])
                )
                if marginal_saving <= 0.0:
                    break_even = np.nan
                    status = "no_break_even_neural_not_faster_at_scale"
                else:
                    break_even = max(int(math.ceil(numerator / marginal_saving)), 1)
                    status = "complete"
                rows.append(
                    {
                        "deployment_id": deployment_id,
                        "neural_method_id": neural_method_id,
                        "neural_method": neural["method"],
                        "neural_output_scope": neural["output_scope"],
                        "numerical_method_id": numerical_method_id,
                        "numerical_method": numerical["method"],
                        "scenario_id": scenario["scenario_id"],
                        "evidence_type": scenario["evidence_type"],
                        "upfront_seconds": upfront,
                        "upfront_hours": _numeric(scenario["upfront_hours"]),
                        "label_generation_hours": _numeric(
                            scenario["label_generation_hours"]
                        ),
                        "marginal_seconds_saved_per_valuation": marginal_saving,
                        "break_even_valuations": break_even,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def default_business_workloads() -> pd.DataFrame:
    """Return practical portfolio and scenario workloads for interpretation."""

    return pd.DataFrame(
        [
            {
                "scenario": "One-off small portfolio",
                "contracts": 100,
                "scenarios_per_contract": 1,
                "runs_per_year": 1,
            },
            {
                "scenario": "Daily 10,000-contract portfolio",
                "contracts": 10_000,
                "scenarios_per_contract": 1,
                "runs_per_year": 250,
            },
            {
                "scenario": "10,000 contracts under 100 scenarios",
                "contracts": 10_000,
                "scenarios_per_contract": 100,
                "runs_per_year": 250,
            },
            {
                "scenario": "10,000 contracts under 1,000 scenarios",
                "contracts": 10_000,
                "scenarios_per_contract": 1_000,
                "runs_per_year": 250,
            },
            {
                "scenario": "Intraday 10-million-valuation grid",
                "contracts": 10_000,
                "scenarios_per_contract": 1_000,
                "runs_per_year": 1_000,
            },
        ]
    )


def build_business_case_scenarios(
    runtime_curves: pd.DataFrame,
    upfront_scenarios: pd.DataFrame,
    *,
    workloads: pd.DataFrame | None = None,
    numerical_method_id: str = "project_numba_crr",
    preferred_evidence_type: str = "measured",
) -> pd.DataFrame:
    """Translate crossover economics into recognisable business workloads."""

    scenario_table = workloads.copy() if workloads is not None else default_business_workloads()
    required = {"scenario", "contracts", "scenarios_per_contract", "runs_per_year"}
    missing = sorted(required.difference(scenario_table.columns))
    if missing:
        raise ValueError(f"Workload table is missing columns: {missing}")

    numerical = _curve_row(runtime_curves, numerical_method_id, timing_mode="warm")
    rows: list[dict[str, Any]] = []
    for neural_method_id, deployment_id in METHOD_DEPLOYMENT_IDS.items():
        try:
            neural = _curve_row(runtime_curves, neural_method_id, timing_mode="warm")
        except KeyError:
            continue
        candidates = upfront_scenarios.loc[
            upfront_scenarios["deployment_id"].astype(str).eq(deployment_id)
            & upfront_scenarios["status"].astype(str).eq("complete")
        ].copy()
        if candidates.empty:
            upfront = np.nan
            evidence_type = "missing"
            scenario_id = "missing"
        else:
            preferred = candidates.loc[
                candidates["evidence_type"].astype(str).eq(preferred_evidence_type)
            ]
            chosen = preferred.iloc[0] if not preferred.empty else candidates.iloc[0]
            upfront = _numeric(chosen["upfront_seconds"])
            evidence_type = str(chosen["evidence_type"])
            scenario_id = str(chosen["scenario_id"])

        for _, workload in scenario_table.iterrows():
            valuations_per_run = int(workload["contracts"]) * int(
                workload["scenarios_per_contract"]
            )
            runs_per_year = int(workload["runs_per_year"])
            numerical_seconds = predict_runtime_seconds(numerical, valuations_per_run)
            neural_seconds = predict_runtime_seconds(neural, valuations_per_run)
            saved_per_run = numerical_seconds - neural_seconds
            annual_saving = saved_per_run * runs_per_year
            payback_runs = (
                math.ceil(upfront / saved_per_run)
                if np.isfinite(upfront) and saved_per_run > 0.0
                else np.nan
            )
            payback_years = (
                payback_runs / runs_per_year
                if np.isfinite(payback_runs) and runs_per_year > 0
                else np.nan
            )
            rows.append(
                {
                    "scenario": workload["scenario"],
                    "contracts": int(workload["contracts"]),
                    "scenarios_per_contract": int(workload["scenarios_per_contract"]),
                    "valuations_per_run": valuations_per_run,
                    "runs_per_year": runs_per_year,
                    "neural_method_id": neural_method_id,
                    "neural_method": neural["method"],
                    "neural_output_scope": neural["output_scope"],
                    "numerical_method_id": numerical_method_id,
                    "numerical_method": numerical["method"],
                    "numerical_seconds_per_run": numerical_seconds,
                    "neural_seconds_per_run": neural_seconds,
                    "seconds_saved_per_run": saved_per_run,
                    "annual_seconds_saved": annual_saving,
                    "annual_hours_saved": annual_saving / 3600.0,
                    "upfront_scenario_id": scenario_id,
                    "upfront_evidence_type": evidence_type,
                    "upfront_seconds": upfront,
                    "payback_runs": payback_runs,
                    "payback_years": payback_years,
                    "neural_faster_for_workload": bool(saved_per_run > 0.0),
                }
            )
    return pd.DataFrame(rows)


def run_business_case_analysis(
    project_root: Path,
    runtime_scaling: pd.DataFrame,
    *,
    overrides_seconds: Mapping[str, float | None] | None = None,
    assumed_label_generation_hours: Sequence[float] = DEFAULT_LABEL_GENERATION_HOUR_SCENARIOS,
    assumed_total_build_hours: Sequence[float] = DEFAULT_TOTAL_BUILD_HOUR_SCENARIOS,
) -> dict[str, pd.DataFrame]:
    """Build curves, crossover, upfront, lifecycle, and workload evidence."""

    curves = fit_runtime_curves(runtime_scaling)
    operational = build_operational_crossover(curves, runtime_scaling)
    inventory = build_upfront_cost_inventory(
        project_root,
        overrides_seconds=overrides_seconds,
    )
    upfront = build_upfront_cost_scenarios(
        inventory,
        assumed_label_generation_hours=assumed_label_generation_hours,
        assumed_total_build_hours=assumed_total_build_hours,
    )
    lifecycle = build_lifecycle_break_even(curves, upfront)
    workloads = build_business_case_scenarios(curves, upfront)
    return {
        "runtime_curves": curves,
        "operational_crossover": operational,
        "upfront_cost_inventory": inventory,
        "upfront_cost_scenarios": upfront,
        "lifecycle_break_even": lifecycle,
        "business_case_scenarios": workloads,
    }


__all__ = [
    "DEFAULT_LABEL_GENERATION_HOUR_SCENARIOS",
    "DEFAULT_TOTAL_BUILD_HOUR_SCENARIOS",
    "METHOD_DEPLOYMENT_IDS",
    "RuntimeCurveConfig",
    "build_business_case_scenarios",
    "build_lifecycle_break_even",
    "build_operational_crossover",
    "build_upfront_cost_inventory",
    "build_upfront_cost_scenarios",
    "default_business_workloads",
    "fit_runtime_curves",
    "predict_runtime_seconds",
    "run_business_case_analysis",
]
