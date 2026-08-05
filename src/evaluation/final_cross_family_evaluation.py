"""Cross-family evidence for OOD, runtime, and Longstaff--Schwartz results.

Static surrogates, exercise classifiers, numerical trees, and path-based LSM
methods solve different tasks.  This module normalizes their exported evidence
without putting incomparable methods into one undifferentiated leaderboard.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.final_artifact_adapters import FinalNotebookPackage


OOD_MODEL_ALIASES = {
    "Direct MLP": ("direct_mlp", "Direct MLP", "04", True),
    "Non-negative premium": (
        "nonnegative_premium_mlp",
        "Non-negative premium MLP",
        "05",
        True,
    ),
    "Constrained floor residual": (
        "constrained_floor_residual_mlp",
        "Constrained floor residual MLP",
        "05",
        True,
    ),
    "Price-only constrained residual": (
        "price_only_constrained_residual_mlp",
        "Price-only constrained residual MLP",
        "06",
        True,
    ),
    "Multi-task constrained residual": (
        "multitask_constrained_residual_mlp",
        "Multi-task constrained residual MLP",
        "06",
        True,
    ),
}

RUNTIME_MODEL_ALIASES = {
    "Direct MLP": ("direct_mlp", "Direct MLP", "static neural inference"),
    "Non-negative premium": (
        "nonnegative_premium_mlp",
        "Non-negative premium MLP",
        "static neural inference",
    ),
    "Constrained floor residual": (
        "constrained_floor_residual_mlp",
        "Constrained floor residual MLP",
        "static neural inference",
    ),
    "Exercise-only classifier": (
        "exercise_only_classifier",
        "Exercise-only classifier",
        "static neural inference",
    ),
    "Price-only constrained residual": (
        "price_only_constrained_residual_mlp",
        "Price-only constrained residual MLP",
        "static neural inference",
    ),
    "Multi-task model": (
        "multitask_constrained_residual_mlp",
        "Multi-task constrained residual MLP",
        "static neural inference",
    ),
    "Notebook 08 warm-start": (
        "integrated_warm_start_model",
        "Integrated warm-start deployment model",
        "static neural inference",
    ),
    "Notebook 08 selected scratch": (
        "integrated_scratch_model",
        "Integrated balanced-scratch benchmark",
        "static neural inference",
    ),
    "CRR": ("crr", "High-resolution CRR", "numerical valuation"),
    "Classical LSM end-to-end": (
        "classical_lsm_end_to_end",
        "Classical LSM end-to-end",
        "path-based valuation",
    ),
    "Classical LSM fit and valuation": (
        "classical_lsm_fit_and_valuation",
        "Classical LSM fit and valuation",
        "path-based valuation",
    ),
    "Neural LSM end-to-end": (
        "neural_lsm_end_to_end",
        "Neural LSM end-to-end",
        "path-based valuation",
    ),
    "Neural LSM evaluation": (
        "neural_lsm_evaluation",
        "Neural LSM evaluation",
        "path-based valuation",
    ),
    "Policy path simulation": (
        "policy_path_simulation",
        "Policy path simulation",
        "simulation component",
    ),
    "Valuation path simulation": (
        "valuation_path_simulation",
        "Valuation path simulation",
        "simulation component",
    ),
}

# Runtime packages intentionally repeat some benchmark rows so each upstream
# notebook can compare its own model with earlier baselines.  Notebook 09 keeps
# one authoritative owner for every logical method instead of treating those
# repeated benchmark rows as conflicting evidence.
RUNTIME_SOURCE_OWNERS = {
    "direct_mlp": "04",
    "nonnegative_premium_mlp": "05",
    "constrained_floor_residual_mlp": "05",
    "exercise_only_classifier": "06",
    "price_only_constrained_residual_mlp": "06",
    "multitask_constrained_residual_mlp": "06",
    "integrated_warm_start_model": "08",
    "integrated_scratch_model": "08",
    "crr": "07",
    "classical_lsm_end_to_end": "07",
    "classical_lsm_fit_and_valuation": "07",
    "neural_lsm_end_to_end": "07",
    "neural_lsm_evaluation": "07",
    "policy_path_simulation": "07",
    "valuation_path_simulation": "07",
    "neural_lsm_policy_training": "07",
}



def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _normalize_ood_name(value: Any) -> str:
    name = str(value)
    for prefix in ("american_put_ood_", "ood_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in (".parquet", ".csv", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _in_domain_mae_map(static_model_metrics: pd.DataFrame) -> dict[str, float]:
    required = {"model_id", "normalized_mae"}
    if not required.issubset(static_model_metrics.columns):
        raise ValueError(
            "Phase 4 static_model_metrics must contain model_id and normalized_mae"
        )
    return {
        str(row.model_id): float(row.normalized_mae)
        for row in static_model_metrics.itertuples(index=False)
    }


def build_static_ood_comparison(
    packages: Mapping[str, FinalNotebookPackage],
    static_model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build one normalized OOD-pricing row per model and regime."""

    in_domain = _in_domain_mae_map(static_model_metrics)
    rows: list[dict[str, Any]] = []

    # Notebook 04 owns the Direct MLP OOD result.
    for record in packages["04"].final_metrics.get("ood") or []:
        rows.append(
            {
                "ood_set": _normalize_ood_name(
                    record.get("ood_set", record.get("component"))
                ),
                "model_id": "direct_mlp",
                "model": "Direct MLP",
                "source_notebook": "04",
                "observations": record.get("observations"),
                "ood_normalized_mae": record.get("mae"),
                "ood_normalized_rmse": record.get("rmse"),
                "h6_eligible": True,
            }
        )

    # Notebook 05 owns its residual candidates; duplicate Direct MLP rows are skipped.
    for record in packages["05"].final_metrics.get("ood") or []:
        alias = OOD_MODEL_ALIASES.get(str(record.get("model")))
        if alias is None or alias[0] == "direct_mlp":
            continue
        model_id, model, source, eligible = alias
        rows.append(
            {
                "ood_set": _normalize_ood_name(
                    record.get("ood_set", record.get("component"))
                ),
                "model_id": model_id,
                "model": model,
                "source_notebook": source,
                "observations": record.get("observations"),
                "ood_normalized_mae": record.get("mae"),
                "ood_normalized_rmse": record.get("rmse"),
                "h6_eligible": eligible,
            }
        )

    for record in packages["06"].final_metrics.get("ood_pricing") or []:
        alias = OOD_MODEL_ALIASES.get(str(record.get("model")))
        if alias is None:
            continue
        model_id, model, source, eligible = alias
        rows.append(
            {
                "ood_set": _normalize_ood_name(
                    record.get("ood_set", record.get("component"))
                ),
                "model_id": model_id,
                "model": model,
                "source_notebook": source,
                "observations": record.get("observations"),
                "ood_normalized_mae": record.get("mae"),
                "ood_normalized_rmse": record.get("rmse"),
                "h6_eligible": eligible,
            }
        )

    nb08 = packages["08"].final_metrics
    integrated_ood_sources = (
        (
            nb08.get("ood") or [],
            "integrated_warm_start",
            "Integrated warm-start",
            "08",
            True,
        ),
        (
            (nb08.get("scratch_benchmark") or {}).get("ood") or [],
            "integrated_scratch",
            "Integrated balanced-scratch",
            "08_scratch",
            True,
        ),
    )
    for records, id_prefix, label_prefix, source, eligible in integrated_ood_sources:
        for record in records:
            common = {
                "ood_set": _normalize_ood_name(record.get("component", record.get("ood_set"))),
                "source_notebook": source,
                "observations": record.get("observations"),
            }
            rows.extend(
                [
                    {
                        **common,
                        "model_id": f"{id_prefix}_constrained_price",
                        "model": f"{label_prefix} constrained price",
                        "ood_normalized_mae": record.get("constrained_mae"),
                        "ood_normalized_rmse": record.get("constrained_rmse"),
                        "h6_eligible": eligible,
                    },
                    {
                        **common,
                        "model_id": f"{id_prefix}_direct_price_head",
                        "model": f"{label_prefix} direct price head",
                        "ood_normalized_mae": record.get("direct_mae"),
                        "ood_normalized_rmse": record.get("direct_rmse"),
                        "h6_eligible": False,
                    },
                ]
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if table[["model_id", "ood_set"]].duplicated().any():
        duplicates = table.loc[
            table[["model_id", "ood_set"]].duplicated(keep=False),
            ["model_id", "ood_set", "source_notebook"],
        ]
        raise ValueError(
            "Duplicate static OOD ownership:\n" + duplicates.to_string(index=False)
        )

    table["in_domain_normalized_mae"] = table["model_id"].map(in_domain)
    for column in (
        "ood_normalized_mae",
        "ood_normalized_rmse",
        "in_domain_normalized_mae",
        "observations",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["ood_normalized_mae", "in_domain_normalized_mae"]].isna().any().any():
        invalid = table.loc[
            table[["ood_normalized_mae", "in_domain_normalized_mae"]]
            .isna()
            .any(axis=1),
            ["model_id", "ood_set"],
        ]
        raise ValueError("Incomplete OOD pricing evidence:\n" + invalid.to_string(index=False))
    table["ood_to_in_domain_mae_ratio"] = (
        table["ood_normalized_mae"] / table["in_domain_normalized_mae"]
    )
    table["relative_ood_deterioration"] = (
        table["ood_to_in_domain_mae_ratio"] - 1.0
    )
    return table.sort_values(["model_id", "ood_set"]).reset_index(drop=True)


def build_static_ood_model_summary(
    static_ood_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate OOD error separately for every eligible static pricing model."""

    if static_ood_comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_id, group in static_ood_comparison.groupby("model_id", sort=True):
        weights = pd.to_numeric(group["observations"], errors="coerce")
        values = pd.to_numeric(group["ood_normalized_mae"], errors="coerce")
        if weights.notna().all() and float(weights.sum()) > 0.0:
            aggregate = float(np.average(values, weights=weights))
        else:
            aggregate = float(values.mean())
        in_domain = float(group["in_domain_normalized_mae"].iloc[0])
        ratio = aggregate / in_domain if in_domain > 0.0 else np.inf
        rows.append(
            {
                "model_id": model_id,
                "model": group["model"].iloc[0],
                "source_notebook": group["source_notebook"].iloc[0],
                "h6_eligible": bool(group["h6_eligible"].iloc[0]),
                "regimes": int(group["ood_set"].nunique()),
                "total_ood_observations": int(weights.fillna(0).sum()),
                "in_domain_normalized_mae": in_domain,
                "aggregate_ood_normalized_mae": aggregate,
                "aggregate_ood_to_in_domain_ratio": float(ratio),
                "minimum_regime_ratio": float(
                    group["ood_to_in_domain_mae_ratio"].min()
                ),
                "maximum_regime_ratio": float(
                    group["ood_to_in_domain_mae_ratio"].max()
                ),
                "regimes_with_higher_error": int(
                    (group["ood_to_in_domain_mae_ratio"] > 1.0).sum()
                ),
                "regimes_with_material_deterioration": int(
                    (group["ood_to_in_domain_mae_ratio"] >= 1.25).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["h6_eligible", "aggregate_ood_to_in_domain_ratio"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_lsm_evidence(
    packages: Mapping[str, FinalNotebookPackage],
) -> dict[str, pd.DataFrame]:
    """Normalize Notebook 07 held-out, OOD, policy, bound, and runtime evidence."""

    payload = packages["07"].final_metrics
    heldout = pd.DataFrame(payload.get("heldout_pricing") or [])
    if not heldout.empty:
        heldout.insert(0, "source_notebook", "07")
    ood = pd.DataFrame(payload.get("ood_pricing") or [])
    if not ood.empty:
        ood.insert(0, "source_notebook", "07")
        if "ood_set" in ood.columns:
            ood["ood_set"] = ood["ood_set"].map(_normalize_ood_name)
    bounds = pd.DataFrame(payload.get("financial_bounds") or [])
    if not bounds.empty:
        bounds.insert(0, "source_notebook", "07")
    policy = pd.DataFrame(payload.get("policy_summary") or [])
    if not policy.empty:
        policy.insert(0, "source_notebook", "07")
    runtime = pd.DataFrame(payload.get("runtime") or [])
    if not runtime.empty:
        runtime.insert(0, "source_notebook", "07")

    coverage_payload = payload.get("coverage") or {}
    coverage = pd.DataFrame(
        [
            {"source_notebook": "07", "metric": str(key), "coverage": value}
            for key, value in coverage_payload.items()
        ]
    )
    break_even_payload = payload.get("runtime_break_even") or {}
    break_even = pd.DataFrame(
        [
            {"source_notebook": "07", "metric": str(key), "value": value}
            for key, value in break_even_payload.items()
        ]
    )
    return {
        "lsm_heldout_pricing": heldout,
        "lsm_ood_pricing": ood,
        "lsm_financial_bounds": bounds,
        "lsm_policy_summary": policy,
        "lsm_runtime_source": runtime,
        "lsm_coverage": coverage,
        "lsm_runtime_break_even": break_even,
    }


def _largest_batch_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the largest available batch for each model name."""

    frame = pd.DataFrame(records)
    if frame.empty or "model" not in frame.columns:
        return []
    frame["observations"] = pd.to_numeric(frame.get("observations"), errors="coerce")
    frame = frame.sort_values("observations", ascending=False, na_position="last")
    return frame.drop_duplicates("model", keep="first").to_dict(orient="records")


def build_runtime_comparison(
    packages: Mapping[str, FinalNotebookPackage],
) -> pd.DataFrame:
    """Build separate static, numerical, path-based, simulation, and training rows."""

    rows: list[dict[str, Any]] = []

    for notebook, key in (("04", "runtime"), ("05", "runtime"), ("06", "inference")):
        records = packages[notebook].final_metrics.get(key) or []
        for record in _largest_batch_records(records):
            alias = RUNTIME_MODEL_ALIASES.get(str(record.get("model")))
            if alias is None:
                continue
            method_id, method, family = alias
            if RUNTIME_SOURCE_OWNERS.get(method_id) != notebook:
                # This row is a repeated benchmark owned by an earlier notebook.
                continue
            observations = _numeric(record.get("observations"))
            median = _numeric(record.get("median_seconds", record.get("seconds")))
            per_observation = _numeric(record.get("seconds_per_observation"))
            if not np.isfinite(per_observation) and observations > 0.0:
                per_observation = median / observations
            rows.append(
                {
                    "method_id": method_id,
                    "method": method,
                    "source_notebook": notebook,
                    "benchmark_family": family,
                    "observations_per_measurement": observations,
                    "benchmark_repetitions": record.get("repeats"),
                    "median_seconds": median,
                    "seconds_per_observation": per_observation,
                    "observations_per_second": (
                        1.0 / per_observation if per_observation > 0.0 else np.nan
                    ),
                    "device": record.get("device"),
                    "cost_frequency": "repeated marginal inference",
                }
            )

    # Notebook 08 owns two integrated runtime rows: the preferred warm-start
    # deployment model and the larger balanced-scratch robustness benchmark.
    for record in packages["08"].final_metrics.get("runtime") or []:
        name = str(record.get("model"))
        alias = RUNTIME_MODEL_ALIASES.get(name)
        if alias is None or RUNTIME_SOURCE_OWNERS.get(alias[0]) != "08":
            continue
        observations = _numeric(record.get("observations"))
        median = _numeric(record.get("median_seconds"))
        per_observation = _numeric(record.get("seconds_per_observation"))
        if not np.isfinite(per_observation) and observations > 0.0:
            per_observation = median / observations
        rows.append(
            {
                "method_id": alias[0],
                "method": alias[1],
                "source_notebook": "08",
                "benchmark_family": alias[2],
                "observations_per_measurement": observations,
                "benchmark_repetitions": record.get("repeats"),
                "median_seconds": median,
                "seconds_per_observation": per_observation,
                "observations_per_second": 1.0 / per_observation if per_observation > 0.0 else np.nan,
                "device": record.get("device"),
                "cost_frequency": "repeated marginal inference",
            }
        )

    nb07 = packages["07"].final_metrics
    for record in nb07.get("runtime") or []:
        alias = RUNTIME_MODEL_ALIASES.get(str(record.get("method")))
        if alias is None:
            continue
        if RUNTIME_SOURCE_OWNERS.get(alias[0]) != "07":
            continue
        median = _numeric(record.get("median", record.get("median_seconds")))
        rows.append(
            {
                "method_id": alias[0],
                "method": alias[1],
                "source_notebook": "07",
                "benchmark_family": alias[2],
                "observations_per_measurement": 1.0,
                "benchmark_repetitions": record.get("count"),
                "median_seconds": median,
                "seconds_per_observation": median,
                "observations_per_second": 1.0 / median if median > 0.0 else np.nan,
                "device": "cpu",
                "cost_frequency": "repeated per-contract valuation",
            }
        )

    training_seconds = _numeric((nb07.get("training") or {}).get("runtime_seconds"))
    if np.isfinite(training_seconds):
        rows.append(
            {
                "method_id": "neural_lsm_policy_training",
                "method": "Neural LSM policy training",
                "source_notebook": "07",
                "benchmark_family": "up-front training",
                "observations_per_measurement": np.nan,
                "benchmark_repetitions": 1,
                "median_seconds": training_seconds,
                "seconds_per_observation": np.nan,
                "observations_per_second": np.nan,
                "device": "cpu",
                "cost_frequency": "one-time per trained contract domain",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    duplicates = table["method_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_rows = table.loc[
            duplicates,
            ["method_id", "source_notebook", "benchmark_family"],
        ]
        raise ValueError(
            "Duplicate authoritative runtime rows remain after source-ownership "
            "filtering:\n" + duplicate_rows.to_string(index=False)
        )
    return table.sort_values(
        ["benchmark_family", "seconds_per_observation", "method"],
        na_position="last",
    ).reset_index(drop=True)


def run_phase_5_cross_family_evaluation(
    packages: Mapping[str, FinalNotebookPackage],
    static_model_metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build OOD, LSM, and runtime evidence without mixing task families."""

    static_ood = build_static_ood_comparison(packages, static_model_metrics)
    results = {
        "static_ood_comparison": static_ood,
        "static_ood_model_summary": build_static_ood_model_summary(static_ood),
        "runtime_comparison": build_runtime_comparison(packages),
    }
    results.update(build_lsm_evidence(packages))
    return results


def assert_cross_family_evidence_ready(results: Mapping[str, pd.DataFrame]) -> None:
    """Validate the Phase 5 OOD, LSM, and runtime evidence package."""

    required_nonempty = (
        "static_ood_comparison",
        "static_ood_model_summary",
        "runtime_comparison",
        "lsm_heldout_pricing",
        "lsm_ood_pricing",
    )
    missing = [name for name in required_nonempty if name not in results]
    if missing:
        raise RuntimeError(f"Missing cross-family evidence tables: {missing}")
    for name in required_nonempty:
        table = results[name]
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise RuntimeError(f"Cross-family evidence table {name!r} is empty")

    ood = results["static_ood_comparison"]
    eligible = ood.loc[ood["h6_eligible"]]
    expected_regimes = {
        "high_volatility",
        "extreme_moneyness",
        "long_maturity",
        "rate_dividend",
    }
    for model_id, group in eligible.groupby("model_id"):
        regimes = set(group["ood_set"])
        if regimes != expected_regimes:
            raise RuntimeError(
                f"H6 model {model_id} has regimes {sorted(regimes)}, "
                f"expected {sorted(expected_regimes)}"
            )

    runtime = results["runtime_comparison"]
    required_runtime = {"constrained_floor_residual_mlp", "crr"}
    if not required_runtime.issubset(set(runtime["method_id"])):
        raise RuntimeError(
            "Runtime evidence lacks selected static pricer or CRR benchmark"
        )


__all__ = [
    "RUNTIME_SOURCE_OWNERS",
    "assert_cross_family_evidence_ready",
    "build_lsm_evidence",
    "build_runtime_comparison",
    "build_static_ood_comparison",
    "build_static_ood_model_summary",
    "run_phase_5_cross_family_evaluation",
]
