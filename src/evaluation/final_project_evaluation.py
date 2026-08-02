"""Explicit cross-model result adapters and aggregation for Notebook 09."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.artifact_registry import (
    default_artifact_registry,
    load_registered_artifact,
    resolve_artifact_path,
)


STATIC_METRIC_COLUMNS = (
    "observations",
    "mae",
    "rmse",
    "median_absolute_error",
    "max_absolute_error",
    "mean_error",
    "r2",
    "within_0.001",
    "within_0.005",
    "within_0.01",
    "within_0.05",
)


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested mappings and lists using stable dotted paths."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, prefix=name))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                item_name = f"{name}[{index}]"
                if isinstance(item, Mapping):
                    result.update(flatten_mapping(item, prefix=item_name))
                else:
                    result[item_name] = item
        else:
            result[name] = value
    return result


def metric_inventory_from_json_files(
    named_paths: Mapping[str, Path],
) -> pd.DataFrame:
    """Build a long-form metric inventory from available JSON files."""

    rows: list[dict[str, Any]] = []
    for source_name, path in named_paths.items():
        path = Path(path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            flattened = flatten_mapping(payload)
        else:
            flattened = {"payload": payload}
        for metric, value in flattened.items():
            rows.append(
                {
                    "source": source_name,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=["source", "metric", "value"])


def build_metric_inventory(project_root: Path) -> pd.DataFrame:
    """Inventory canonical JSON packages only; exclude Notebook 09 outputs."""

    paths: dict[str, Path] = {}
    for spec in default_artifact_registry():
        path = resolve_artifact_path(project_root, spec)
        if path is not None and path.suffix.lower() == ".json":
            paths[spec.name] = path
    return metric_inventory_from_json_files(paths)


def _frame(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, pd.Series):
        return records.to_frame().T
    if isinstance(records, Mapping):
        return pd.DataFrame([records])
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise TypeError(f"Cannot convert {type(records).__name__} to DataFrame")


def _first_column_to_model(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    if "model" not in result.columns:
        candidate = result.columns[0]
        if candidate not in STATIC_METRIC_COLUMNS:
            result = result.rename(columns={candidate: "model"})
    return result


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


MODEL_SOURCE_OWNERS = {
    "Black–Scholes proxy": "04",
    "Direct MLP": "04",
    "Zero premium": "05",
    "Mean premium": "05",
    "Unconstrained premium": "05",
    "Non-negative premium": "05",
    "Constrained floor residual": "05",
    "Price-only constrained residual": "06",
    "Multi-task constrained residual": "06",
    "Final integrated constrained price": "08",
    "Final integrated direct head": "08",
}

MODEL_NAME_ALIASES = {
    "Multi-task model": "Multi-task constrained residual",
    "Multi-task price": "Multi-task constrained residual",
    "Final integrated exercise head": "Final integrated constrained price",
    "Constrained residual": "Final integrated constrained price",
    "Direct price": "Final integrated direct head",
}

CONSISTENCY_COLUMN_ALIASES = {
    "negative_price_count": "negative_price_violations",
    "negative_price_rate": "negative_price_violation_rate",
    "below_intrinsic_count": "below_intrinsic_violations",
    "below_intrinsic_rate": "below_intrinsic_violation_rate",
    "below_european_count": "below_european_violations",
    "below_european_rate": "below_european_violation_rate",
    "below_financial_floor_count": "below_financial_floor_violations",
    "below_financial_floor_rate": "below_financial_floor_violation_rate",
}

CANONICAL_CONSISTENCY_COLUMNS = (
    "model",
    "source_notebook",
    "observations",
    "negative_price_violations",
    "negative_price_violation_rate",
    "below_intrinsic_violations",
    "below_intrinsic_violation_rate",
    "below_european_violations",
    "below_european_violation_rate",
    "below_financial_floor_violations",
    "below_financial_floor_violation_rate",
)


def _source_string(value: Any) -> str:
    text = str(value)
    return text.zfill(2) if text.isdigit() else text


def _canonical_model_name(value: Any) -> str:
    model = str(value)
    return MODEL_NAME_ALIASES.get(model, model)


def _filter_owned_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the authoritative notebook for models repeated as benchmarks."""

    if table.empty or not {"model", "source_notebook"}.issubset(table.columns):
        return table
    result = table.copy()
    result["source_notebook"] = result["source_notebook"].map(_source_string)
    owner = result["model"].map(MODEL_SOURCE_OWNERS)
    result = result.loc[owner.isna() | owner.eq(result["source_notebook"])].copy()
    return result


def _deduplicate_with_source_ownership(
    table: pd.DataFrame,
    *,
    keys: list[str],
) -> pd.DataFrame:
    if table.empty:
        return table
    result = _filter_owned_rows(table)
    result["_source_rank"] = (
        result["source_notebook"]
        .map({"04": 0, "05": 1, "06": 2, "07": 3, "08": 4})
        .fillna(99)
    )
    result = (
        result.sort_values("_source_rank")
        .drop_duplicates(subset=keys, keep="first")
        .drop(columns="_source_rank")
    )
    return result


def _normalize_regime_name(value: Any) -> str:
    name = str(value)
    for prefix in ("american_put_ood_", "ood_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in (".parquet", ".csv", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _canonical_consistency_record(
    record: Mapping[str, Any],
    *,
    model: str,
    source_notebook: str,
    observations: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": _canonical_model_name(model),
        "source_notebook": _source_string(source_notebook),
        "observations": observations,
    }
    for key, value in record.items():
        canonical_key = CONSISTENCY_COLUMN_ALIASES.get(str(key), str(key))
        if canonical_key in {"model", "source_notebook"}:
            continue
        if canonical_key == "observations" and observations is not None:
            continue
        result[canonical_key] = value
    return result


def _model_observations(records: Any) -> dict[str, Any]:
    frame = _first_column_to_model(_frame(records))
    if frame.empty or not {"model", "observations"}.issubset(frame.columns):
        return {}
    return {
        _canonical_model_name(record["model"]): record.get("observations")
        for record in frame.to_dict(orient="records")
    }


def _maximum_violation_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    row = frame.iloc[0]
    rate_columns = [
        column
        for column in frame.columns
        if column.endswith("_violation_rate")
    ]
    values = [_numeric(row.get(column)) for column in rate_columns]
    finite = [value for value in values if np.isfinite(value)]
    if finite:
        return max(finite)
    observations = _numeric(row.get("observations"))
    total = _numeric(row.get("total_bound_violations"))
    if observations > 0 and np.isfinite(total):
        return total / observations
    return float("nan")


def load_project_results(project_root: Path) -> dict[str, Any]:
    """Load canonical Notebook 04-08 result packages."""

    root = Path(project_root)
    results = {
        "direct": load_registered_artifact(root, "direct_final_metrics"),
        "premium": load_registered_artifact(root, "premium_final_metrics"),
        "multitask": load_registered_artifact(root, "multitask_final_metrics"),
        "lsm": load_registered_artifact(root, "lsm_final_metrics"),
        "integrated": {
            "test_metrics": load_registered_artifact(root, "integrated_test_metrics"),
            "pricing_metrics": load_registered_artifact(root, "integrated_pricing_metrics"),
            "exercise_metrics": load_registered_artifact(root, "integrated_exercise_metrics"),
            "continuation_metrics": load_registered_artifact(
                root, "integrated_continuation_metrics"
            ),
            "consistency_metrics": load_registered_artifact(
                root, "integrated_consistency_metrics"
            ),
            "boundary_analysis": load_registered_artifact(
                root, "integrated_boundary_analysis"
            ),
            "ood_metrics": load_registered_artifact(root, "integrated_ood_metrics"),
            "runtime": load_registered_artifact(root, "integrated_runtime"),
        },
    }
    return results


def _append_metric_rows(
    rows: list[dict[str, Any]],
    records: Any,
    *,
    source_notebook: str,
    model_aliases: Mapping[str, str] | None = None,
) -> None:
    frame = _first_column_to_model(_frame(records))
    if frame.empty:
        return
    aliases = dict(model_aliases or {})
    if "model" not in frame.columns:
        return
    for record in frame.to_dict(orient="records"):
        model = _canonical_model_name(record.pop("model"))
        rows.append(
            {
                "model": aliases.get(model, model),
                "source_notebook": source_notebook,
                **record,
            }
        )


def build_static_pricing_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """Build one in-domain table with explicit source ownership."""

    rows: list[dict[str, Any]] = []

    direct = results.get("direct") or {}
    for key, label in (
        ("black_scholes_proxy", "Black–Scholes proxy"),
        ("direct_mlp", "Direct MLP"),
    ):
        metrics = (direct.get("pricing") or {}).get(key)
        if isinstance(metrics, Mapping):
            rows.append(
                {
                    "model": label,
                    "source_notebook": "04",
                    **metrics,
                }
            )

    premium = results.get("premium") or {}
    premium_frame = _first_column_to_model(_frame(premium.get("pricing")))
    if not premium_frame.empty and "model" in premium_frame.columns:
        for record in premium_frame.to_dict(orient="records"):
            model = _canonical_model_name(record.pop("model"))
            if model == "Direct MLP":
                continue
            rows.append(
                {
                    "model": model,
                    "source_notebook": "05",
                    **record,
                }
            )

    multitask = results.get("multitask") or {}
    _append_metric_rows(
        rows,
        multitask.get("pricing"),
        source_notebook="06",
        model_aliases=MODEL_NAME_ALIASES,
    )

    integrated = results.get("integrated") or {}
    pricing = _first_column_to_model(_frame(integrated.get("pricing_metrics")))
    if not pricing.empty and "model" in pricing.columns:
        for record in pricing.to_dict(orient="records"):
            model = _canonical_model_name(record.pop("model"))
            rows.append(
                {
                    "model": model,
                    "source_notebook": "08",
                    **record,
                }
            )
    elif isinstance(integrated.get("test_metrics"), Mapping):
        metrics = integrated["test_metrics"]
        rows.extend(
            [
                {
                    "model": "Final integrated constrained price",
                    "source_notebook": "08",
                    "mae": metrics.get("constrained_mae"),
                    "rmse": metrics.get("constrained_rmse"),
                },
                {
                    "model": "Final integrated direct head",
                    "source_notebook": "08",
                    "mae": metrics.get("direct_mae"),
                    "rmse": metrics.get("direct_rmse"),
                },
            ]
        )

    if not rows:
        return pd.DataFrame(columns=("model", "source_notebook", *STATIC_METRIC_COLUMNS))

    table = pd.DataFrame(rows)
    table["model"] = table["model"].map(_canonical_model_name)
    table = _deduplicate_with_source_ownership(table, keys=["model"])
    ordered = ["model", "source_notebook"] + [
        column for column in STATIC_METRIC_COLUMNS if column in table.columns
    ]
    extras = [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered + extras].sort_values(
        ["mae", "model"], na_position="last"
    ).reset_index(drop=True)


def _wide_consistency_from_checks(
    frame: pd.DataFrame,
    *,
    model: str,
    source_notebook: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "source_notebook": source_notebook,
    }
    if {"check", "violations"}.issubset(frame.columns):
        for record in frame.to_dict(orient="records"):
            check = str(record["check"])
            row[f"{check}_violations"] = record.get("violations")
            row[f"{check}_violation_rate"] = record.get("violation_rate")
    return row


def build_financial_consistency_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """Normalize financial checks from Notebooks 04, 05, 06, and 08."""

    rows: list[dict[str, Any]] = []

    direct = results.get("direct") or {}
    direct_observations = (
        (direct.get("pricing") or {}).get("direct_mlp", {}).get("observations")
    )
    direct_checks = _frame(direct.get("financial_consistency"))
    if not direct_checks.empty:
        wide = _wide_consistency_from_checks(
            direct_checks,
            model="Direct MLP",
            source_notebook="04",
        )
        rows.append(
            _canonical_consistency_record(
                wide,
                model="Direct MLP",
                source_notebook="04",
                observations=direct_observations,
            )
        )

    premium = results.get("premium") or {}
    premium_observations = _model_observations(premium.get("pricing"))
    premium_frame = _first_column_to_model(
        _frame(premium.get("financial_consistency"))
    )
    if not premium_frame.empty and "model" in premium_frame.columns:
        for record in premium_frame.to_dict(orient="records"):
            model = _canonical_model_name(record.pop("model"))
            if model == "Direct MLP":
                continue
            rows.append(
                _canonical_consistency_record(
                    record,
                    model=model,
                    source_notebook="05",
                    observations=premium_observations.get(model),
                )
            )

    multitask = results.get("multitask") or {}
    multitask_frame = _first_column_to_model(
        _frame(multitask.get("financial_consistency"))
    )
    if not multitask_frame.empty and "model" in multitask_frame.columns:
        for record in multitask_frame.to_dict(orient="records"):
            model = _canonical_model_name(record.pop("model"))
            rows.append(
                _canonical_consistency_record(
                    record,
                    model=model,
                    source_notebook="06",
                    observations=record.get("observations"),
                )
            )

    integrated = results.get("integrated") or {}
    consistency = integrated.get("consistency_metrics")
    if isinstance(consistency, Mapping):
        boundary = _frame(integrated.get("boundary_analysis"))
        observations = (
            pd.to_numeric(boundary.get("observations"), errors="coerce").sum()
            if not boundary.empty and "observations" in boundary.columns
            else None
        )
        constrained_record = {
            "negative_price_violation_rate": consistency.get(
                "constrained_negative_rate"
            ),
            "below_european_violation_rate": consistency.get(
                "constrained_below_european_rate"
            ),
            "below_intrinsic_violation_rate": consistency.get(
                "constrained_below_intrinsic_rate"
            ),
            "any_contradiction_rate": consistency.get(
                "any_contradiction_rate"
            ),
            "exercise_probability_rmse": consistency.get(
                "exercise_probability_rmse"
            ),
            "decision_disagreement_rate": consistency.get(
                "decision_disagreement_rate"
            ),
            "residual_reconstruction_mae": consistency.get(
                "residual_reconstruction_mae"
            ),
        }
        direct_record = {
            "negative_price_violation_rate": consistency.get(
                "direct_negative_rate"
            ),
            "below_european_violation_rate": consistency.get(
                "direct_below_european_rate"
            ),
            "below_intrinsic_violation_rate": consistency.get(
                "direct_below_intrinsic_rate"
            ),
        }
        rows.extend(
            [
                _canonical_consistency_record(
                    constrained_record,
                    model="Final integrated constrained price",
                    source_notebook="08",
                    observations=observations,
                ),
                _canonical_consistency_record(
                    direct_record,
                    model="Final integrated direct head",
                    source_notebook="08",
                    observations=observations,
                ),
            ]
        )

    if not rows:
        return pd.DataFrame(columns=CANONICAL_CONSISTENCY_COLUMNS)

    table = pd.DataFrame(rows)
    table["model"] = table["model"].map(_canonical_model_name)
    table = _deduplicate_with_source_ownership(table, keys=["model"])
    for column in CANONICAL_CONSISTENCY_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    ordered = list(CANONICAL_CONSISTENCY_COLUMNS)
    extras = [column for column in table.columns if column not in ordered]
    return table.loc[:, ordered + extras].reset_index(drop=True)


def build_boundary_comparison(results: Mapping[str, Any]) -> pd.DataFrame:
    """Build exercise-classification and near-boundary diagnostics."""

    rows: list[dict[str, Any]] = []
    multitask = results.get("multitask") or {}

    classification = _first_column_to_model(_frame(multitask.get("classification")))
    location = _first_column_to_model(_frame(multitask.get("boundary_location")))
    location_map: dict[str, dict[str, Any]] = {}
    if not location.empty and "model" in location.columns:
        location_map = {
            _canonical_model_name(record["model"]): record
            for record in location.to_dict(orient="records")
        }

    if not classification.empty and "model" in classification.columns:
        for record in classification.to_dict(orient="records"):
            model = _canonical_model_name(record.pop("model"))
            location_record = location_map.get(model, {})
            rows.append(
                {
                    "model": model,
                    "source_notebook": "06",
                    **record,
                    "boundary_location_error": location_record.get(
                        "boundary_mae",
                        location_record.get(
                            "mean_absolute_error",
                            location_record.get("boundary_location_error"),
                        ),
                    ),
                }
            )

    integrated = results.get("integrated") or {}
    exercise = integrated.get("exercise_metrics")
    boundary = _frame(integrated.get("boundary_analysis"))
    integrated_row: dict[str, Any] = {
        "model": "Final integrated constrained price",
        "source_notebook": "08",
    }
    if isinstance(exercise, Mapping):
        integrated_row.update(exercise)
    if not boundary.empty:
        nearest = boundary.loc[
            boundary["boundary_band"].astype(str).eq("≤0.001")
        ]
        if nearest.empty:
            nearest = boundary.head(1)
        if not nearest.empty:
            record = nearest.iloc[0].to_dict()
            integrated_row.update(
                {
                    "near_boundary_observations": record.get("observations"),
                    "near_boundary_price_mae": record.get("price_mae"),
                    "near_boundary_exercise_accuracy": record.get(
                        "exercise_accuracy"
                    ),
                    "near_boundary_exercise_f1": record.get("exercise_f1"),
                    "near_boundary_balanced_accuracy": record.get(
                        "balanced_accuracy"
                    ),
                }
            )
    if len(integrated_row) > 2:
        rows.append(integrated_row)

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table["model"] = table["model"].map(_canonical_model_name)
    return _deduplicate_with_source_ownership(table, keys=["model"]).reset_index(
        drop=True
    )


def _pricing_mae_map(static_pricing: pd.DataFrame) -> dict[str, float]:
    if static_pricing.empty or not {"model", "mae"}.issubset(static_pricing.columns):
        return {}
    return {
        str(record["model"]): _numeric(record["mae"])
        for record in static_pricing.to_dict(orient="records")
    }


def build_ood_comparison(
    results: Mapping[str, Any],
    static_pricing: pd.DataFrame,
) -> pd.DataFrame:
    """Build one authoritative row per static model and normalized OOD regime."""

    rows: list[dict[str, Any]] = []
    in_domain = _pricing_mae_map(static_pricing)

    direct = results.get("direct") or {}
    for record in _frame(direct.get("ood")).to_dict(orient="records"):
        rows.append(
            {
                "model": "Direct MLP",
                "regime": _normalize_regime_name(
                    record.get("ood_set", record.get("component"))
                ),
                "observations": record.get("observations"),
                "ood_mae": record.get("mae"),
                "ood_rmse": record.get("rmse"),
                "source_notebook": "04",
            }
        )

    premium = results.get("premium") or {}
    for record in _frame(premium.get("ood")).to_dict(orient="records"):
        model = _canonical_model_name(record.get("model"))
        if model == "Direct MLP":
            continue
        rows.append(
            {
                "model": model,
                "regime": _normalize_regime_name(
                    record.get("ood_set", record.get("component"))
                ),
                "observations": record.get("observations"),
                "ood_mae": record.get("mae"),
                "ood_rmse": record.get("rmse"),
                "source_notebook": "05",
            }
        )

    multitask = results.get("multitask") or {}
    for record in _frame(multitask.get("ood")).to_dict(orient="records"):
        model = _canonical_model_name(record.get("model"))
        if pd.notna(record.get("mae")):
            rows.append(
                {
                    "model": model,
                    "regime": _normalize_regime_name(
                        record.get("ood_set", record.get("component"))
                    ),
                    "observations": record.get("observations"),
                    "ood_mae": record.get("mae"),
                    "ood_rmse": record.get("rmse"),
                    "source_notebook": "06",
                }
            )

    integrated = results.get("integrated") or {}
    for record in _frame(integrated.get("ood_metrics")).to_dict(orient="records"):
        rows.append(
            {
                "model": "Final integrated constrained price",
                "regime": _normalize_regime_name(
                    record.get("component", record.get("ood_set"))
                ),
                "observations": record.get("observations"),
                "ood_mae": record.get("constrained_mae", record.get("mae")),
                "ood_rmse": record.get("constrained_rmse", record.get("rmse")),
                "source_notebook": "08",
            }
        )

    columns = (
        "model",
        "regime",
        "observations",
        "in_domain_mae",
        "ood_mae",
        "ood_rmse",
        "ood_deterioration",
        "source_notebook",
    )
    if not rows:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(rows)
    table["model"] = table["model"].map(_canonical_model_name)
    table["regime"] = table["regime"].map(_normalize_regime_name)
    table = _deduplicate_with_source_ownership(
        table,
        keys=["model", "regime"],
    )
    table["in_domain_mae"] = table["model"].map(in_domain)
    denominator = pd.to_numeric(table["in_domain_mae"], errors="coerce")
    table["ood_deterioration"] = (
        pd.to_numeric(table["ood_mae"], errors="coerce") - denominator
    ) / denominator
    return table.loc[:, list(columns)].sort_values(
        ["model", "regime"]
    ).reset_index(drop=True)


def build_runtime_comparison_from_results(
    results: Mapping[str, Any],
) -> pd.DataFrame:
    """Normalize runtime records and remove benchmark duplicates."""

    rows: list[dict[str, Any]] = []

    direct = results.get("direct") or {}
    for record in _frame(direct.get("runtime")).to_dict(orient="records"):
        seconds = record.get("median_seconds", record.get("seconds"))
        rows.append(
            {
                "model": "Direct MLP",
                "device": record.get("device"),
                "observations": record.get("observations"),
                "seconds": seconds,
                "source_notebook": "04",
                "cost_type": "marginal inference",
            }
        )

    premium = results.get("premium") or {}
    for record in _frame(premium.get("runtime")).to_dict(orient="records"):
        model = _canonical_model_name(record.get("model"))
        if model == "Direct MLP":
            continue
        rows.append(
            {
                "model": model,
                "device": record.get("device"),
                "observations": record.get("observations"),
                "seconds": record.get("median_seconds", record.get("seconds")),
                "source_notebook": "05",
                "cost_type": "marginal inference",
            }
        )

    integrated = results.get("integrated") or {}
    runtime = integrated.get("runtime")
    if isinstance(runtime, Mapping):
        rows.append(
            {
                "model": "Final integrated multi-head",
                "device": runtime.get("device"),
                "observations": runtime.get("observations"),
                "seconds": runtime.get("seconds"),
                "source_notebook": "08",
                "cost_type": "marginal inference",
            }
        )

    lsm = results.get("lsm") or {}
    for record in _frame(lsm.get("runtime")).to_dict(orient="records"):
        rows.append(
            {
                "model": record.get("method"),
                "device": None,
                "observations": 1,
                "benchmark_contracts": record.get("count"),
                "seconds": record.get("median", record.get("mean")),
                "source_notebook": "07",
                "cost_type": "per-contract valuation",
            }
        )
    training = (lsm.get("training") or {}).get("runtime_seconds")
    if training is not None:
        rows.append(
            {
                "model": "Neural LSM policy training",
                "device": None,
                "observations": 1,
                "seconds": training,
                "source_notebook": "07",
                "cost_type": "up-front training",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["model"] = table["model"].map(_canonical_model_name)
    table = _deduplicate_with_source_ownership(
        table,
        keys=["model", "observations", "cost_type", "device"],
    )
    table["seconds_per_observation"] = (
        pd.to_numeric(table["seconds"], errors="coerce")
        / pd.to_numeric(table["observations"], errors="coerce")
    )
    table["observations_per_second"] = 1.0 / table["seconds_per_observation"]
    return table.sort_values(
        ["cost_type", "seconds_per_observation"],
        na_position="last",
    ).reset_index(drop=True)


def build_static_ablation_table(
    static_pricing: pd.DataFrame,
    financial_consistency: pd.DataFrame,
    boundary_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Attach architectural attributes to comparable static-model outcomes."""

    architecture = {
        "Black–Scholes proxy": (False, False, False, False, False, False, False),
        "Direct MLP": (True, False, True, False, False, False, False),
        "Unconstrained premium": (False, True, False, False, False, False, False),
        "Non-negative premium": (False, True, True, False, False, False, False),
        "Constrained floor residual": (False, True, True, True, False, False, False),
        "Price-only constrained residual": (False, True, True, True, False, False, False),
        "Multi-task constrained residual": (False, True, True, True, True, False, True),
        "Final integrated constrained price": (False, True, True, True, True, True, True),
        "Final integrated direct head": (True, False, True, False, True, True, True),
    }
    rows = []
    for record in static_pricing.to_dict(orient="records"):
        model = _canonical_model_name(record["model"])
        if model not in architecture:
            continue
        (
            direct_learning,
            residual_learning,
            nonnegative_output,
            financial_floor,
            exercise_head,
            continuation_head,
            shared_backbone,
        ) = architecture[model]
        rows.append(
            {
                "model": model,
                "direct_learning": direct_learning,
                "residual_learning": residual_learning,
                "nonnegative_output": nonnegative_output,
                "financial_floor": financial_floor,
                "exercise_head": exercise_head,
                "continuation_head": continuation_head,
                "shared_backbone": shared_backbone,
                "test_mae": record.get("mae"),
                "test_rmse": record.get("rmse"),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if not financial_consistency.empty:
        consistency = financial_consistency.copy()
        consistency["model"] = consistency["model"].map(_canonical_model_name)
        rate_columns = [
            column
            for column in consistency.columns
            if column.endswith("_violation_rate")
        ]
        if rate_columns:
            consistency["reported_financial_violation_rate"] = consistency[
                rate_columns
            ].max(axis=1, skipna=True)
            table = table.merge(
                consistency[["model", "reported_financial_violation_rate"]],
                on="model",
                how="left",
                validate="one_to_one",
            )

    if not boundary_comparison.empty and {"model", "f1"}.issubset(
        boundary_comparison.columns
    ):
        boundary = boundary_comparison[["model", "f1"]].copy()
        boundary["model"] = boundary["model"].map(_canonical_model_name)
        boundary = boundary.drop_duplicates("model", keep="first")
        table = table.merge(
            boundary.rename(columns={"f1": "exercise_f1"}),
            on="model",
            how="left",
            validate="one_to_one",
        )
    return table.sort_values("test_mae").reset_index(drop=True)


def build_lsm_comparison(results: Mapping[str, Any]) -> pd.DataFrame:
    """Keep LSM results separate because they use contract-level raw prices."""

    lsm = results.get("lsm") or {}
    table = _first_column_to_model(_frame(lsm.get("heldout_pricing")))
    if table.empty:
        return table
    table = table.rename(columns={"model": "method"})

    coverage = lsm.get("coverage") or {}
    coverage_map = {
        "classical_lsm_price": coverage.get("Classical LSM 95% CI coverage"),
        "neural_lsm_price": coverage.get("Neural LSM 95% CI coverage"),
    }
    table["ci_coverage"] = table["method"].map(coverage_map)

    runtime = _frame(lsm.get("runtime"))
    if not runtime.empty and "method" in runtime.columns:
        runtime_map = {
            "classical_lsm_price": "Classical LSM",
            "neural_lsm_price": "Neural LSM evaluation",
        }
        runtime_rows = []
        for method_key, runtime_name in runtime_map.items():
            match = runtime.loc[runtime["method"].eq(runtime_name)]
            runtime_rows.append(
                {
                    "method": method_key,
                    "valuation_seconds": (
                        match["median"].iloc[0]
                        if not match.empty and "median" in match
                        else np.nan
                    ),
                }
            )
        table = table.merge(pd.DataFrame(runtime_rows), on="method", how="left")

    training_seconds = (lsm.get("training") or {}).get("runtime_seconds")
    table["training_seconds"] = np.where(
        table["method"].eq("neural_lsm_price"),
        training_seconds,
        np.nan,
    )
    return table.reset_index(drop=True)


def _lookup_metric(
    table: pd.DataFrame,
    model: str,
    metric: str,
) -> float:
    if table.empty or not {"model", metric}.issubset(table.columns):
        return float("nan")
    match = table.loc[table["model"].eq(model), metric]
    return _numeric(match.iloc[0]) if not match.empty else float("nan")


def build_hypothesis_evidence(
    results: Mapping[str, Any],
    static_pricing: pd.DataFrame,
    financial_consistency: pd.DataFrame,
    ood_comparison: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Derive the exact H1-H6 evidence fields from canonical outputs."""

    evidence: dict[str, Any] = {
        "black_scholes_mae": _lookup_metric(
            static_pricing, "Black–Scholes proxy", "mae"
        ),
        "direct_mlp_mae": _lookup_metric(static_pricing, "Direct MLP", "mae"),
        "best_residual_mae": _lookup_metric(
            static_pricing, "Constrained floor residual", "mae"
        ),
    }

    if not financial_consistency.empty and "model" in financial_consistency.columns:
        direct = financial_consistency.loc[
            financial_consistency["model"].eq("Direct MLP")
        ]
        constrained = financial_consistency.loc[
            financial_consistency["model"].eq("Constrained floor residual")
        ]

        evidence["direct_violation_rate"] = _maximum_violation_rate(direct)
        evidence["constrained_violation_rate"] = _maximum_violation_rate(
            constrained
        )

    multitask = results.get("multitask") or {}
    h4 = (multitask.get("hypothesis") or {}).get("evidence", {})
    evidence.update(
        {
            "price_only_boundary_f1": h4.get("classifier_boundary_f1"),
            "multitask_boundary_f1": h4.get("multitask_boundary_f1"),
            "price_only_boundary_error": h4.get("price_only_boundary_mae"),
            "multitask_boundary_error": h4.get("multitask_boundary_mae"),
            "required_h4_f1_gain": h4.get("required_f1_improvement", 0.02),
            "required_h4_error_improvement": h4.get(
                "required_mae_improvement", 0.0
            ),
        }
    )

    if not runtime_comparison.empty:
        crr = runtime_comparison.loc[runtime_comparison["model"].eq("CRR")]
        neural = runtime_comparison.loc[
            runtime_comparison["model"].eq("Neural LSM evaluation")
        ]
        evidence["crr_seconds_per_option"] = (
            _numeric(crr["seconds_per_observation"].iloc[0])
            if not crr.empty
            else float("nan")
        )
        evidence["neural_seconds_per_option"] = (
            _numeric(neural["seconds_per_observation"].iloc[0])
            if not neural.empty
            else float("nan")
        )

    preferred_model = "Final integrated constrained price"
    in_domain_mae = _lookup_metric(static_pricing, preferred_model, "mae")
    if not np.isfinite(in_domain_mae):
        preferred_model = "Constrained floor residual"
        in_domain_mae = _lookup_metric(static_pricing, preferred_model, "mae")
    evidence["h6_model"] = preferred_model
    evidence["in_domain_mae"] = in_domain_mae

    ood = ood_comparison.loc[ood_comparison["model"].eq(preferred_model)]
    if not ood.empty:
        weights = pd.to_numeric(ood["observations"], errors="coerce")
        values = pd.to_numeric(ood["ood_mae"], errors="coerce")
        valid = values.notna()
        if valid.any():
            if weights[valid].notna().all() and float(weights[valid].sum()) > 0:
                aggregate = float(np.average(values[valid], weights=weights[valid]))
            else:
                aggregate = float(values[valid].mean())
            evidence["aggregate_ood_mae"] = aggregate

    return evidence



def validate_final_evaluation_semantics(
    static_pricing: pd.DataFrame,
    financial_consistency: pd.DataFrame,
    ood_comparison: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    static_ablation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate the evidence layer after file-level artifact validation."""

    checks: list[dict[str, Any]] = []

    def add(name: str, valid: bool, details: str) -> None:
        checks.append(
            {
                "check": name,
                "valid": bool(valid),
                "details": details,
            }
        )

    direct = static_pricing.loc[
        static_pricing.get("model", pd.Series(dtype=str)).eq("Direct MLP")
    ]
    direct_source = (
        _source_string(direct["source_notebook"].iloc[0])
        if not direct.empty and "source_notebook" in direct.columns
        else ""
    )
    add(
        "direct_model_uses_notebook_04",
        direct_source == "04",
        f"resolved source={direct_source or 'missing'}",
    )

    direct_consistency = financial_consistency.loc[
        financial_consistency.get("model", pd.Series(dtype=str)).eq("Direct MLP")
    ]
    constrained_consistency = financial_consistency.loc[
        financial_consistency.get("model", pd.Series(dtype=str)).eq(
            "Constrained floor residual"
        )
    ]
    direct_rate = _maximum_violation_rate(direct_consistency)
    constrained_rate = _maximum_violation_rate(constrained_consistency)
    add(
        "direct_violation_rate_available",
        np.isfinite(direct_rate),
        f"rate={direct_rate}",
    )
    add(
        "constrained_violation_rate_available",
        np.isfinite(constrained_rate),
        f"rate={constrained_rate}",
    )

    ood_duplicates = (
        int(ood_comparison.duplicated(["model", "regime"]).sum())
        if not ood_comparison.empty
        and {"model", "regime"}.issubset(ood_comparison.columns)
        else 0
    )
    add(
        "ood_model_regime_rows_are_unique",
        ood_duplicates == 0,
        f"duplicate_rows={ood_duplicates}",
    )
    prefixed_regimes = (
        ood_comparison["regime"]
        .astype(str)
        .str.startswith(("american_put_ood_", "ood_"))
        .sum()
        if not ood_comparison.empty and "regime" in ood_comparison.columns
        else 0
    )
    add(
        "ood_regime_names_are_normalized",
        int(prefixed_regimes) == 0,
        f"prefixed_rows={int(prefixed_regimes)}",
    )

    runtime_keys = ["model", "observations", "cost_type", "device"]
    runtime_duplicates = (
        int(runtime_comparison.duplicated(runtime_keys).sum())
        if not runtime_comparison.empty
        and set(runtime_keys).issubset(runtime_comparison.columns)
        else 0
    )
    add(
        "runtime_benchmarks_are_unique",
        runtime_duplicates == 0,
        f"duplicate_rows={runtime_duplicates}",
    )

    if static_ablation is not None and not static_ablation.empty:
        for model in (
            "Multi-task constrained residual",
            "Final integrated constrained price",
        ):
            match = static_ablation.loc[static_ablation["model"].eq(model)]
            value = (
                _numeric(match["exercise_f1"].iloc[0])
                if not match.empty and "exercise_f1" in match.columns
                else float("nan")
            )
            add(
                f"{model}_exercise_f1_available",
                np.isfinite(value),
                f"f1={value}",
            )

    return pd.DataFrame(checks)



def build_literature_handoff(
    hypothesis_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Create a transparent citation-mapping handoff without inventing references."""

    if hypothesis_decisions.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "project_finding": hypothesis_decisions.apply(
                lambda row: f"{row['hypothesis']}: {row['decision']}",
                axis=1,
            ),
            "primary_evidence": hypothesis_decisions["primary_evidence"],
            "citation_key": "",
            "citation_status": "Manual mapping required from supplied papers",
            "scope_limitation": hypothesis_decisions["limitation"],
        }
    )


# Backward-compatible generic helpers used elsewhere in the repository.

def align_prediction_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    id_column: str,
) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("frames cannot be empty")
    aligned: dict[str, pd.DataFrame] = {}
    reference: np.ndarray | None = None
    for name, frame in frames.items():
        if id_column not in frame:
            raise ValueError(f"Frame {name!r} lacks {id_column!r}")
        ordered = frame.sort_values(id_column).reset_index(drop=True)
        identifiers = ordered[id_column].to_numpy()
        if pd.Series(identifiers).duplicated().any():
            raise ValueError(f"Frame {name!r} contains duplicate identifiers")
        if reference is None:
            reference = identifiers
        elif not np.array_equal(reference, identifiers):
            raise ValueError("Prediction frames do not contain identical identifiers")
        aligned[name] = ordered
    return aligned


def build_consolidated_pricing_table(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    *,
    metric_names: tuple[str, ...] = STATIC_METRIC_COLUMNS[1:],
) -> pd.DataFrame:
    rows = []
    for model, metrics in metrics_by_model.items():
        row = {"model": model}
        for metric in metric_names:
            row[metric] = metrics.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def calculate_ood_deterioration(
    in_domain: pd.DataFrame,
    out_of_domain: pd.DataFrame,
    *,
    model_column: str = "model",
    metric_column: str = "mae",
    regime_column: str = "regime",
) -> pd.DataFrame:
    required = {model_column, metric_column}
    if not required.issubset(in_domain.columns) or not required.issubset(
        out_of_domain.columns
    ):
        raise ValueError("in-domain and OOD tables must contain model and metric columns")
    if regime_column not in out_of_domain:
        raise ValueError(f"out_of_domain frame lacks {regime_column!r}")
    base = in_domain[[model_column, metric_column]].rename(
        columns={metric_column: "in_domain_metric"}
    )
    merged = out_of_domain.merge(base, on=model_column, how="left", validate="many_to_one")
    merged["ood_deterioration"] = (
        merged[metric_column] - merged["in_domain_metric"]
    ) / merged["in_domain_metric"]
    return merged


def build_runtime_comparison(
    records: pd.DataFrame,
    *,
    model_column: str = "model",
    observations_column: str = "observations",
    seconds_column: str = "seconds",
) -> pd.DataFrame:
    required = {model_column, observations_column, seconds_column}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Runtime records are missing columns: {sorted(missing)}")
    result = records.copy()
    result["seconds_per_observation"] = (
        result[seconds_column] / result[observations_column]
    )
    result["observations_per_second"] = (
        result[observations_column] / result[seconds_column]
    )
    return result


def financial_consistency_table(
    records: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = [{"model": model, **values} for model, values in records.items()]
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


__all__ = [
    "STATIC_METRIC_COLUMNS",
    "align_prediction_frames",
    "build_boundary_comparison",
    "build_consolidated_pricing_table",
    "build_financial_consistency_table",
    "build_hypothesis_evidence",
    "build_literature_handoff",
    "build_lsm_comparison",
    "build_metric_inventory",
    "build_ood_comparison",
    "build_runtime_comparison",
    "build_runtime_comparison_from_results",
    "build_static_ablation_table",
    "build_static_pricing_table",
    "calculate_ood_deterioration",
    "financial_consistency_table",
    "flatten_mapping",
    "load_project_results",
    "metric_inventory_from_json_files",
    "validate_final_evaluation_semantics",
]
