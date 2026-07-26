"""Validation and audit utilities for synthetic American put datasets.

The functions in this module do not modify pricing labels. They expose structural,
financial, and sampling-quality issues so that dataset decisions remain auditable
before any neural network is trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .generation import ParameterRanges


REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "spot",
    "strike",
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
    "intrinsic_value",
    "continuation_value",
    "european_price",
    "raw_american_price",
    "american_price",
    "pricing_floor_adjustment",
    "early_exercise_premium",
    "normalized_european_price",
    "normalized_american_price",
    "normalized_early_exercise_premium",
    "exercise_now",
    "tree_steps",
)

PARAMETER_COLUMNS: tuple[str, ...] = (
    "moneyness",
    "time_to_maturity",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in REQUIRED_COLUMNS if column != "exercise_now"
)


@dataclass(slots=True)
class DatasetQualityReport:
    """Container holding the full dataset audit."""

    schema_checks: pd.DataFrame
    parameter_checks: pd.DataFrame
    financial_checks: pd.DataFrame
    floor_summary: pd.DataFrame
    exercise_summary: pd.DataFrame

    @property
    def passed(self) -> bool:
        """Return True when all mandatory checks pass."""

        tables = (
            self.schema_checks,
            self.parameter_checks,
            self.financial_checks,
        )
        return all(bool(table["passed"].all()) for table in tables)


def load_option_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet dataset based on its suffix."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    raise ValueError("Dataset path must end in .csv or .parquet.")


def _check_record(
    name: str,
    passed: bool,
    *,
    observed: object,
    expected: object,
    mandatory: bool = True,
) -> dict[str, object]:
    return {
        "check": name,
        "observed": observed,
        "expected": expected,
        "mandatory": mandatory,
        "passed": bool(passed),
    }


def validate_dataset_schema(
    dataset: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Validate shape, columns, values, identifiers, and duplicate parameters."""

    missing_columns = sorted(set(required_columns).difference(dataset.columns))
    records: list[dict[str, object]] = [
        _check_record(
            "Dataset is non-empty",
            not dataset.empty,
            observed=len(dataset),
            expected="> 0 rows",
        ),
        _check_record(
            "All required columns are present",
            not missing_columns,
            observed=missing_columns,
            expected="No missing columns",
        ),
    ]

    if missing_columns or dataset.empty:
        return pd.DataFrame.from_records(records)

    null_count = int(dataset.loc[:, required_columns].isna().sum().sum())
    records.append(
        _check_record(
            "Required columns contain no missing values",
            null_count == 0,
            observed=null_count,
            expected=0,
        )
    )

    numeric_columns = [c for c in NUMERIC_COLUMNS if c in dataset.columns]
    numeric_values = dataset[numeric_columns].to_numpy(dtype=np.float64)
    infinite_count = int(np.isinf(numeric_values).sum())
    records.append(
        _check_record(
            "Numeric columns contain no infinite values",
            infinite_count == 0,
            observed=infinite_count,
            expected=0,
        )
    )

    duplicate_ids = int(dataset["sample_id"].duplicated().sum())
    records.append(
        _check_record(
            "sample_id is unique",
            duplicate_ids == 0,
            observed=duplicate_ids,
            expected=0,
        )
    )

    parameter_identity = [
        "spot",
        "strike",
        "time_to_maturity",
        "risk_free_rate",
        "dividend_yield",
        "volatility",
    ]
    duplicate_parameters = int(dataset.duplicated(parameter_identity).sum())
    records.append(
        _check_record(
            "Parameter vectors are unique",
            duplicate_parameters == 0,
            observed=duplicate_parameters,
            expected=0,
            mandatory=False,
        )
    )

    valid_exercise_values = dataset["exercise_now"].isin([True, False, 0, 1])
    invalid_exercise_values = int((~valid_exercise_values).sum())
    records.append(
        _check_record(
            "exercise_now contains only boolean labels",
            invalid_exercise_values == 0,
            observed=invalid_exercise_values,
            expected=0,
        )
    )

    positive_steps = bool((dataset["tree_steps"] > 0).all())
    integer_steps = bool(
        np.allclose(
            dataset["tree_steps"].to_numpy(dtype=float),
            np.round(dataset["tree_steps"].to_numpy(dtype=float)),
        )
    )
    records.append(
        _check_record(
            "tree_steps contains positive integers",
            positive_steps and integer_steps,
            observed={
                "minimum": float(dataset["tree_steps"].min()),
                "integer_valued": integer_steps,
            },
            expected="Positive integer values",
        )
    )

    return pd.DataFrame.from_records(records)


def validate_parameter_ranges(
    dataset: pd.DataFrame,
    *,
    ranges: ParameterRanges | None = None,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Check whether all sampled parameters remain inside the intended domain."""

    domain = ranges or ParameterRanges()
    records: list[dict[str, object]] = []

    for column in PARAMETER_COLUMNS:
        if column not in dataset.columns:
            records.append(
                _check_record(
                    f"{column} is available",
                    False,
                    observed="Missing column",
                    expected="Column present",
                )
            )
            continue

        lower, upper = getattr(domain, column)
        observed_min = float(dataset[column].min())
        observed_max = float(dataset[column].max())
        passed = bool(
            (dataset[column] >= lower - tolerance).all()
            and (dataset[column] <= upper + tolerance).all()
        )
        records.append(
            _check_record(
                f"{column} remains inside generation bounds",
                passed,
                observed=(observed_min, observed_max),
                expected=(lower, upper),
            )
        )

    if {"spot", "strike", "moneyness"}.issubset(dataset.columns):
        recomputed = dataset["spot"] / dataset["strike"]
        max_difference = float(np.max(np.abs(recomputed - dataset["moneyness"])))
        records.append(
            _check_record(
                "moneyness equals spot divided by strike",
                max_difference <= 1e-12,
                observed=max_difference,
                expected="Maximum absolute difference <= 1e-12",
            )
        )

    if {"moneyness", "log_moneyness"}.issubset(dataset.columns):
        recomputed_log = np.log(dataset["moneyness"])
        max_difference = float(
            np.max(np.abs(recomputed_log - dataset["log_moneyness"]))
        )
        records.append(
            _check_record(
                "log_moneyness equals log(moneyness)",
                max_difference <= 1e-12,
                observed=max_difference,
                expected="Maximum absolute difference <= 1e-12",
            )
        )

    return pd.DataFrame.from_records(records)


def validate_financial_bounds(
    dataset: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Validate pricing identities, no-arbitrage floors, and exercise labels."""

    required = {
        "spot",
        "strike",
        "intrinsic_value",
        "continuation_value",
        "european_price",
        "raw_american_price",
        "american_price",
        "pricing_floor_adjustment",
        "early_exercise_premium",
        "normalized_european_price",
        "normalized_american_price",
        "normalized_early_exercise_premium",
        "exercise_now",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        return pd.DataFrame.from_records(
            [
                _check_record(
                    "Financial columns are available",
                    False,
                    observed=missing,
                    expected="No missing financial columns",
                )
            ]
        )

    checks: list[tuple[str, pd.Series]] = [
        ("American price is non-negative", dataset["american_price"] >= -tolerance),
        (
            "American price is not below intrinsic value",
            dataset["american_price"] + tolerance >= dataset["intrinsic_value"],
        ),
        (
            "American price is not below European price",
            dataset["american_price"] + tolerance >= dataset["european_price"],
        ),
        (
            "Early-exercise premium is non-negative",
            dataset["early_exercise_premium"] >= -tolerance,
        ),
        (
            "Pricing-floor adjustment is non-negative",
            dataset["pricing_floor_adjustment"] >= -tolerance,
        ),
        (
            "American price equals raw price plus floor adjustment",
            np.isclose(
                dataset["american_price"],
                dataset["raw_american_price"] + dataset["pricing_floor_adjustment"],
                atol=tolerance,
                rtol=0.0,
            ),
        ),
        (
            "Premium equals American minus European price",
            np.isclose(
                dataset["early_exercise_premium"],
                dataset["american_price"] - dataset["european_price"],
                atol=tolerance,
                rtol=0.0,
            ),
        ),
        (
            "Normalized American target is consistent",
            np.isclose(
                dataset["normalized_american_price"],
                dataset["american_price"] / dataset["strike"],
                atol=tolerance,
                rtol=0.0,
            ),
        ),
        (
            "Normalized European target is consistent",
            np.isclose(
                dataset["normalized_european_price"],
                dataset["european_price"] / dataset["strike"],
                atol=tolerance,
                rtol=0.0,
            ),
        ),
        (
            "Normalized premium target is consistent",
            np.isclose(
                dataset["normalized_early_exercise_premium"],
                dataset["early_exercise_premium"] / dataset["strike"],
                atol=tolerance,
                rtol=0.0,
            ),
        ),
        (
            "Exercise label matches intrinsic-continuation comparison",
            dataset["exercise_now"].astype(bool)
            == (dataset["intrinsic_value"] >= dataset["continuation_value"] - 1e-12),
        ),
    ]

    records: list[dict[str, object]] = []
    for name, passed_values in checks:
        passed_array = np.asarray(passed_values, dtype=bool)
        violations = int((~passed_array).sum())
        records.append(
            {
                "check": name,
                "observations": int(len(dataset)),
                "violations": violations,
                "violation_rate": violations / len(dataset) if len(dataset) else np.nan,
                "passed": violations == 0,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_floor_adjustments(
    dataset: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Quantify how often and by how much finite-tree labels were repaired."""

    if "pricing_floor_adjustment" not in dataset.columns:
        raise ValueError("Dataset is missing pricing_floor_adjustment.")
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")

    adjustments = dataset["pricing_floor_adjustment"].astype(float)
    repaired = adjustments > tolerance
    repaired_values = adjustments[repaired]

    summary = {
        "observations": int(len(dataset)),
        "repaired_observations": int(repaired.sum()),
        "repair_rate": float(repaired.mean()),
        "total_adjustment": float(adjustments.sum()),
        "mean_adjustment_all": float(adjustments.mean()),
        "mean_adjustment_when_repaired": (
            float(repaired_values.mean()) if not repaired_values.empty else 0.0
        ),
        "median_adjustment_when_repaired": (
            float(repaired_values.median()) if not repaired_values.empty else 0.0
        ),
        "maximum_adjustment": float(adjustments.max()),
    }
    return pd.DataFrame([summary])


def summarize_exercise_balance(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return counts and shares for continuation and immediate exercise labels."""

    if "exercise_now" not in dataset.columns:
        raise ValueError("Dataset is missing exercise_now.")
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")

    labels = dataset["exercise_now"].astype(bool)
    counts = labels.value_counts(dropna=False).reindex([False, True], fill_value=0)

    return pd.DataFrame(
        {
            "exercise_now": [False, True],
            "decision": ["Continue", "Exercise"],
            "observations": [int(counts.loc[False]), int(counts.loc[True])],
            "share": [
                float(counts.loc[False] / len(dataset)),
                float(counts.loc[True] / len(dataset)),
            ],
        }
    )


def build_dataset_quality_report(
    dataset: pd.DataFrame,
    *,
    ranges: ParameterRanges | None = None,
    tolerance: float = 1e-10,
) -> DatasetQualityReport:
    """Run the complete pre-model dataset audit."""

    return DatasetQualityReport(
        schema_checks=validate_dataset_schema(dataset),
        parameter_checks=validate_parameter_ranges(dataset, ranges=ranges),
        financial_checks=validate_financial_bounds(dataset, tolerance=tolerance),
        floor_summary=summarize_floor_adjustments(dataset),
        exercise_summary=summarize_exercise_balance(dataset),
    )


def assert_dataset_quality(report: DatasetQualityReport) -> None:
    """Raise a detailed error when any mandatory dataset check fails."""

    failed_tables: list[str] = []
    for name, table in (
        ("schema", report.schema_checks),
        ("parameters", report.parameter_checks),
        ("financial", report.financial_checks),
    ):
        mandatory = table.get("mandatory", pd.Series(True, index=table.index)).astype(bool)
        failed = table.loc[mandatory & ~table["passed"].astype(bool)]
        if not failed.empty:
            failed_tables.append(f"{name}: {failed['check'].tolist()}")

    if failed_tables:
        raise ValueError("Dataset quality checks failed — " + "; ".join(failed_tables))


__all__ = [
    "DatasetQualityReport",
    "NUMERIC_COLUMNS",
    "PARAMETER_COLUMNS",
    "REQUIRED_COLUMNS",
    "assert_dataset_quality",
    "build_dataset_quality_report",
    "load_option_dataset",
    "summarize_exercise_balance",
    "summarize_floor_adjustments",
    "validate_dataset_schema",
    "validate_financial_bounds",
    "validate_parameter_ranges",
]
