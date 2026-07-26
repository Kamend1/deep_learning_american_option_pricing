"""Tests for synthetic dataset generation and dataset-quality auditing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.data.dataset_validation import (
    REQUIRED_COLUMNS,
    assert_dataset_quality,
    build_dataset_quality_report,
)
from src.data.generation import ParameterRanges, generate_american_put_dataset


def _small_dataset(seed: int = 42) -> pd.DataFrame:
    return generate_american_put_dataset(
        n_samples=48,
        tree_steps=40,
        seed=seed,
        strike=100.0,
    )


def test_generation_returns_requested_rows_and_required_columns() -> None:
    dataset = _small_dataset()

    assert len(dataset) == 48
    assert set(REQUIRED_COLUMNS).issubset(dataset.columns)
    assert dataset["sample_id"].is_unique


def test_generation_is_reproducible_for_fixed_seed() -> None:
    first = _small_dataset(seed=123)
    second = _small_dataset(seed=123)

    pdt.assert_frame_equal(first, second, check_exact=True)


def test_generation_changes_when_seed_changes() -> None:
    first = _small_dataset(seed=123)
    second = _small_dataset(seed=124)

    assert not np.allclose(first["moneyness"], second["moneyness"])


def test_generated_parameters_remain_inside_declared_ranges() -> None:
    ranges = ParameterRanges(
        moneyness=(0.70, 1.30),
        time_to_maturity=(0.05, 1.50),
        volatility=(0.10, 0.60),
        risk_free_rate=(0.01, 0.08),
        dividend_yield=(0.00, 0.05),
    )
    dataset = generate_american_put_dataset(
        n_samples=40,
        tree_steps=30,
        seed=9,
        ranges=ranges,
    )

    for column in (
        "moneyness",
        "time_to_maturity",
        "volatility",
        "risk_free_rate",
        "dividend_yield",
    ):
        lower, upper = getattr(ranges, column)
        assert dataset[column].between(lower, upper, inclusive="both").all()


def test_generated_dataset_passes_quality_report() -> None:
    dataset = _small_dataset()
    report = build_dataset_quality_report(dataset)

    assert report.passed
    assert_dataset_quality(report)
    assert bool(report.financial_checks["passed"].all())


def test_no_missing_or_infinite_values_in_required_columns() -> None:
    dataset = _small_dataset()

    assert not dataset.loc[:, REQUIRED_COLUMNS].isna().any().any()
    numeric = dataset.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_american_labels_respect_no_arbitrage_floors() -> None:
    dataset = _small_dataset()

    assert (dataset["american_price"] >= dataset["intrinsic_value"] - 1e-10).all()
    assert (dataset["american_price"] >= dataset["european_price"] - 1e-10).all()
    assert (dataset["early_exercise_premium"] >= -1e-10).all()


def test_exercise_labels_match_root_comparison() -> None:
    dataset = _small_dataset()
    expected = dataset["intrinsic_value"] >= dataset["continuation_value"] - 1e-10

    assert (dataset["exercise_now"].astype(bool) == expected).all()
