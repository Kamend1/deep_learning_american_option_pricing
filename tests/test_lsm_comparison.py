import numpy as np
import pandas as pd
import pytest

from src.evaluation.lsm_comparison import (
    aggregate_seed_results,
    align_contract_results,
    compare_lsm_methods,
    confidence_interval_coverage,
    exercise_policy_metrics,
    runtime_summary,
    stopping_time_total_variation,
)


def test_contract_alignment_requires_exact_ids() -> None:
    reference = pd.DataFrame({"contract_id": ["a", "b"], "price": [1.0, 2.0]})
    candidate = pd.DataFrame({"contract_id": ["b", "a"], "price": [2.1, 0.9]})
    merged = align_contract_results(reference, candidate)
    assert merged["contract_id"].tolist() == ["a", "b"]


def test_pricing_comparison_ranks_lower_error_first() -> None:
    frame = pd.DataFrame(
        {
            "crr": [1.0, 2.0, 3.0],
            "classical": [1.1, 2.1, 3.1],
            "neural": [1.01, 2.01, 3.01],
        }
    )
    table = compare_lsm_methods(
        frame, benchmark_column="crr", method_columns=["classical", "neural"]
    )
    assert table.iloc[0]["method"] == "neural"


def test_interval_coverage_and_policy_metrics() -> None:
    coverage = confidence_interval_coverage(
        [1.0, 2.0], [0.9, 2.1], [1.1, 2.2]
    )
    assert coverage == 0.5
    metrics = exercise_policy_metrics(
        [2, 5, 5, 3], [2, 5, 4, 5], maturity_index=5
    )
    assert 0.0 <= metrics["exact_step_agreement"] <= 1.0
    assert stopping_time_total_variation([1, 1, 5], [1, 5, 5], n_steps=5) >= 0.0


def test_runtime_and_seed_aggregation() -> None:
    runtime = pd.DataFrame(
        {"method": ["a", "a", "b"], "runtime_seconds": [1.0, 2.0, 3.0]}
    )
    summary = runtime_summary(runtime)
    assert set(summary["method"]) == {"a", "b"}
    seed_frame = pd.DataFrame(
        {
            "contract_id": ["c", "c", "c"],
            "method": ["m", "m", "m"],
            "price": [1.0, 1.1, 0.9],
        }
    )
    aggregated = aggregate_seed_results(seed_frame)
    assert aggregated.iloc[0]["seed_count"] == 3
