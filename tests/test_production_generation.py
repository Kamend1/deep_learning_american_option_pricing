"""Tests for scalable production-data design and small-batch pricing."""

from __future__ import annotations

import numpy as np

from src.data.production_generation import (
    CORE_RANGES,
    ProductionDatasetConfig,
    build_component_specs,
    build_priced_frame,
    sample_parameter_chunk,
    validate_generated_frame,
)


def test_default_design_contains_exactly_1_45_million_rows() -> None:
    config = ProductionDatasetConfig()
    specs = build_component_specs(config)

    assert config.total_observations == 1_450_000
    assert config.in_domain_observations == 1_250_000
    assert config.ood_observations == 200_000
    assert sum(spec.observations for spec in specs) == 1_450_000
    assert [spec.name for spec in specs] == [
        "core",
        "boundary",
        "ood_high_volatility",
        "ood_extreme_moneyness",
        "ood_long_maturity",
        "ood_rate_dividend",
    ]


def test_parameter_sampling_is_reproducible() -> None:
    first = sample_parameter_chunk(
        n_samples=32,
        ranges=CORE_RANGES,
        seed=123,
        strike=100.0,
    )
    second = sample_parameter_chunk(
        n_samples=32,
        ranges=CORE_RANGES,
        seed=123,
        strike=100.0,
    )
    for column in first:
        np.testing.assert_allclose(first[column], second[column])


def test_small_priced_frame_is_financially_valid() -> None:
    config = ProductionDatasetConfig(tree_steps=25, chunk_size=16)
    parameters = sample_parameter_chunk(
        n_samples=16,
        ranges=CORE_RANGES,
        seed=7,
        strike=config.strike,
    )
    frame = build_priced_frame(
        parameters=parameters,
        sample_ids=np.arange(16, dtype=np.int64),
        component="core",
        tree_steps=config.tree_steps,
        split_eligible=True,
        config=config,
    )

    validate_generated_frame(frame)
    assert len(frame) == 16
    assert frame["sample_id"].is_unique
    assert set(frame["split"]).issubset({"train", "validation", "test"})
    assert (frame["american_price"] >= frame["intrinsic_value"] - 1e-10).all()
    assert (frame["american_price"] >= frame["european_price"] - 1e-10).all()
