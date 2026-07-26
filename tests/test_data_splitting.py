"""Tests for deterministic splitting and out-of-domain isolation."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.splitting import (
    OODRangeSpec,
    SplitConfig,
    build_split_manifest,
    create_out_of_domain_sets,
    create_train_validation_test_split,
    validate_ood_exclusion,
    validate_split_integrity,
)


def _dataset(size: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": range(size),
            "exercise_now": [index % 5 == 0 for index in range(size)],
            "moneyness": [0.40 + index * 0.012 for index in range(size)],
            "volatility": [0.10 + (index % 10) * 0.10 for index in range(size)],
            "time_to_maturity": [0.25 + (index % 8) * 0.50 for index in range(size)],
            "risk_free_rate": [0.01 + (index % 6) * 0.03 for index in range(size)],
            "dividend_yield": [0.01 + (index % 5) * 0.03 for index in range(size)],
        }
    )


def test_split_is_reproducible() -> None:
    data = _dataset()
    config = SplitConfig(seed=17)

    first = create_train_validation_test_split(data, config=config)
    second = create_train_validation_test_split(data, config=config)

    for name in ("train", "validation", "test"):
        pdt.assert_frame_equal(first[name], second[name], check_exact=True)


def test_split_allocates_every_row_once_and_is_disjoint() -> None:
    data = _dataset()
    config = SplitConfig(seed=17)
    splits = create_train_validation_test_split(data, config=config)

    all_ids = pd.concat(
        [frame[["sample_id"]] for frame in splits.values()],
        ignore_index=True,
    )
    assert len(all_ids) == len(data)
    assert all_ids["sample_id"].is_unique
    assert set(all_ids["sample_id"]) == set(data["sample_id"])
    assert bool(validate_split_integrity(splits, config=config)["passed"].all())


def test_split_proportions_are_close_to_configuration() -> None:
    data = _dataset(size=200)
    config = SplitConfig(seed=2)
    splits = create_train_validation_test_split(data, config=config)

    assert len(splits["train"]) == pytest.approx(140, abs=1)
    assert len(splits["validation"]) == pytest.approx(30, abs=1)
    assert len(splits["test"]) == pytest.approx(30, abs=1)


def test_stratification_preserves_exercise_rate_approximately() -> None:
    data = _dataset(size=200)
    splits = create_train_validation_test_split(data, config=SplitConfig(seed=3))
    overall_rate = data["exercise_now"].mean()

    for frame in splits.values():
        assert frame["exercise_now"].mean() == pytest.approx(overall_rate, abs=0.03)


def test_split_manifest_contains_counts_and_configuration() -> None:
    data = _dataset()
    config = SplitConfig(seed=5)
    splits = create_train_validation_test_split(data, config=config)
    manifest = build_split_manifest(splits, config=config)

    assert manifest["total_observations"] == len(data)
    assert manifest["split_counts"]["train"] == len(splits["train"])
    assert manifest["split_config"]["seed"] == 5


def test_invalid_split_fractions_raise() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train_fraction=0.80, validation_fraction=0.15, test_fraction=0.15)


def test_ood_selection_uses_inclusive_ranges() -> None:
    data = _dataset()
    specs = (
        OODRangeSpec(
            name="high_vol",
            description="Test",
            ranges={"volatility": (0.80, 1.00)},
        ),
    )
    ood = create_out_of_domain_sets(data, specs=specs)

    assert not ood["high_vol"].empty
    assert ood["high_vol"]["volatility"].between(0.80, 1.00).all()
    assert (ood["high_vol"]["ood_set"] == "high_vol").all()


def test_ood_exclusion_detects_overlap_with_training() -> None:
    data = _dataset()
    splits = create_train_validation_test_split(data, config=SplitConfig(seed=4))
    overlapping = {"stress": splits["train"].head(3).copy()}

    result = validate_ood_exclusion(splits["train"], overlapping)

    assert result.loc[0, "overlap_with_training"] == 3
    assert not bool(result.loc[0, "passed"])
