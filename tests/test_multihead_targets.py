"""Tests for integrated static-model target preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.multihead_targets import (
    add_integrated_targets,
    validate_integrated_targets,
)


def test_integrated_targets_are_derived_from_raw_prices() -> None:
    frame = pd.DataFrame(
        {
            "strike": [100.0, 100.0],
            "intrinsic_value": [10.0, 0.0],
            "continuation_value": [9.0, 5.0],
            "european_price": [9.5, 4.0],
            "american_price": [10.0, 5.5],
            "exercise_now": [True, False],
        }
    )

    result = add_integrated_targets(frame)

    assert result["normalized_intrinsic_value"].tolist() == [0.1, 0.0]
    assert result["normalized_continuation_value"].tolist() == [0.09, 0.05]
    assert result["normalized_financial_floor"].tolist() == [0.1, 0.04]
    assert result["normalized_floor_residual"].tolist() == [0.0, 0.015]
    validate_integrated_targets(result)


def test_inconsistent_exercise_label_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "strike": [100.0],
            "intrinsic_value": [10.0],
            "continuation_value": [9.0],
            "european_price": [9.5],
            "american_price": [10.0],
            "exercise_now": [False],
        }
    )
    with pytest.raises(ValueError):
        add_integrated_targets(frame)
