import numpy as np
import pandas as pd
import pytest
import torch

from src.training.losses import (
    PremiumWeightConfig,
    WeightedSmoothL1Loss,
    apply_premium_weighting,
    fit_premium_weighting,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0.0, 0.001, 0.01, 0.05],
            "boundary_distance_normalized": [0.20, 0.05, 0.005, 0.0],
            "exercise_now": [False, False, True, True],
        }
    )


def test_weighted_loss_matches_unweighted_with_unit_weights() -> None:
    prediction = torch.tensor([[0.0], [2.0]])
    target = torch.tensor([[0.0], [0.0]])
    loss_fn = WeightedSmoothL1Loss(beta=1.0)
    unweighted = loss_fn(prediction, target)
    weighted = loss_fn(prediction, target, torch.ones_like(target))
    assert torch.allclose(unweighted, weighted)


def test_weighted_loss_prioritizes_high_weight_error() -> None:
    prediction = torch.tensor([[1.0], [1.0]])
    target = torch.tensor([[0.0], [0.0]])
    weights = torch.tensor([[1.0], [5.0]])
    loss = WeightedSmoothL1Loss()(prediction, target, weights)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_magnitude_weights_are_training_fitted_and_normalized() -> None:
    frame = _frame()
    fitted = fit_premium_weighting(
        frame,
        target_column="target",
        mode="magnitude",
    )
    weights = apply_premium_weighting(frame, fitted=fitted)
    assert weights.dtype == np.float32
    assert weights.mean() == pytest.approx(1.0, abs=1e-6)
    assert weights[-1] > weights[0]


def test_boundary_weights_prioritize_boundary_and_exercise_rows() -> None:
    frame = _frame()
    fitted = fit_premium_weighting(
        frame,
        target_column="target",
        mode="boundary",
        config=PremiumWeightConfig(boundary_band=0.01),
    )
    weights = apply_premium_weighting(frame, fitted=fitted)
    assert weights[-1] > weights[0]
    assert np.isfinite(weights).all()


def test_validation_transform_reuses_training_reference() -> None:
    training = _frame()
    validation = _frame().assign(target=[0.0, 0.1, 0.2, 0.3])
    fitted = fit_premium_weighting(
        training,
        target_column="target",
        mode="magnitude",
    )
    reference = fitted.magnitude_reference
    _ = apply_premium_weighting(validation, fitted=fitted)
    assert fitted.magnitude_reference == reference


def test_negative_weights_are_rejected() -> None:
    loss_fn = WeightedSmoothL1Loss()
    with pytest.raises(ValueError):
        loss_fn(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.tensor([[1.0], [-1.0]]),
        )
