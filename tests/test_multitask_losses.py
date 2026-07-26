import numpy as np
import pandas as pd
import pytest
import torch

from src.training.multitask_losses import (
    MultiTaskLossConfig,
    MultiTaskPricingLoss,
    calculate_positive_class_weight,
)


def test_positive_class_weight_uses_negative_positive_ratio() -> None:
    labels = pd.Series([0, 0, 0, 1])
    assert calculate_positive_class_weight(labels) == pytest.approx(3.0)


def test_multitask_loss_is_finite_and_backpropagates() -> None:
    predicted = torch.tensor([[0.2], [0.4]], requires_grad=True)
    target = torch.tensor([[0.1], [0.3]])
    logits = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    exercise = torch.tensor([[1.0], [0.0]])
    loss_fn = MultiTaskPricingLoss(
        config=MultiTaskLossConfig(exercise_lambda=0.5),
        positive_class_weight=2.0,
    )
    result = loss_fn(predicted, target, logits, exercise)
    assert set(result) == {"loss", "regression_loss", "classification_loss"}
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert predicted.grad is not None
    assert logits.grad is not None


def test_all_zero_and_all_one_batches_are_finite() -> None:
    loss_fn = MultiTaskPricingLoss(positive_class_weight=3.0)
    for exercise in (torch.zeros(4, 1), torch.ones(4, 1)):
        result = loss_fn(
            torch.zeros(4, 1),
            torch.zeros(4, 1),
            torch.zeros(4, 1),
            exercise,
        )
        assert torch.isfinite(result["loss"])


def test_invalid_labels_rejected_for_pos_weight() -> None:
    with pytest.raises(ValueError):
        calculate_positive_class_weight(np.array([0.0, 0.2, 1.0]))
