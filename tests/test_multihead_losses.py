"""Tests for final-model multi-objective losses."""

from __future__ import annotations

import pytest
import torch

from src.training.multihead_losses import (
    IntegratedMultiHeadLoss,
    MultiHeadLossWeights,
    multihead_loss_preset,
)


def _batch(batch_size: int = 12):
    raw = {
        "floor_residual": torch.rand(batch_size, 1, requires_grad=True),
        "direct_price": torch.rand(batch_size, 1, requires_grad=True),
        "continuation_value": torch.rand(batch_size, 1, requires_grad=True),
        "exercise_logits": torch.randn(batch_size, 1, requires_grad=True),
    }
    european = torch.rand(batch_size, 1) * 0.2
    intrinsic = torch.rand(batch_size, 1) * 0.2
    floor = torch.maximum(european, intrinsic)
    residual_target = torch.rand(batch_size, 1) * 0.05
    direct_target = floor + residual_target
    continuation_target = torch.rand(batch_size, 1) * 0.2
    exercise_target = (intrinsic >= continuation_target).float()
    return (
        raw,
        residual_target,
        direct_target,
        continuation_target,
        exercise_target,
        european,
        intrinsic,
    )


@pytest.mark.parametrize(
    "name",
    ["balanced", "pricing_focused", "decision_focused"],
)
def test_all_presets_produce_finite_loss(name: str) -> None:
    batch = _batch()
    loss_fn = IntegratedMultiHeadLoss(
        config=multihead_loss_preset(name),
        positive_class_weight=2.0,
    )

    result = loss_fn(
        batch[0],
        floor_residual_target=batch[1],
        direct_price_target=batch[2],
        continuation_target=batch[3],
        exercise_target=batch[4],
        normalized_european=batch[5],
        normalized_intrinsic=batch[6],
    )

    assert set(result) == {
        "loss",
        "floor_residual_loss",
        "direct_price_loss",
        "continuation_loss",
        "exercise_loss",
        "price_consistency_loss",
        "exercise_consistency_loss",
    }
    assert all(torch.isfinite(value) for value in result.values())
    result["loss"].backward()
    assert all(value.grad is not None for value in batch[0].values())


def test_sample_weights_are_supported() -> None:
    batch = _batch(5)
    loss_fn = IntegratedMultiHeadLoss()
    result = loss_fn(
        batch[0],
        floor_residual_target=batch[1],
        direct_price_target=batch[2],
        continuation_target=batch[3],
        exercise_target=batch[4],
        normalized_european=batch[5],
        normalized_intrinsic=batch[6],
        sample_weight=torch.tensor([[1.0], [0.0], [2.0], [1.0], [1.0]]),
    )
    assert torch.isfinite(result["loss"])


def test_invalid_loss_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiHeadLossWeights(floor_residual=-1.0)
    with pytest.raises(ValueError):
        multihead_loss_preset("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        IntegratedMultiHeadLoss(positive_class_weight=0.0)


def test_zero_consistency_losses_for_coherent_heads() -> None:
    european = torch.tensor([[0.10], [0.05]])
    intrinsic = torch.tensor([[0.08], [0.10]])
    continuation = torch.tensor([[0.12], [0.04]])
    floor = torch.maximum(european, intrinsic)
    residual = torch.tensor([[0.02], [0.01]], requires_grad=True)
    direct = (floor + residual).detach().clone().requires_grad_(True)
    exercise_probability = torch.sigmoid(50.0 * (intrinsic - continuation))
    logits = torch.logit(exercise_probability.clamp(1e-6, 1.0 - 1e-6)).detach()
    logits.requires_grad_(True)
    raw = {
        "floor_residual": residual,
        "direct_price": direct,
        "continuation_value": continuation.clone().requires_grad_(True),
        "exercise_logits": logits,
    }
    target = (intrinsic >= continuation).float()
    loss_fn = IntegratedMultiHeadLoss()
    result = loss_fn(
        raw,
        floor_residual_target=residual.detach(),
        direct_price_target=direct.detach(),
        continuation_target=continuation,
        exercise_target=target,
        normalized_european=european,
        normalized_intrinsic=intrinsic,
    )

    assert float(result["price_consistency_loss"].detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(result["exercise_consistency_loss"].detach()) == pytest.approx(0.0, abs=1e-7)
