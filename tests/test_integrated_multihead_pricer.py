"""Tests for the final integrated static multi-head architecture."""

from __future__ import annotations

import pytest
import torch

from src.models.integrated_multihead_pricer import (
    IntegratedAmericanPutMultiHeadMLP,
    IntegratedMultiHeadConfig,
    copy_compatible_backbone_weights,
    reconstruct_integrated_outputs,
)
from src.models.multitask_pricer import MultiTaskAmericanPutMLP, MultiTaskMLPConfig


def test_integrated_model_returns_all_four_heads() -> None:
    model = IntegratedAmericanPutMultiHeadMLP()
    features = torch.randn(11, 5)

    outputs = model(features)

    assert set(outputs) == {
        "floor_residual",
        "direct_price",
        "continuation_value",
        "exercise_logits",
    }
    assert all(value.shape == (11, 1) for value in outputs.values())
    assert torch.all(outputs["floor_residual"] >= 0.0)
    assert torch.all(outputs["direct_price"] >= 0.0)
    assert torch.all(outputs["continuation_value"] >= 0.0)


def test_reconstructed_price_respects_financial_floor() -> None:
    model = IntegratedAmericanPutMultiHeadMLP()
    features = torch.randn(8, 5)
    european = torch.tensor([[0.05], [0.10], [0.20], [0.02]] * 2)
    intrinsic = torch.tensor([[0.08], [0.03], [0.15], [0.05]] * 2)

    reconstructed = reconstruct_integrated_outputs(
        model(features),
        normalized_european=european,
        normalized_intrinsic=intrinsic,
    )

    floor = torch.maximum(european, intrinsic)
    assert torch.all(reconstructed["constrained_price"] >= floor)
    assert torch.allclose(
        reconstructed["constrained_price"],
        floor + reconstructed["floor_residual"],
    )
    assert torch.all(
        (reconstructed["exercise_probability"] >= 0.0)
        & (reconstructed["exercise_probability"] <= 1.0)
    )
    assert torch.all(
        (reconstructed["continuation_exercise_probability"] >= 0.0)
        & (reconstructed["continuation_exercise_probability"] <= 1.0)
    )


def test_all_heads_propagate_gradients() -> None:
    model = IntegratedAmericanPutMultiHeadMLP(
        IntegratedMultiHeadConfig(dropout=0.0)
    )
    features = torch.randn(16, 5)
    outputs = model(features)
    loss = sum(value.mean() for value in outputs.values())

    loss.backward()

    backbone_gradients = [
        parameter.grad
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    assert backbone_gradients
    assert all(gradient is not None for gradient in backbone_gradients)
    assert any(torch.any(gradient != 0.0) for gradient in backbone_gradients)


def test_step6_compatible_backbone_can_be_warm_started() -> None:
    source = MultiTaskAmericanPutMLP(MultiTaskMLPConfig())
    target = IntegratedAmericanPutMultiHeadMLP(
        IntegratedMultiHeadConfig.step6_compatible()
    )

    report = copy_compatible_backbone_weights(target, source.state_dict())

    assert report["copied_count"] > 0
    assert report["skipped_count"] == 0
    for key, value in source.backbone.state_dict().items():
        assert torch.equal(target.backbone.state_dict()[key], value)


def test_invalid_feature_shape_is_rejected() -> None:
    model = IntegratedAmericanPutMultiHeadMLP()
    with pytest.raises(ValueError):
        model(torch.randn(5))
    with pytest.raises(ValueError):
        model(torch.randn(5, 4))


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        IntegratedMultiHeadConfig(dropout=1.0)
    with pytest.raises(ValueError):
        IntegratedMultiHeadConfig(decision_sharpness=0.0)
