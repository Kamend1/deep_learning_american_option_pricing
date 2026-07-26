"""Tests for the direct American put MLP."""

from __future__ import annotations

import torch

from src.models.direct_pricer import DirectAmericanPutMLP, DirectMLPConfig


def test_forward_shape_and_non_negative_output() -> None:
    model = DirectAmericanPutMLP(
        DirectMLPConfig(output_activation="softplus")
    )
    output = model(torch.randn(17, 5))
    assert output.shape == (17, 1)
    assert torch.all(output >= 0.0)


def test_gradients_propagate() -> None:
    model = DirectAmericanPutMLP()
    inputs = torch.randn(32, 5)
    targets = torch.rand(32, 1)
    loss = torch.nn.functional.smooth_l1_loss(model(inputs), targets)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_parameter_count_is_stable() -> None:
    model = DirectAmericanPutMLP()
    assert 20_000 < model.trainable_parameter_count < 30_000
