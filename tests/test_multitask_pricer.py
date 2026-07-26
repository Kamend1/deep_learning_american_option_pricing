import torch

from src.models.multitask_pricer import (
    ExerciseClassifierMLP,
    MultiTaskAmericanPutMLP,
    MultiTaskMLPConfig,
)


def test_multitask_forward_shapes_and_non_negative_residual() -> None:
    model = MultiTaskAmericanPutMLP()
    features = torch.randn(16, 5)
    residual, logits = model(features)
    assert residual.shape == (16, 1)
    assert logits.shape == (16, 1)
    assert torch.all(residual >= 0.0)


def test_multitask_gradients_flow_through_shared_backbone() -> None:
    model = MultiTaskAmericanPutMLP()
    features = torch.randn(8, 5)
    residual, logits = model(features)
    loss = residual.mean() + logits.mean()
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )


def test_exercise_classifier_returns_logits() -> None:
    model = ExerciseClassifierMLP()
    logits = model(torch.randn(10, 5))
    assert logits.shape == (10, 1)
    assert torch.isfinite(logits).all()


def test_fixed_seed_reproduces_initial_parameters() -> None:
    torch.manual_seed(42)
    first = MultiTaskAmericanPutMLP(MultiTaskMLPConfig())
    torch.manual_seed(42)
    second = MultiTaskAmericanPutMLP(MultiTaskMLPConfig())
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters())
    )
