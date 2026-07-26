import pytest
import torch

from src.models.direct_pricer import DirectAmericanPutMLP, DirectMLPConfig
from src.models.premium_pricer import (
    PremiumAmericanPutMLP,
    PremiumMLPConfig,
    calculate_normalized_residual_target,
    reconstruct_normalized_american_price,
)


def test_premium_model_output_shape_and_gradient() -> None:
    model = PremiumAmericanPutMLP()
    features = torch.randn(16, 5, requires_grad=True)
    output = model(features)
    assert output.shape == (16, 1)
    output.mean().backward()
    assert features.grad is not None


def test_softplus_premium_is_non_negative() -> None:
    model = PremiumAmericanPutMLP(
        PremiumMLPConfig(output_activation="softplus")
    )
    output = model(torch.randn(32, 5))
    assert torch.all(output >= 0.0)


def test_unconstrained_output_can_be_signed() -> None:
    model = PremiumAmericanPutMLP(
        PremiumMLPConfig(output_activation="linear")
    )
    assert not isinstance(model.network[-1], torch.nn.Softplus)


def test_reconstruction_uses_european_base() -> None:
    residual = torch.tensor([[0.02], [0.03]])
    european = torch.tensor([[0.10], [0.20]])
    result = reconstruct_normalized_american_price(
        residual,
        normalized_european=european,
        residual_base="european",
    )
    assert torch.allclose(result, torch.tensor([[0.12], [0.23]]))


def test_reconstruction_uses_financial_floor() -> None:
    residual = torch.tensor([[0.02], [0.03]])
    european = torch.tensor([[0.10], [0.20]])
    intrinsic = torch.tensor([[0.15], [0.05]])
    result = reconstruct_normalized_american_price(
        residual,
        normalized_european=european,
        normalized_intrinsic=intrinsic,
        residual_base="financial_floor",
    )
    assert torch.allclose(result, torch.tensor([[0.17], [0.23]]))


def test_residual_target_reconstructs_original_price() -> None:
    american = torch.tensor([[0.18], [0.24]])
    european = torch.tensor([[0.10], [0.20]])
    intrinsic = torch.tensor([[0.15], [0.05]])
    target = calculate_normalized_residual_target(
        american,
        normalized_european=european,
        normalized_intrinsic=intrinsic,
        residual_base="financial_floor",
    )
    reconstructed = reconstruct_normalized_american_price(
        target,
        normalized_european=european,
        normalized_intrinsic=intrinsic,
        residual_base="financial_floor",
    )
    assert torch.allclose(reconstructed, american)


def test_parameter_count_matches_direct_model_capacity() -> None:
    direct = DirectAmericanPutMLP(DirectMLPConfig())
    premium = PremiumAmericanPutMLP(PremiumMLPConfig())
    assert premium.trainable_parameter_count == direct.trainable_parameter_count


def test_invalid_feature_shape_raises() -> None:
    model = PremiumAmericanPutMLP()
    with pytest.raises(ValueError):
        model(torch.randn(5))
