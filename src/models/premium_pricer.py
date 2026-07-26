"""Residual neural architectures for American put-option pricing.

The models in this module predict a normalized residual rather than the full
American option price. Two economically distinct residual bases are supported:

- ``european``: V_A / K - V_E / K, the normalized early-exercise premium;
- ``financial_floor``: V_A / K - max(V_E, intrinsic) / K.

The second formulation guarantees both the European and intrinsic lower bounds
when the residual output is constrained to be non-negative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


OutputActivation = Literal["linear", "softplus"]
ResidualBase = Literal["european", "financial_floor"]


@dataclass(frozen=True, slots=True)
class PremiumMLPConfig:
    """Serializable configuration for a residual American put pricer."""

    input_features: int = 5
    hidden_sizes: tuple[int, ...] = (128, 128, 64, 32)
    batch_norm_after: tuple[int, ...] = (0, 1)
    output_activation: OutputActivation = "softplus"
    residual_base: ResidualBase = "european"

    def __post_init__(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive.")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values.")
        invalid = set(self.batch_norm_after) - set(range(len(self.hidden_sizes)))
        if invalid:
            raise ValueError(f"Invalid batch_norm_after indices: {sorted(invalid)}")
        if self.output_activation not in {"linear", "softplus"}:
            raise ValueError("Unsupported output activation.")
        if self.residual_base not in {"european", "financial_floor"}:
            raise ValueError("Unsupported residual base.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PremiumAmericanPutMLP(nn.Module):
    """Feed-forward network predicting a normalized pricing residual."""

    def __init__(self, config: PremiumMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or PremiumMLPConfig()

        layers: list[nn.Module] = []
        in_features = self.config.input_features
        for index, hidden in enumerate(self.config.hidden_sizes):
            layers.append(nn.Linear(in_features, hidden))
            if index in self.config.batch_norm_after:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.SiLU())
            in_features = hidden

        layers.append(nn.Linear(in_features, 1))
        if self.config.output_activation == "softplus":
            layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.0,
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature tensor; received shape {tuple(features.shape)}."
            )
        if features.shape[1] != self.config.input_features:
            raise ValueError(
                f"Expected {self.config.input_features} features; "
                f"received {features.shape[1]}."
            )
        return self.network(features)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def normalized_financial_floor(
    normalized_european: torch.Tensor,
    normalized_intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Return max(European value, intrinsic value) in normalized units."""

    if normalized_european.shape != normalized_intrinsic.shape:
        raise ValueError("European and intrinsic tensors must have equal shape.")
    return torch.maximum(normalized_european, normalized_intrinsic)


def reconstruct_normalized_american_price(
    residual: torch.Tensor,
    *,
    normalized_european: torch.Tensor,
    normalized_intrinsic: torch.Tensor | None = None,
    residual_base: ResidualBase = "european",
) -> torch.Tensor:
    """Reconstruct normalized American value from a predicted residual."""

    if residual.shape != normalized_european.shape:
        raise ValueError("Residual and European tensors must have equal shape.")

    if residual_base == "european":
        base = normalized_european
    elif residual_base == "financial_floor":
        if normalized_intrinsic is None:
            raise ValueError(
                "normalized_intrinsic is required for the financial-floor base."
            )
        base = normalized_financial_floor(
            normalized_european,
            normalized_intrinsic,
        )
    else:
        raise ValueError("Unsupported residual base.")

    return base + residual


def calculate_normalized_residual_target(
    normalized_american: torch.Tensor,
    *,
    normalized_european: torch.Tensor,
    normalized_intrinsic: torch.Tensor | None = None,
    residual_base: ResidualBase = "european",
) -> torch.Tensor:
    """Calculate the normalized training residual for the selected base."""

    if normalized_american.shape != normalized_european.shape:
        raise ValueError("American and European tensors must have equal shape.")

    zero_residual = torch.zeros_like(normalized_american)
    base_price = reconstruct_normalized_american_price(
        zero_residual,
        normalized_european=normalized_european,
        normalized_intrinsic=normalized_intrinsic,
        residual_base=residual_base,
    )
    return normalized_american - base_price


__all__ = [
    "PremiumAmericanPutMLP",
    "PremiumMLPConfig",
    "ResidualBase",
    "calculate_normalized_residual_target",
    "normalized_financial_floor",
    "reconstruct_normalized_american_price",
]
