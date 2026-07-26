"""Direct multilayer perceptron for normalized American put prices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


OutputActivation = Literal["linear", "softplus"]


@dataclass(frozen=True, slots=True)
class DirectMLPConfig:
    """Serializable architecture configuration for the direct baseline."""

    input_features: int = 5
    hidden_sizes: tuple[int, ...] = (128, 128, 64, 32)
    batch_norm_after: tuple[int, ...] = (0, 1)
    output_activation: OutputActivation = "softplus"

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DirectAmericanPutMLP(nn.Module):
    """Feed-forward baseline predicting normalized American put value."""

    def __init__(self, config: DirectMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectMLPConfig()

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
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = ["DirectAmericanPutMLP", "DirectMLPConfig"]
