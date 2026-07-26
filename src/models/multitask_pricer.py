"""Multi-task neural models for American put pricing and exercise decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MultiTaskMLPConfig:
    """Serializable architecture configuration for shared-backbone models."""

    input_features: int = 5
    shared_hidden_sizes: tuple[int, ...] = (128, 128, 64)
    batch_norm_after: tuple[int, ...] = (0, 1)
    regression_head_sizes: tuple[int, ...] = (32,)
    classification_head_sizes: tuple[int, ...] = (32,)
    residual_softplus: bool = True

    def __post_init__(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive.")
        if not self.shared_hidden_sizes or any(
            size <= 0 for size in self.shared_hidden_sizes
        ):
            raise ValueError("shared_hidden_sizes must contain positive values.")
        if any(size <= 0 for size in self.regression_head_sizes):
            raise ValueError("regression_head_sizes must contain positive values.")
        if any(size <= 0 for size in self.classification_head_sizes):
            raise ValueError("classification_head_sizes must contain positive values.")
        invalid = set(self.batch_norm_after) - set(
            range(len(self.shared_hidden_sizes))
        )
        if invalid:
            raise ValueError(f"Invalid batch_norm_after indices: {sorted(invalid)}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_head(
    *,
    input_features: int,
    hidden_sizes: tuple[int, ...],
    output_features: int,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_features
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(in_features, hidden), nn.SiLU()])
        in_features = hidden
    layers.append(nn.Linear(in_features, output_features))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class MultiTaskAmericanPutMLP(nn.Module):
    """Shared-backbone model with residual-regression and exercise heads.

    The regression head predicts a non-negative normalized residual above a
    financial floor. The classification head returns raw exercise logits.
    """

    def __init__(self, config: MultiTaskMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiTaskMLPConfig()

        shared_layers: list[nn.Module] = []
        in_features = self.config.input_features
        for index, hidden in enumerate(self.config.shared_hidden_sizes):
            shared_layers.append(nn.Linear(in_features, hidden))
            if index in self.config.batch_norm_after:
                shared_layers.append(nn.BatchNorm1d(hidden))
            shared_layers.append(nn.SiLU())
            in_features = hidden

        self.backbone = nn.Sequential(*shared_layers)
        final_activation: nn.Module | None = (
            nn.Softplus() if self.config.residual_softplus else None
        )
        self.regression_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.regression_head_sizes,
            output_features=1,
            final_activation=final_activation,
        )
        self.classification_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.classification_head_sizes,
            output_features=1,
        )
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

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature tensor; received {tuple(features.shape)}."
            )
        if features.shape[1] != self.config.input_features:
            raise ValueError(
                f"Expected {self.config.input_features} features; "
                f"received {features.shape[1]}."
            )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_features(features)
        shared = self.backbone(features)
        residual = self.regression_head(shared)
        exercise_logits = self.classification_head(shared)
        return residual, exercise_logits

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


@dataclass(frozen=True, slots=True)
class ExerciseClassifierConfig:
    """Configuration for the exercise-only neural baseline."""

    input_features: int = 5
    hidden_sizes: tuple[int, ...] = (128, 128, 64, 32)
    batch_norm_after: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive.")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values.")
        invalid = set(self.batch_norm_after) - set(range(len(self.hidden_sizes)))
        if invalid:
            raise ValueError(f"Invalid batch_norm_after indices: {sorted(invalid)}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ExerciseClassifierMLP(nn.Module):
    """Exercise-versus-continuation baseline returning raw logits."""

    def __init__(self, config: ExerciseClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExerciseClassifierConfig()
        layers: list[nn.Module] = []
        in_features = self.config.input_features
        for index, hidden in enumerate(self.config.hidden_sizes):
            layers.append(nn.Linear(in_features, hidden))
            if index in self.config.batch_norm_after:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.SiLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
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
            raise ValueError("Expected a 2D feature tensor.")
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


__all__ = [
    "ExerciseClassifierConfig",
    "ExerciseClassifierMLP",
    "MultiTaskAmericanPutMLP",
    "MultiTaskMLPConfig",
]
