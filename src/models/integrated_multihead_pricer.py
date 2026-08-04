"""Integrated static multi-head model for American put pricing.

The model uses one shared representation of the static option state and four
specialized heads:

- non-negative residual above the financial floor;
- direct normalized American price;
- normalized continuation value;
- exercise-versus-continuation logit.

The constrained price reconstructed from the residual head is the authoritative
pricing output because it is guaranteed to be no lower than both the European
value and intrinsic value supplied to the reconstruction function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class IntegratedMultiHeadConfig:
    """Serializable architecture configuration for the final static model."""

    input_features: int = 5
    shared_hidden_sizes: tuple[int, ...] = (192, 192, 96)
    batch_norm_after: tuple[int, ...] = (0, 1)
    dropout: float = 0.10
    residual_head_sizes: tuple[int, ...] = (48,)
    direct_head_sizes: tuple[int, ...] = (48,)
    continuation_head_sizes: tuple[int, ...] = (48,)
    exercise_head_sizes: tuple[int, ...] = (48,)
    decision_sharpness: float = 50.0

    def __post_init__(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive.")
        if not self.shared_hidden_sizes or any(
            value <= 0 for value in self.shared_hidden_sizes
        ):
            raise ValueError("shared_hidden_sizes must contain positive values.")
        for name, sizes in {
            "residual_head_sizes": self.residual_head_sizes,
            "direct_head_sizes": self.direct_head_sizes,
            "continuation_head_sizes": self.continuation_head_sizes,
            "exercise_head_sizes": self.exercise_head_sizes,
        }.items():
            if any(value <= 0 for value in sizes):
                raise ValueError(f"{name} must contain positive values.")
        invalid = set(self.batch_norm_after) - set(
            range(len(self.shared_hidden_sizes))
        )
        if invalid:
            raise ValueError(f"Invalid batch_norm_after indices: {sorted(invalid)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.decision_sharpness <= 0.0:
            raise ValueError("decision_sharpness must be positive.")

    @classmethod
    def step6_compatible(cls, **overrides: object) -> "IntegratedMultiHeadConfig":
        """Return a backbone configuration compatible with the Step 6 model."""

        values: dict[str, object] = {
            "shared_hidden_sizes": (128, 128, 64),
            "batch_norm_after": (0, 1),
            "dropout": 0.0,
            "residual_head_sizes": (32,),
            "direct_head_sizes": (32,),
            "continuation_head_sizes": (32,),
            "exercise_head_sizes": (32,),
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "IntegratedMultiHeadConfig":
        """Reconstruct a configuration from saved JSON/checkpoint metadata."""

        if not isinstance(raw, Mapping):
            raise TypeError("raw must be a mapping.")
        values = dict(raw)
        tuple_fields = (
            "shared_hidden_sizes",
            "batch_norm_after",
            "residual_head_sizes",
            "direct_head_sizes",
            "continuation_head_sizes",
            "exercise_head_sizes",
        )
        for field in tuple_fields:
            if field in values:
                values[field] = tuple(values[field])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_head(
    *,
    input_features: int,
    hidden_sizes: tuple[int, ...],
    output_features: int = 1,
    non_negative: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_features
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(in_features, hidden), nn.SiLU()])
        in_features = hidden
    layers.append(nn.Linear(in_features, output_features))
    if non_negative:
        layers.append(nn.Softplus())
    return nn.Sequential(*layers)


class IntegratedAmericanPutMultiHeadMLP(nn.Module):
    """Shared-backbone American put model with four financially related heads."""

    def __init__(self, config: IntegratedMultiHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or IntegratedMultiHeadConfig()

        shared_layers: list[nn.Module] = []
        in_features = self.config.input_features
        for index, hidden in enumerate(self.config.shared_hidden_sizes):
            shared_layers.append(nn.Linear(in_features, hidden))
            if index in self.config.batch_norm_after:
                shared_layers.append(nn.BatchNorm1d(hidden))
            shared_layers.append(nn.SiLU())
            if self.config.dropout > 0.0:
                shared_layers.append(nn.Dropout(self.config.dropout))
            in_features = hidden

        self.backbone = nn.Sequential(*shared_layers)
        self.floor_residual_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.residual_head_sizes,
            non_negative=True,
        )
        self.direct_price_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.direct_head_sizes,
            non_negative=True,
        )
        self.continuation_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.continuation_head_sizes,
            non_negative=True,
        )
        self.exercise_head = _build_head(
            input_features=in_features,
            hidden_sizes=self.config.exercise_head_sizes,
            non_negative=False,
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
                f"Expected a two-dimensional feature tensor; got {features.shape}."
            )
        if features.shape[1] != self.config.input_features:
            raise ValueError(
                f"Expected {self.config.input_features} features; "
                f"received {features.shape[1]}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or infinite values.")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_features(features)
        shared = self.backbone(features)
        return {
            "floor_residual": self.floor_residual_head(shared),
            "direct_price": self.direct_price_head(shared),
            "continuation_value": self.continuation_head(shared),
            "exercise_logits": self.exercise_head(shared),
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def _as_column(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape (batch, 1).")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return tensor


def reconstruct_integrated_outputs(
    raw_outputs: Mapping[str, torch.Tensor],
    *,
    normalized_european: torch.Tensor,
    normalized_intrinsic: torch.Tensor,
    decision_sharpness: float = 50.0,
) -> dict[str, torch.Tensor]:
    """Reconstruct financially meaningful outputs from the four raw heads."""

    required = {
        "floor_residual",
        "direct_price",
        "continuation_value",
        "exercise_logits",
    }
    missing = required - set(raw_outputs)
    if missing:
        raise ValueError(f"raw_outputs are missing keys: {sorted(missing)}")
    if decision_sharpness <= 0.0:
        raise ValueError("decision_sharpness must be positive.")

    residual = _as_column(raw_outputs["floor_residual"], name="floor_residual")
    direct_price = _as_column(raw_outputs["direct_price"], name="direct_price")
    continuation = _as_column(
        raw_outputs["continuation_value"],
        name="continuation_value",
    )
    exercise_logits = _as_column(
        raw_outputs["exercise_logits"],
        name="exercise_logits",
    )
    european = _as_column(normalized_european, name="normalized_european")
    intrinsic = _as_column(normalized_intrinsic, name="normalized_intrinsic")

    shapes = {
        residual.shape,
        direct_price.shape,
        continuation.shape,
        exercise_logits.shape,
        european.shape,
        intrinsic.shape,
    }
    if len(shapes) != 1:
        raise ValueError("All head outputs and financial inputs must share one shape.")

    financial_floor = torch.maximum(european, intrinsic)
    constrained_price = financial_floor + residual
    exercise_probability = torch.sigmoid(exercise_logits)
    continuation_exercise_probability = torch.sigmoid(
        decision_sharpness * (intrinsic - continuation)
    )

    return {
        "financial_floor": financial_floor,
        "floor_residual": residual,
        "constrained_price": constrained_price,
        "direct_price": direct_price,
        "continuation_value": continuation,
        "exercise_logits": exercise_logits,
        "exercise_probability": exercise_probability,
        "continuation_exercise_probability": continuation_exercise_probability,
        "price_head_gap": direct_price - constrained_price,
        "exercise_probability_gap": (
            exercise_probability - continuation_exercise_probability
        ),
    }


def copy_compatible_backbone_weights(
    target_model: IntegratedAmericanPutMultiHeadMLP,
    source_state_dict: Mapping[str, torch.Tensor],
    *,
    source_prefix: str = "backbone.",
) -> dict[str, object]:
    """Copy shape-compatible Step 6 backbone parameters into the final model."""

    target_state = target_model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []

    for key, target_value in target_state.items():
        if not key.startswith("backbone."):
            continue
        source_key = source_prefix + key.removeprefix("backbone.")
        source_value = source_state_dict.get(source_key)
        if source_value is None or source_value.shape != target_value.shape:
            skipped.append(key)
            continue
        target_state[key] = source_value.detach().clone().to(target_value.dtype)
        copied.append(key)

    target_model.load_state_dict(target_state)
    return {
        "copied_keys": copied,
        "skipped_keys": skipped,
        "copied_count": len(copied),
        "skipped_count": len(skipped),
    }


__all__ = [
    "IntegratedAmericanPutMultiHeadMLP",
    "IntegratedMultiHeadConfig",
    "copy_compatible_backbone_weights",
    "reconstruct_integrated_outputs",
]
