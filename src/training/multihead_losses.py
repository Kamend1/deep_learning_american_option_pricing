"""Multi-objective losses for the final integrated static pricing model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np
import torch
from torch import nn

from src.models.integrated_multihead_pricer import reconstruct_integrated_outputs


LossPresetName = Literal["balanced", "pricing_focused", "decision_focused"]


@dataclass(frozen=True, slots=True)
class MultiHeadLossWeights:
    """Relative contribution of every objective in the integrated loss."""

    floor_residual: float = 1.0
    direct_price: float = 0.5
    continuation: float = 0.5
    exercise: float = 0.5
    price_consistency: float = 0.10
    exercise_consistency: float = 0.10

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("All multi-head loss weights must be finite and non-negative.")
        if sum(values.values()) <= 0.0:
            raise ValueError("At least one multi-head loss weight must be positive.")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "MultiHeadLossWeights":
        """Reconstruct loss weights from saved metadata."""

        if not isinstance(raw, Mapping):
            raise TypeError("raw must be a mapping.")
        return cls(**{key: float(value) for key, value in raw.items()})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultiHeadLossConfig:
    """Numerical settings for multi-head regression and consistency losses."""

    weights: MultiHeadLossWeights = MultiHeadLossWeights()
    regression_beta: float = 1.0
    decision_sharpness: float = 50.0

    def __post_init__(self) -> None:
        if self.regression_beta < 0.0:
            raise ValueError("regression_beta cannot be negative.")
        if self.decision_sharpness <= 0.0:
            raise ValueError("decision_sharpness must be positive.")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "MultiHeadLossConfig":
        """Reconstruct a loss configuration from checkpoint metadata."""

        if not isinstance(raw, Mapping):
            raise TypeError("raw must be a mapping.")
        values = dict(raw)
        raw_weights = values.get("weights", {})
        values["weights"] = MultiHeadLossWeights.from_dict(raw_weights)
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": self.weights.to_dict(),
            "regression_beta": self.regression_beta,
            "decision_sharpness": self.decision_sharpness,
        }


def multihead_loss_preset(name: LossPresetName) -> MultiHeadLossConfig:
    """Return one of the three predefined, auditable loss configurations."""

    presets: dict[str, MultiHeadLossWeights] = {
        "balanced": MultiHeadLossWeights(
            floor_residual=1.0,
            direct_price=0.5,
            continuation=0.5,
            exercise=0.5,
            price_consistency=0.10,
            exercise_consistency=0.10,
        ),
        "pricing_focused": MultiHeadLossWeights(
            floor_residual=1.50,
            direct_price=1.00,
            continuation=0.25,
            exercise=0.25,
            price_consistency=0.20,
            exercise_consistency=0.05,
        ),
        "decision_focused": MultiHeadLossWeights(
            floor_residual=0.75,
            direct_price=0.25,
            continuation=1.00,
            exercise=1.00,
            price_consistency=0.05,
            exercise_consistency=0.25,
        ),
    }
    try:
        weights = presets[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown loss preset {name!r}; expected one of {sorted(presets)}."
        ) from exc
    return MultiHeadLossConfig(weights=weights)


def _as_column(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape (batch, 1).")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return tensor


def _smooth_l1_elements(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    absolute = torch.abs(prediction - target)
    if beta == 0.0:
        return absolute
    return torch.where(
        absolute < beta,
        0.5 * absolute.pow(2) / beta,
        absolute - 0.5 * beta,
    )


class IntegratedMultiHeadLoss(nn.Module):
    """Joint price, continuation, exercise, and consistency objective."""

    def __init__(
        self,
        *,
        config: MultiHeadLossConfig | None = None,
        positive_class_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config or multihead_loss_preset("balanced")
        if not np.isfinite(positive_class_weight) or positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be finite and positive.")
        self.register_buffer(
            "positive_class_weight",
            torch.tensor([positive_class_weight], dtype=torch.float32),
        )

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if sample_weight is None:
            return values.mean()
        sample_weight = _as_column(sample_weight, name="sample_weight")
        if sample_weight.shape != values.shape:
            raise ValueError("sample_weight must match the element-wise loss shape.")
        if torch.any(sample_weight < 0.0):
            raise ValueError("sample_weight cannot be negative.")
        denominator = sample_weight.sum().clamp_min(torch.finfo(values.dtype).eps)
        return (values * sample_weight).sum() / denominator

    def forward(
        self,
        raw_outputs: Mapping[str, torch.Tensor],
        *,
        floor_residual_target: torch.Tensor,
        direct_price_target: torch.Tensor,
        continuation_target: torch.Tensor,
        exercise_target: torch.Tensor,
        normalized_european: torch.Tensor,
        normalized_intrinsic: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        floor_residual_target = _as_column(
            floor_residual_target,
            name="floor_residual_target",
        )
        direct_price_target = _as_column(
            direct_price_target,
            name="direct_price_target",
        )
        continuation_target = _as_column(
            continuation_target,
            name="continuation_target",
        )
        exercise_target = _as_column(exercise_target, name="exercise_target")
        normalized_european = _as_column(
            normalized_european,
            name="normalized_european",
        )
        normalized_intrinsic = _as_column(
            normalized_intrinsic,
            name="normalized_intrinsic",
        )
        shapes = {
            floor_residual_target.shape,
            direct_price_target.shape,
            continuation_target.shape,
            exercise_target.shape,
            normalized_european.shape,
            normalized_intrinsic.shape,
        }
        if len(shapes) != 1:
            raise ValueError("All integrated targets must share one shape.")
        if torch.any((exercise_target < 0.0) | (exercise_target > 1.0)):
            raise ValueError("exercise_target must be binary.")

        reconstructed = reconstruct_integrated_outputs(
            raw_outputs,
            normalized_european=normalized_european,
            normalized_intrinsic=normalized_intrinsic,
            decision_sharpness=self.config.decision_sharpness,
        )
        beta = self.config.regression_beta
        floor_residual_loss = self._weighted_mean(
            _smooth_l1_elements(
                reconstructed["floor_residual"],
                floor_residual_target,
                beta=beta,
            ),
            sample_weight,
        )
        direct_price_loss = self._weighted_mean(
            _smooth_l1_elements(
                reconstructed["direct_price"],
                direct_price_target,
                beta=beta,
            ),
            sample_weight,
        )
        continuation_loss = self._weighted_mean(
            _smooth_l1_elements(
                reconstructed["continuation_value"],
                continuation_target,
                beta=beta,
            ),
            sample_weight,
        )
        exercise_elements = nn.functional.binary_cross_entropy_with_logits(
            reconstructed["exercise_logits"],
            exercise_target,
            pos_weight=self.positive_class_weight.to(
                reconstructed["exercise_logits"].dtype
            ),
            reduction="none",
        )
        exercise_loss = self._weighted_mean(exercise_elements, sample_weight)
        price_consistency_loss = self._weighted_mean(
            _smooth_l1_elements(
                reconstructed["direct_price"],
                reconstructed["constrained_price"],
                beta=beta,
            ),
            sample_weight,
        )
        exercise_consistency_loss = self._weighted_mean(
            (
                reconstructed["exercise_probability"]
                - reconstructed["continuation_exercise_probability"]
            ).pow(2),
            sample_weight,
        )

        weights = self.config.weights
        total = (
            weights.floor_residual * floor_residual_loss
            + weights.direct_price * direct_price_loss
            + weights.continuation * continuation_loss
            + weights.exercise * exercise_loss
            + weights.price_consistency * price_consistency_loss
            + weights.exercise_consistency * exercise_consistency_loss
        )
        return {
            "loss": total,
            "floor_residual_loss": floor_residual_loss,
            "direct_price_loss": direct_price_loss,
            "continuation_loss": continuation_loss,
            "exercise_loss": exercise_loss,
            "price_consistency_loss": price_consistency_loss,
            "exercise_consistency_loss": exercise_consistency_loss,
        }


__all__ = [
    "IntegratedMultiHeadLoss",
    "LossPresetName",
    "MultiHeadLossConfig",
    "MultiHeadLossWeights",
    "multihead_loss_preset",
]
