"""Loss functions and training-only class weighting for multi-task learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MultiTaskLossConfig:
    """Controls the relative influence of pricing and exercise objectives."""

    exercise_lambda: float = 0.5
    regression_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.exercise_lambda < 0.0:
            raise ValueError("exercise_lambda cannot be negative.")
        if self.regression_beta < 0.0:
            raise ValueError("regression_beta cannot be negative.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_positive_class_weight(
    training_labels: pd.Series | np.ndarray | torch.Tensor,
    *,
    maximum_weight: float | None = 50.0,
) -> float:
    """Calculate negative/positive ratio from training labels only."""

    if isinstance(training_labels, torch.Tensor):
        values = training_labels.detach().cpu().numpy()
    elif isinstance(training_labels, pd.Series):
        values = training_labels.to_numpy()
    else:
        values = np.asarray(training_labels)

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("training_labels cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("training_labels contain non-finite values.")
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("training_labels must be binary.")

    positives = float(values.sum())
    negatives = float(len(values) - positives)
    if positives == 0.0:
        raise ValueError("Cannot calculate pos_weight without positive labels.")
    weight = negatives / positives
    if maximum_weight is not None:
        if maximum_weight <= 0.0:
            raise ValueError("maximum_weight must be positive.")
        weight = min(weight, maximum_weight)
    return float(weight)


class MultiTaskPricingLoss(nn.Module):
    """Smooth-L1 regression plus weighted binary cross entropy."""

    def __init__(
        self,
        *,
        config: MultiTaskLossConfig | None = None,
        positive_class_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config or MultiTaskLossConfig()
        if not np.isfinite(positive_class_weight) or positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be finite and positive.")
        self.register_buffer(
            "positive_class_weight",
            torch.tensor([positive_class_weight], dtype=torch.float32),
        )

    def _weighted_mean(
        self,
        values: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if sample_weight is None:
            return values.mean()
        if sample_weight.ndim == 1:
            sample_weight = sample_weight.unsqueeze(1)
        if sample_weight.shape != values.shape:
            raise ValueError("sample_weight must match the loss tensor shape.")
        if not torch.isfinite(sample_weight).all():
            raise ValueError("sample_weight contains non-finite values.")
        if torch.any(sample_weight < 0.0):
            raise ValueError("sample_weight cannot be negative.")
        denominator = sample_weight.sum().clamp_min(torch.finfo(values.dtype).eps)
        return (values * sample_weight).sum() / denominator

    def forward(
        self,
        predicted_residual: torch.Tensor,
        target_residual: torch.Tensor,
        exercise_logits: torch.Tensor,
        exercise_target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tensors = (
            predicted_residual,
            target_residual,
            exercise_logits,
            exercise_target,
        )
        if any(tensor.ndim == 1 for tensor in tensors):
            tensors = tuple(
                tensor.unsqueeze(1) if tensor.ndim == 1 else tensor
                for tensor in tensors
            )
            (
                predicted_residual,
                target_residual,
                exercise_logits,
                exercise_target,
            ) = tensors
        if predicted_residual.shape != target_residual.shape:
            raise ValueError("Regression prediction and target shapes differ.")
        if exercise_logits.shape != exercise_target.shape:
            raise ValueError("Classification logit and target shapes differ.")
        if predicted_residual.shape != exercise_logits.shape:
            raise ValueError("Regression and classification batch shapes differ.")

        absolute_error = torch.abs(predicted_residual - target_residual)
        beta = self.config.regression_beta
        if beta == 0.0:
            regression_elements = absolute_error
        else:
            regression_elements = torch.where(
                absolute_error < beta,
                0.5 * absolute_error.pow(2) / beta,
                absolute_error - 0.5 * beta,
            )
        regression_loss = self._weighted_mean(
            regression_elements,
            sample_weight,
        )

        positive_class_weight = self.positive_class_weight.to(
            device=exercise_logits.device,
            dtype=exercise_logits.dtype,
        )

        classification_elements = nn.functional.binary_cross_entropy_with_logits(
            exercise_logits,
            exercise_target,
            pos_weight=positive_class_weight,
            reduction="none",
        )
        classification_loss = self._weighted_mean(
            classification_elements,
            sample_weight,
        )
        total_loss = (
            regression_loss
            + self.config.exercise_lambda * classification_loss
        )
        return {
            "loss": total_loss,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
        }


__all__ = [
    "MultiTaskLossConfig",
    "MultiTaskPricingLoss",
    "calculate_positive_class_weight",
]
