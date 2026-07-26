"""Weighted regression losses for early-exercise-premium learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn


WeightingMode = Literal["unweighted", "magnitude", "boundary"]


@dataclass(frozen=True, slots=True)
class PremiumWeightConfig:
    """Controls for training-only premium and boundary sample weights."""

    magnitude_quantile: float = 0.75
    magnitude_scale: float = 3.0
    boundary_band: float = 0.01
    boundary_scale: float = 4.0
    exercise_scale: float = 1.0
    max_weight: float = 10.0
    normalize_mean: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.magnitude_quantile < 1.0:
            raise ValueError("magnitude_quantile must be between zero and one.")
        if self.magnitude_scale < 0.0:
            raise ValueError("magnitude_scale cannot be negative.")
        if self.boundary_band <= 0.0:
            raise ValueError("boundary_band must be positive.")
        if self.boundary_scale < 0.0 or self.exercise_scale < 0.0:
            raise ValueError("Weight scales cannot be negative.")
        if self.max_weight < 1.0:
            raise ValueError("max_weight must be at least one.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FittedPremiumWeighting:
    """Training-fitted weighting state applied unchanged to later splits."""

    mode: WeightingMode
    target_column: str
    magnitude_reference: float
    config: PremiumWeightConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "target_column": self.target_column,
            "magnitude_reference": self.magnitude_reference,
            "config": self.config.to_dict(),
        }


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 loss with optional observation-level sample weights."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        if beta < 0.0:
            raise ValueError("beta cannot be negative.")
        self.beta = float(beta)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target tensors must have equal shape.")

        absolute_error = torch.abs(prediction - target)
        if self.beta == 0.0:
            element_loss = absolute_error
        else:
            element_loss = torch.where(
                absolute_error < self.beta,
                0.5 * absolute_error.pow(2) / self.beta,
                absolute_error - 0.5 * self.beta,
            )

        if sample_weight is None:
            return element_loss.mean()

        if sample_weight.ndim == 1:
            sample_weight = sample_weight.unsqueeze(1)
        if sample_weight.shape != element_loss.shape:
            raise ValueError(
                "sample_weight must have the same shape as prediction and target."
            )
        if not torch.isfinite(sample_weight).all():
            raise ValueError("sample_weight contains non-finite values.")
        if torch.any(sample_weight < 0.0):
            raise ValueError("sample_weight cannot contain negative values.")

        denominator = sample_weight.sum().clamp_min(torch.finfo(element_loss.dtype).eps)
        return (element_loss * sample_weight).sum() / denominator


def fit_premium_weighting(
    training_frame: pd.DataFrame,
    *,
    target_column: str,
    mode: WeightingMode,
    config: PremiumWeightConfig | None = None,
) -> FittedPremiumWeighting:
    """Fit weighting references exclusively from the training split."""

    cfg = config or PremiumWeightConfig()
    if mode not in {"unweighted", "magnitude", "boundary"}:
        raise ValueError("Unsupported weighting mode.")
    if target_column not in training_frame.columns:
        raise ValueError(f"Training frame is missing {target_column!r}.")

    target = np.abs(training_frame[target_column].to_numpy(dtype=np.float64))
    if not np.isfinite(target).all():
        raise ValueError("Training target contains non-finite values.")

    positive = target[target > 0.0]
    if len(positive):
        reference = float(np.quantile(positive, cfg.magnitude_quantile))
    else:
        reference = 1.0
    reference = max(reference, np.finfo(np.float64).eps)

    return FittedPremiumWeighting(
        mode=mode,
        target_column=target_column,
        magnitude_reference=reference,
        config=cfg,
    )


def apply_premium_weighting(
    frame: pd.DataFrame,
    *,
    fitted: FittedPremiumWeighting,
    boundary_distance_column: str = "boundary_distance_normalized",
    exercise_column: str = "exercise_now",
) -> np.ndarray:
    """Apply training-fitted weighting parameters to any split."""

    if fitted.target_column not in frame.columns:
        raise ValueError(f"Frame is missing {fitted.target_column!r}.")

    target = np.abs(frame[fitted.target_column].to_numpy(dtype=np.float64))
    weights = np.ones(len(frame), dtype=np.float64)
    cfg = fitted.config

    if fitted.mode in {"magnitude", "boundary"}:
        ratio = np.clip(target / fitted.magnitude_reference, 0.0, 1.0)
        weights += cfg.magnitude_scale * ratio

    if fitted.mode == "boundary":
        required = [boundary_distance_column, exercise_column]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Boundary weighting frame is missing columns: {missing}")
        distance = np.abs(
            frame[boundary_distance_column].to_numpy(dtype=np.float64)
        )
        exercise = frame[exercise_column].astype(bool).to_numpy(dtype=np.float64)
        if not np.isfinite(distance).all():
            raise ValueError("Boundary distance contains non-finite values.")
        closeness = np.exp(-distance / cfg.boundary_band)
        weights += cfg.boundary_scale * closeness
        weights += cfg.exercise_scale * exercise

    weights = np.clip(weights, 0.0, cfg.max_weight)
    if cfg.normalize_mean:
        mean_weight = float(weights.mean()) if len(weights) else 1.0
        if mean_weight <= 0.0 or not np.isfinite(mean_weight):
            raise ValueError("Cannot normalize invalid sample weights.")
        weights = weights / mean_weight

    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Calculated sample weights are invalid.")
    return weights.astype(np.float32, copy=False)


__all__ = [
    "FittedPremiumWeighting",
    "PremiumWeightConfig",
    "WeightedSmoothL1Loss",
    "apply_premium_weighting",
    "fit_premium_weighting",
]
