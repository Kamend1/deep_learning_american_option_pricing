"""Training, checkpointing, and inference loops for the integrated model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.integrated_multihead_pricer import reconstruct_integrated_outputs
from src.training.checkpointing import atomic_torch_save
from src.training.multihead_losses import IntegratedMultiHeadLoss


@dataclass(frozen=True, slots=True)
class IntegratedTrainingConfig:
    """Training settings for the final static multi-head model."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 12
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    seed: int = 42
    min_delta: float = 1e-8

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.early_stopping_patience <= 0 or self.scheduler_patience <= 0:
            raise ValueError("Patience values must be positive.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must lie between zero and one.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when supplied.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def set_integrated_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tensor(batch: dict[str, object], key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a tensor.")
    return value.to(device, non_blocking=True)


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def run_integrated_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: IntegratedMultiHeadLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Run one training or evaluation epoch for all four model heads."""

    training = optimizer is not None
    model.train(training)
    component_names = (
        "loss",
        "floor_residual_loss",
        "direct_price_loss",
        "continuation_loss",
        "exercise_loss",
        "price_consistency_loss",
        "exercise_consistency_loss",
    )
    totals = {name: 0.0 for name in component_names}
    totals.update(
        {
            "constrained_absolute_error": 0.0,
            "continuation_absolute_error": 0.0,
            "classification_correct": 0.0,
            "decision_disagreement": 0.0,
            "observations": 0.0,
        }
    )

    for batch in loader:
        features = _tensor(batch, "features", device)
        floor_residual_target = _tensor(
            batch, "floor_residual_target", device
        )
        direct_price_target = _tensor(batch, "direct_price_target", device)
        continuation_target = _tensor(batch, "continuation_target", device)
        exercise_target = _tensor(batch, "exercise_target", device)
        normalized_european = _tensor(batch, "normalized_european", device)
        normalized_intrinsic = _tensor(batch, "normalized_intrinsic", device)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                raise TypeError("sample_weight must be a tensor.")
            sample_weight = sample_weight.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, mixed_precision):
                raw_outputs = model(features)
                losses = loss_fn(
                    raw_outputs,
                    floor_residual_target=floor_residual_target,
                    direct_price_target=direct_price_target,
                    continuation_target=continuation_target,
                    exercise_target=exercise_target,
                    normalized_european=normalized_european,
                    normalized_intrinsic=normalized_intrinsic,
                    sample_weight=sample_weight,
                )
                reconstructed = reconstruct_integrated_outputs(
                    raw_outputs,
                    normalized_european=normalized_european,
                    normalized_intrinsic=normalized_intrinsic,
                    decision_sharpness=loss_fn.config.decision_sharpness,
                )
            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["loss"]).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_norm,
                        )
                    optimizer.step()

        batch_size = float(features.shape[0])
        totals["observations"] += batch_size
        for name in component_names:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
        totals["constrained_absolute_error"] += float(
            torch.abs(
                reconstructed["constrained_price"].detach()
                - direct_price_target
            ).sum().cpu()
        )
        totals["continuation_absolute_error"] += float(
            torch.abs(
                reconstructed["continuation_value"].detach()
                - continuation_target
            ).sum().cpu()
        )
        predicted_class = (
            reconstructed["exercise_probability"].detach() >= 0.5
        )
        continuation_class = (
            reconstructed["continuation_exercise_probability"].detach() >= 0.5
        )
        totals["classification_correct"] += float(
            (predicted_class == exercise_target.bool()).sum().cpu()
        )
        totals["decision_disagreement"] += float(
            (predicted_class != continuation_class).sum().cpu()
        )

    observations = max(totals["observations"], 1.0)
    result = {
        name: totals[name] / observations
        for name in component_names
    }
    result.update(
        {
            "constrained_price_mae": (
                totals["constrained_absolute_error"] / observations
            ),
            "continuation_mae": (
                totals["continuation_absolute_error"] / observations
            ),
            "classification_accuracy": (
                totals["classification_correct"] / observations
            ),
            "decision_disagreement_rate": (
                totals["decision_disagreement"] / observations
            ),
            "observations": totals["observations"],
        }
    )
    return result


def fit_integrated_multihead_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    loss_fn: IntegratedMultiHeadLoss,
    config: IntegratedTrainingConfig,
    device: torch.device,
    checkpoint_path: str | Path,
    model_config: dict[str, object] | None = None,
    warm_start_report: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Train the integrated model with scheduling and early stopping."""

    set_integrated_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision and device.type == "cuda",
    )

    checkpoint = Path(checkpoint_path)
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_integrated_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=grad_scaler,
            gradient_clip_norm=config.gradient_clip_norm,
            mixed_precision=config.mixed_precision,
        )
        validation_metrics = run_integrated_epoch(
            model,
            validation_loader,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=config.mixed_precision,
        )
        scheduler.step(validation_metrics["loss"])
        row: dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.perf_counter() - started),
        }
        for name, value in train_metrics.items():
            row[f"train_{name}"] = float(value)
        for name, value in validation_metrics.items():
            row[f"validation_{name}"] = float(value)
        history.append(row)

        improved = validation_metrics["loss"] < best_validation - config.min_delta
        if improved:
            best_validation = validation_metrics["loss"]
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_validation,
                    "training_config": config.to_dict(),
                    "model_config": dict(model_config or {}),
                    "loss_config": loss_fn.config.to_dict(),
                    "positive_class_weight": float(
                        loss_fn.positive_class_weight.item()
                    ),
                    "warm_start_report": dict(warm_start_report or {}),
                    "authoritative_price_output": "constrained_price",
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    return pd.DataFrame(history)


@torch.inference_mode()
def predict_integrated_multihead_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    decision_sharpness: float = 50.0,
    classification_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return aligned predictions, targets, and internal-consistency fields."""

    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie between zero and one.")
    model.eval()
    model.to(device)
    rows: list[pd.DataFrame] = []

    for batch in loader:
        features = _tensor(batch, "features", device)
        european = _tensor(batch, "normalized_european", device)
        intrinsic = _tensor(batch, "normalized_intrinsic", device)
        raw_outputs = model(features)
        reconstructed = reconstruct_integrated_outputs(
            raw_outputs,
            normalized_european=european,
            normalized_intrinsic=intrinsic,
            decision_sharpness=decision_sharpness,
        )
        row_ids = batch["row_id"]
        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.detach().cpu().numpy()

        exercise_probability = reconstructed["exercise_probability"]
        continuation_probability = reconstructed[
            "continuation_exercise_probability"
        ]
        result = {
            "sample_id": np.asarray(row_ids),
            "predicted_floor_residual": (
                reconstructed["floor_residual"].cpu().numpy().reshape(-1)
            ),
            "predicted_normalized_american_price": (
                reconstructed["constrained_price"].cpu().numpy().reshape(-1)
            ),
            "predicted_direct_normalized_american_price": (
                reconstructed["direct_price"].cpu().numpy().reshape(-1)
            ),
            "predicted_normalized_continuation_value": (
                reconstructed["continuation_value"].cpu().numpy().reshape(-1)
            ),
            "exercise_logit": (
                reconstructed["exercise_logits"].cpu().numpy().reshape(-1)
            ),
            "exercise_probability": exercise_probability.cpu().numpy().reshape(-1),
            "continuation_exercise_probability": (
                continuation_probability.cpu().numpy().reshape(-1)
            ),
            "exercise_prediction": (
                exercise_probability >= classification_threshold
            ).cpu().numpy().reshape(-1).astype(bool),
            "continuation_exercise_prediction": (
                continuation_probability >= classification_threshold
            ).cpu().numpy().reshape(-1).astype(bool),
            "normalized_european_price": european.cpu().numpy().reshape(-1),
            "normalized_intrinsic_value": intrinsic.cpu().numpy().reshape(-1),
            "true_normalized_american_price": _tensor(
                batch, "direct_price_target", device
            ).cpu().numpy().reshape(-1),
            "true_normalized_continuation_value": _tensor(
                batch, "continuation_target", device
            ).cpu().numpy().reshape(-1),
            "true_floor_residual": _tensor(
                batch, "floor_residual_target", device
            ).cpu().numpy().reshape(-1),
            "exercise_target": _tensor(
                batch, "exercise_target", device
            ).cpu().numpy().reshape(-1).astype(bool),
        }
        frame = pd.DataFrame(result)
        frame["price_head_gap"] = (
            frame["predicted_direct_normalized_american_price"]
            - frame["predicted_normalized_american_price"]
        )
        frame["exercise_probability_gap"] = (
            frame["exercise_probability"]
            - frame["continuation_exercise_probability"]
        )
        frame["decision_disagreement"] = (
            frame["exercise_prediction"]
            != frame["continuation_exercise_prediction"]
        )
        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


__all__ = [
    "IntegratedTrainingConfig",
    "fit_integrated_multihead_model",
    "predict_integrated_multihead_model",
    "run_integrated_epoch",
    "set_integrated_seed",
]
