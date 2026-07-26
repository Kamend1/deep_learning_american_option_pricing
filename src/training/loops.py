"""Reusable regression training loops for American option models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpointing import atomic_torch_save


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training controls shared by the direct and premium models."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
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
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.scheduler_patience <= 0:
            raise ValueError("scheduler_patience must be positive.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between zero and one.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when supplied.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _move_batch(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = batch["features"]
    target = batch["target"]
    if not isinstance(features, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("DataLoader batches must contain tensor features and targets.")
    return (
        features.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    """Train one epoch and return observation-weighted loss and MAE."""

    model.train()
    total_loss = 0.0
    total_absolute_error = 0.0
    observations = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for batch in loader:
        features, target = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(features)
            loss = loss_fn(prediction, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        batch_size = target.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_absolute_error += float(
            torch.abs(prediction.detach() - target).sum().item()
        )
        observations += batch_size

    if observations == 0:
        raise ValueError("Training loader produced no observations.")
    return {
        "loss": total_loss / observations,
        "mae": total_absolute_error / observations,
    }


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one split without updating model parameters."""

    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    observations = 0

    for batch in loader:
        features, target = _move_batch(batch, device)
        prediction = model(features)
        loss = loss_fn(prediction, target)
        batch_size = target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_absolute_error += float(torch.abs(prediction - target).sum().item())
        observations += batch_size

    if observations == 0:
        raise ValueError("Evaluation loader produced no observations.")
    return {
        "loss": total_loss / observations,
        "mae": total_absolute_error / observations,
    }


def fit_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    device: torch.device,
    checkpoint_path: str | Path,
    config: TrainingConfig | None = None,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Train with early stopping, LR scheduling, AMP, and best checkpointing."""

    cfg = config or TrainingConfig()
    set_global_seed(cfg.seed)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_learning_rate,
    )
    amp_enabled = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )
        validation_metrics = evaluate_one_epoch(
            model,
            validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        scheduler.step(validation_metrics["loss"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "validation_loss": validation_metrics["loss"],
            "validation_mae": validation_metrics["mae"],
            "learning_rate": learning_rate,
            "epoch_seconds": elapsed,
        }
        history.append(row)

        improved = validation_metrics["loss"] < (
            best_validation_loss - cfg.min_delta
        )
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "training_config": cfg.to_dict(),
                    "model_config": model_config or {},
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_metrics['loss']:.6f} | "
            f"Train MAE: {train_metrics['mae']:.6f} | "
            f"Val loss: {validation_metrics['loss']:.6f} | "
            f"Val MAE: {validation_metrics['mae']:.6f} | "
            f"LR: {learning_rate:.2e}"
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    result = pd.DataFrame(history)
    result.attrs["best_epoch"] = best_epoch
    result.attrs["best_validation_loss"] = best_validation_loss
    return result


@torch.inference_mode()
def predict_regression_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Return row IDs, targets, and predictions in loader order."""

    model.eval()
    model.to(device)
    ids: list[object] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch in loader:
        features, target = _move_batch(batch, device)
        prediction = model(features)
        row_ids = batch["row_id"]
        if isinstance(row_ids, torch.Tensor):
            ids.extend(row_ids.cpu().numpy().tolist())
        else:
            ids.extend(list(row_ids))
        targets.append(target.cpu().numpy().reshape(-1))
        predictions.append(prediction.cpu().numpy().reshape(-1))

    return pd.DataFrame(
        {
            "sample_id": ids,
            "target": np.concatenate(targets),
            "prediction": np.concatenate(predictions),
        }
    )


__all__ = [
    "TrainingConfig",
    "evaluate_one_epoch",
    "fit_regression_model",
    "predict_regression_model",
    "set_global_seed",
    "train_one_epoch",
]
