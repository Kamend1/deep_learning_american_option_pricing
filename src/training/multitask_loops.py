"""Training and inference loops for exercise-only and multi-task models.

The training functions provide visible batch-level progress and persistent
epoch summaries. Progress reporting is enabled by default and can be disabled
through ``MultiTaskTrainingConfig(show_progress=False)``.
"""

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
from tqdm.auto import tqdm

from src.training.checkpointing import atomic_torch_save
from src.training.multitask_losses import MultiTaskPricingLoss


@dataclass(frozen=True, slots=True)
class MultiTaskTrainingConfig:
    """Training configuration shared by classifier and multi-task models."""

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
    show_progress: bool = True
    progress_update_interval: int = 25
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs and learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.early_stopping_patience <= 0 or self.scheduler_patience <= 0:
            raise ValueError("Patience values must be positive.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must lie between zero and one.")
        if self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta cannot be negative.")
        if self.progress_update_interval <= 0:
            raise ValueError("progress_update_interval must be positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def set_multitask_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tensor(
    batch: dict[str, object],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    """Move one tensor batch field to the selected device."""

    value = batch[key]

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a tensor.")

    return value.to(device, non_blocking=True)


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    """Return an autocast context compatible with CPU and CUDA."""

    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _progress_loader(
    loader: DataLoader,
    *,
    description: str,
    show_progress: bool,
):
    """Wrap a loader in a notebook- and terminal-compatible progress bar."""

    return tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )


def run_multitask_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: MultiTaskPricingLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
    mixed_precision: bool = True,
    description: str = "Multi-task",
    show_progress: bool = True,
    progress_update_interval: int = 25,
) -> dict[str, float]:
    """Run one visible multi-task training or evaluation epoch."""

    if progress_update_interval <= 0:
        raise ValueError("progress_update_interval must be positive.")

    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "regression_loss": 0.0,
        "classification_loss": 0.0,
        "residual_absolute_error": 0.0,
        "classification_correct": 0.0,
        "observations": 0.0,
    }

    progress = _progress_loader(
        loader,
        description=description,
        show_progress=show_progress,
    )

    for batch_index, batch in enumerate(progress, start=1):
        features = _tensor(batch, "features", device)
        residual_target = _tensor(batch, "residual_target", device)
        exercise_target = _tensor(batch, "exercise_target", device)

        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                raise TypeError("sample_weight must be a tensor.")
            sample_weight = sample_weight.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, mixed_precision):
                predicted_residual, exercise_logits = model(features)
                losses = loss_fn(
                    predicted_residual,
                    residual_target,
                    exercise_logits,
                    exercise_target,
                    sample_weight,
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

        for key in ("loss", "regression_loss", "classification_loss"):
            totals[key] += float(losses[key].detach().cpu()) * batch_size

        totals["residual_absolute_error"] += float(
            torch.abs(
                predicted_residual.detach() - residual_target
            ).sum().cpu()
        )

        predicted_label = (
            torch.sigmoid(exercise_logits.detach()) >= 0.5
        ).float()

        totals["classification_correct"] += float(
            (predicted_label == exercise_target).sum().cpu()
        )

        if (
            show_progress
            and (
                batch_index % progress_update_interval == 0
                or batch_index == len(loader)
            )
        ):
            observations = max(totals["observations"], 1.0)

            progress.set_postfix(
                loss=f"{totals['loss'] / observations:.5f}",
                reg=f"{totals['regression_loss'] / observations:.5f}",
                cls=f"{totals['classification_loss'] / observations:.5f}",
                mae=f"{totals['residual_absolute_error'] / observations:.5f}",
                acc=f"{totals['classification_correct'] / observations:.4f}",
            )

    observations = max(totals["observations"], 1.0)

    return {
        "loss": totals["loss"] / observations,
        "regression_loss": totals["regression_loss"] / observations,
        "classification_loss": totals["classification_loss"] / observations,
        "residual_mae": totals["residual_absolute_error"] / observations,
        "classification_accuracy": (
            totals["classification_correct"] / observations
        ),
        "observations": totals["observations"],
    }


def fit_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    loss_fn: MultiTaskPricingLoss,
    config: MultiTaskTrainingConfig,
    device: torch.device,
    checkpoint_path: str | Path,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Train the multi-task model with visible progress and checkpointing."""

    set_multitask_seed(config.seed)

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

    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(
            "Starting multi-task model training\n"
            f"Device: {device}\n"
            f"Train batches: {len(train_loader):,}\n"
            f"Validation batches: {len(validation_loader):,}\n"
            f"Epoch limit: {config.epochs:,}\n"
            f"Mixed precision: "
            f"{config.mixed_precision and device.type == 'cuda'}",
            flush=True,
        )

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        train_metrics = run_multitask_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=grad_scaler,
            gradient_clip_norm=config.gradient_clip_norm,
            mixed_precision=config.mixed_precision,
            description=f"Epoch {epoch:03d} training",
            show_progress=config.show_progress,
            progress_update_interval=config.progress_update_interval,
        )

        validation_metrics = run_multitask_epoch(
            model,
            validation_loader,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=config.mixed_precision,
            description=f"Epoch {epoch:03d} validation",
            show_progress=config.show_progress,
            progress_update_interval=config.progress_update_interval,
        )

        scheduler.step(validation_metrics["loss"])

        elapsed = time.perf_counter() - started

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_regression_loss": train_metrics["regression_loss"],
            "validation_regression_loss": validation_metrics["regression_loss"],
            "train_classification_loss": (
                train_metrics["classification_loss"]
            ),
            "validation_classification_loss": (
                validation_metrics["classification_loss"]
            ),
            "train_residual_mae": train_metrics["residual_mae"],
            "validation_residual_mae": validation_metrics["residual_mae"],
            "train_classification_accuracy": (
                train_metrics["classification_accuracy"]
            ),
            "validation_classification_accuracy": (
                validation_metrics["classification_accuracy"]
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(elapsed),
        }

        history.append(row)

        improved = (
            validation_metrics["loss"]
            < best_validation - config.min_delta
        )

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
                },
                checkpoint,
            )

        else:
            epochs_without_improvement += 1

        if config.verbose:
            status = "saved" if improved else (
                f"not improved "
                f"({epochs_without_improvement}/"
                f"{config.early_stopping_patience})"
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_metrics['loss']:.6f} | "
                f"Val loss: {validation_metrics['loss']:.6f} | "
                f"Train residual MAE: "
                f"{train_metrics['residual_mae']:.6f} | "
                f"Val residual MAE: "
                f"{validation_metrics['residual_mae']:.6f} | "
                f"Train accuracy: "
                f"{train_metrics['classification_accuracy']:.4f} | "
                f"Val accuracy: "
                f"{validation_metrics['classification_accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s | "
                f"{status}",
                flush=True,
            )

        if (
            not improved
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):
            if config.verbose:
                print(
                    f"Early stopping after epoch {epoch}.",
                    flush=True,
                )
            break

    if not checkpoint.exists():
        raise RuntimeError(
            f"No checkpoint was created at {checkpoint}. "
            "Inspect the training and validation losses."
        )

    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(saved["model_state_dict"])

    if config.verbose:
        print(
            f"Loaded best checkpoint from epoch {saved['epoch']} "
            f"with validation loss "
            f"{saved['validation_loss']:.6f}.",
            flush=True,
        )

    return pd.DataFrame(history)


@torch.inference_mode()
def predict_multitask_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    classification_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return aligned residual, price, and exercise predictions."""

    if not 0.0 < classification_threshold < 1.0:
        raise ValueError(
            "classification_threshold must lie between zero and one."
        )

    model.eval()
    model.to(device)

    rows: list[pd.DataFrame] = []

    for batch in loader:
        features = _tensor(batch, "features", device)

        predicted_residual, exercise_logits = model(features)

        normalized_european = _tensor(
            batch,
            "normalized_european",
            device,
        )

        normalized_intrinsic = _tensor(
            batch,
            "normalized_intrinsic",
            device,
        )

        normalized_floor = torch.maximum(
            normalized_european,
            normalized_intrinsic,
        )

        normalized_price = normalized_floor + predicted_residual
        probability = torch.sigmoid(exercise_logits)

        row_ids = batch["row_id"]

        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.detach().cpu().numpy()

        rows.append(
            pd.DataFrame(
                {
                    "sample_id": np.asarray(row_ids),
                    "predicted_residual": (
                        predicted_residual.cpu().numpy().reshape(-1)
                    ),
                    "predicted_normalized_american_price": (
                        normalized_price.cpu().numpy().reshape(-1)
                    ),
                    "exercise_logit": (
                        exercise_logits.cpu().numpy().reshape(-1)
                    ),
                    "exercise_probability": (
                        probability.cpu().numpy().reshape(-1)
                    ),
                    "exercise_prediction": (
                        probability >= classification_threshold
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool),
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "predicted_residual",
                "predicted_normalized_american_price",
                "exercise_logit",
                "exercise_probability",
                "exercise_prediction",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def run_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
    mixed_precision: bool = True,
    description: str = "Classifier",
    show_progress: bool = True,
    progress_update_interval: int = 25,
) -> dict[str, float]:
    """Run one visible exercise-classifier training or evaluation epoch."""

    if progress_update_interval <= 0:
        raise ValueError("progress_update_interval must be positive.")

    training = optimizer is not None
    model.train(training)

    loss_total = 0.0
    correct = 0.0
    observations = 0.0

    progress = _progress_loader(
        loader,
        description=description,
        show_progress=show_progress,
    )

    for batch_index, batch in enumerate(progress, start=1):
        features = _tensor(batch, "features", device)
        target = _tensor(batch, "exercise_target", device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, mixed_precision):
                logits = model(features)
                loss = loss_fn(logits, target)

            if training:
                assert optimizer is not None

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()

                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_norm,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()

                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_norm,
                        )

                    optimizer.step()

        batch_size = float(features.shape[0])
        observations += batch_size
        loss_total += float(loss.detach().cpu()) * batch_size

        predicted = (
            torch.sigmoid(logits.detach()) >= 0.5
        ).float()

        correct += float(
            (predicted == target).sum().cpu()
        )

        if (
            show_progress
            and (
                batch_index % progress_update_interval == 0
                or batch_index == len(loader)
            )
        ):
            denominator = max(observations, 1.0)

            progress.set_postfix(
                loss=f"{loss_total / denominator:.5f}",
                accuracy=f"{correct / denominator:.4f}",
            )

    denominator = max(observations, 1.0)

    return {
        "loss": loss_total / denominator,
        "accuracy": correct / denominator,
        "observations": observations,
    }


def fit_exercise_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    positive_class_weight: float,
    config: MultiTaskTrainingConfig,
    device: torch.device,
    checkpoint_path: str | Path,
) -> pd.DataFrame:
    """Train the exercise-only classifier with visible progress."""

    set_multitask_seed(config.seed)

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

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_class_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision and device.type == "cuda",
    )

    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(
            "Starting exercise-classifier training\n"
            f"Device: {device}\n"
            f"Train batches: {len(train_loader):,}\n"
            f"Validation batches: {len(validation_loader):,}\n"
            f"Positive class weight: {positive_class_weight:.6f}\n"
            f"Epoch limit: {config.epochs:,}\n"
            f"Mixed precision: "
            f"{config.mixed_precision and device.type == 'cuda'}",
            flush=True,
        )

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        train_metrics = run_classifier_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=grad_scaler,
            gradient_clip_norm=config.gradient_clip_norm,
            mixed_precision=config.mixed_precision,
            description=f"Epoch {epoch:03d} training",
            show_progress=config.show_progress,
            progress_update_interval=config.progress_update_interval,
        )

        validation_metrics = run_classifier_epoch(
            model,
            validation_loader,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=config.mixed_precision,
            description=f"Epoch {epoch:03d} validation",
            show_progress=config.show_progress,
            progress_update_interval=config.progress_update_interval,
        )

        scheduler.step(validation_metrics["loss"])

        elapsed = time.perf_counter() - started

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(elapsed),
            }
        )

        improved = (
            validation_metrics["loss"]
            < best_validation - config.min_delta
        )

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
                    "positive_class_weight": float(positive_class_weight),
                },
                checkpoint,
            )

        else:
            epochs_without_improvement += 1

        if config.verbose:
            status = "saved" if improved else (
                f"not improved "
                f"({epochs_without_improvement}/"
                f"{config.early_stopping_patience})"
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_metrics['loss']:.6f} | "
                f"Val loss: {validation_metrics['loss']:.6f} | "
                f"Train accuracy: {train_metrics['accuracy']:.4f} | "
                f"Val accuracy: {validation_metrics['accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s | "
                f"{status}",
                flush=True,
            )

        if (
            not improved
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):
            if config.verbose:
                print(
                    f"Early stopping after epoch {epoch}.",
                    flush=True,
                )
            break

    if not checkpoint.exists():
        raise RuntimeError(
            f"No checkpoint was created at {checkpoint}. "
            "Inspect the training and validation losses."
        )

    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(saved["model_state_dict"])

    if config.verbose:
        print(
            f"Loaded best checkpoint from epoch {saved['epoch']} "
            f"with validation loss "
            f"{saved['validation_loss']:.6f}.",
            flush=True,
        )

    return pd.DataFrame(history)


@torch.inference_mode()
def predict_exercise_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Return aligned exercise probabilities from the classifier baseline."""

    model.eval()
    model.to(device)

    rows: list[pd.DataFrame] = []

    for batch in loader:
        features = _tensor(batch, "features", device)
        logits = model(features)
        probability = torch.sigmoid(logits)

        row_ids = batch["row_id"]

        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.detach().cpu().numpy()

        rows.append(
            pd.DataFrame(
                {
                    "sample_id": np.asarray(row_ids),
                    "exercise_logit": (
                        logits.cpu().numpy().reshape(-1)
                    ),
                    "exercise_probability": (
                        probability.cpu().numpy().reshape(-1)
                    ),
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "exercise_logit",
                "exercise_probability",
            ]
        )

    return pd.concat(rows, ignore_index=True)


__all__ = [
    "MultiTaskTrainingConfig",
    "fit_exercise_classifier",
    "fit_multitask_model",
    "predict_exercise_classifier",
    "predict_multitask_model",
    "run_classifier_epoch",
    "run_multitask_epoch",
    "set_multitask_seed",
]
