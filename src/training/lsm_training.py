"""Training and evaluation of amortized neural Longstaff–Schwartz policies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.neural_longstaff_schwartz import (
    ContinuationNetworkConfig,
    ContinuationValueNetwork,
    ContractPathBatch,
    FeatureStandardizer,
    NeuralContinuationStep,
    NeuralLSMPolicy,
    build_continuation_features,
)
from src.pricing.longstaff_schwartz import LSMPriceResult


@dataclass(frozen=True)
class NeuralLSMTrainingConfig:
    """Controls for time-indexed continuation-network training."""

    network: ContinuationNetworkConfig = ContinuationNetworkConfig()
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    minimum_samples_per_step: int = 64
    maximum_samples_per_step: int = 250_000
    gradient_clip_norm: float | None = 1.0
    warm_start_from_later_step: bool = True
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    epoch_report_interval: int = 5

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer parameters.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.minimum_samples_per_step < 2:
            raise ValueError("minimum_samples_per_step must be at least two.")
        if self.maximum_samples_per_step < self.minimum_samples_per_step:
            raise ValueError(
                "maximum_samples_per_step cannot be below the minimum."
            )
        if self.epoch_report_interval <= 0:
            raise ValueError("epoch_report_interval must be positive.")


@dataclass
class _BackwardState:
    cashflows: np.ndarray
    exercise_indices: np.ndarray


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def set_lsm_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_contract_separation(
    training_batches: Iterable[ContractPathBatch],
    validation_batches: Iterable[ContractPathBatch],
) -> None:
    """Ensure contract IDs do not leak between policy training and validation."""

    training_ids = {batch.contract.contract_id for batch in training_batches}
    validation_ids = {batch.contract.contract_id for batch in validation_batches}
    overlap = training_ids & validation_ids
    if overlap:
        raise ValueError(f"Contract leakage detected: {sorted(overlap)}.")


def _validate_batch_collection(batches: list[ContractPathBatch]) -> int:
    if not batches:
        raise ValueError("At least one contract batch is required.")
    n_steps = batches[0].n_steps
    if any(batch.n_steps != n_steps for batch in batches):
        raise ValueError("All contract batches must use the same step count.")
    ids = [batch.contract.contract_id for batch in batches]
    if len(ids) != len(set(ids)):
        raise ValueError("Contract IDs must be unique within a collection.")
    return n_steps


def _initial_backward_states(
    batches: list[ContractPathBatch],
) -> dict[str, _BackwardState]:
    states: dict[str, _BackwardState] = {}
    for batch in batches:
        payoff = np.maximum(batch.contract.strike - batch.paths[:, -1], 0.0)
        states[batch.contract.contract_id] = _BackwardState(
            cashflows=payoff.astype(np.float64),
            exercise_indices=np.full(
                batch.n_paths,
                batch.n_steps,
                dtype=np.int32,
            ),
        )
    return states


def _collect_step_samples(
    batches: list[ContractPathBatch],
    states: dict[str, _BackwardState],
    *,
    step_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for batch in batches:
        contract = batch.contract
        state = states[contract.contract_id]
        spots = batch.paths[:, step_index]
        intrinsic = np.maximum(contract.strike - spots, 0.0)
        in_the_money = intrinsic > 0.0
        if not np.any(in_the_money):
            continue

        delta_t = contract.time_to_maturity / batch.n_steps
        future_discount = np.exp(
            -contract.risk_free_rate
            * delta_t
            * (state.exercise_indices[in_the_money] - step_index)
        )
        targets = (
            state.cashflows[in_the_money]
            * future_discount
            / contract.strike
        )
        time_remaining = contract.time_to_maturity * (
            1.0 - step_index / batch.n_steps
        )
        features = build_continuation_features(
            spots[in_the_money],
            contract=contract,
            time_remaining=time_remaining,
        )
        feature_parts.append(features)
        target_parts.append(targets.astype(np.float64))

    if not feature_parts:
        return (
            np.empty((0, 5), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    return np.concatenate(feature_parts), np.concatenate(target_parts)


def _subsample(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= maximum:
        return features, targets
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(features), size=maximum, replace=False)
    return features[selected], targets[selected]


def _fit_one_network(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    config: NeuralLSMTrainingConfig,
    standardizer: FeatureStandardizer,
    initial_state_dict: dict[str, torch.Tensor] | None,
    step_index: int,
) -> tuple[ContinuationValueNetwork, list[dict[str, float | int]]]:
    device = torch.device(config.device)
    model = ContinuationValueNetwork(config.network).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    train_x = torch.from_numpy(standardizer.transform(training_features))
    train_y = torch.from_numpy(training_targets.astype(np.float32))
    val_x = torch.from_numpy(standardizer.transform(validation_features))
    val_y = torch.from_numpy(validation_targets.astype(np.float32))

    generator = torch.Generator().manual_seed(config.seed + step_index)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(config.batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    best_state = deepcopy(model.state_dict())
    best_validation = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    _log(
        f"Step {step_index:03d} network | "
        f"train samples: {len(train_x):,} | "
        f"validation samples: {len(val_x):,} | "
        f"batches per epoch: {len(loader):,} | "
        f"device: {device}",
        enabled=config.verbose,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        observation_count = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()

            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )

            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            observation_count += len(batch_x)

        train_loss = running_loss / max(observation_count, 1)
        model.eval()
        with torch.no_grad():
            validation_predictions = model(val_x.to(device))
            validation_loss = float(
                loss_fn(
                    validation_predictions,
                    val_y.to(device),
                ).item()
            )

        history.append(
            {
                "step_index": step_index,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        improved = validation_loss < best_validation - 1e-10
        if improved:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            status = "best"
        else:
            epochs_without_improvement += 1
            status = (
                f"no improvement "
                f"{epochs_without_improvement}/{config.patience}"
            )

        should_report = (
            epoch == 1
            or epoch % config.epoch_report_interval == 0
            or epoch == config.epochs
            or epochs_without_improvement >= config.patience
        )
        if should_report:
            _log(
                f"Step {step_index:03d} | "
                f"Epoch {epoch:03d}/{config.epochs:03d} | "
                f"train loss: {train_loss:.8f} | "
                f"validation loss: {validation_loss:.8f} | "
                f"best validation: {best_validation:.8f} | "
                f"time: {time.perf_counter() - epoch_started:.2f}s | "
                f"{status}",
                enabled=config.verbose,
            )

        if epochs_without_improvement >= config.patience:
            _log(
                f"Step {step_index:03d} early stopping at epoch {epoch}.",
                enabled=config.verbose,
            )
            break

    model.load_state_dict(best_state)
    _log(
        f"Step {step_index:03d} network complete | "
        f"epochs run: {len(history)} | "
        f"best validation loss: {best_validation:.8f}",
        enabled=config.verbose,
    )
    return model, history


def _predict_normalized_continuation(
    model: ContinuationValueNetwork,
    standardizer: FeatureStandardizer,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    model.eval()
    transformed = standardizer.transform(features)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(transformed), batch_size):
            tensor = torch.from_numpy(
                transformed[start : start + batch_size]
            ).to(device)
            outputs.append(model(tensor).detach().cpu().numpy())
    return (
        np.concatenate(outputs)
        if outputs
        else np.empty(0, dtype=np.float32)
    )


def _update_backward_states(
    batches: list[ContractPathBatch],
    states: dict[str, _BackwardState],
    *,
    step_index: int,
    model: ContinuationValueNetwork,
    standardizer: FeatureStandardizer,
    device: torch.device,
) -> None:
    for batch in batches:
        contract = batch.contract
        state = states[contract.contract_id]
        spots = batch.paths[:, step_index]
        intrinsic = np.maximum(contract.strike - spots, 0.0)
        in_the_money = intrinsic > 0.0
        if not np.any(in_the_money):
            continue

        time_remaining = contract.time_to_maturity * (
            1.0 - step_index / batch.n_steps
        )
        features = build_continuation_features(
            spots[in_the_money],
            contract=contract,
            time_remaining=time_remaining,
        )
        continuation = _predict_normalized_continuation(
            model,
            standardizer,
            features,
            device=device,
        ) * contract.strike
        exercise_local = intrinsic[in_the_money] > continuation
        path_indices = np.flatnonzero(in_the_money)[exercise_local]
        state.cashflows[path_indices] = intrinsic[path_indices]
        state.exercise_indices[path_indices] = step_index


def fit_neural_lsm_policy(
    training_batches: list[ContractPathBatch],
    validation_batches: list[ContractPathBatch],
    *,
    config: NeuralLSMTrainingConfig | None = None,
) -> tuple[NeuralLSMPolicy, pd.DataFrame]:
    """Fit one pooled continuation network per exercise index."""

    config = config or NeuralLSMTrainingConfig()
    validate_contract_separation(training_batches, validation_batches)
    n_steps = _validate_batch_collection(training_batches)
    if _validate_batch_collection(validation_batches) != n_steps:
        raise ValueError(
            "Training and validation batches need equal step counts."
        )
    set_lsm_seed(config.seed)

    training_states = _initial_backward_states(training_batches)
    validation_states = _initial_backward_states(validation_batches)
    policy_steps: dict[int, NeuralContinuationStep] = {}
    history_rows: list[dict[str, float | int]] = []
    later_state: dict[str, torch.Tensor] | None = None
    device = torch.device(config.device)
    total_started = time.perf_counter()
    candidate_steps = list(range(n_steps - 1, 0, -1))

    _log(
        "Starting neural Longstaff-Schwartz training\n"
        f"Device: {device}\n"
        f"Training contracts: {len(training_batches):,}\n"
        f"Validation contracts: {len(validation_batches):,}\n"
        f"Exercise steps considered: {len(candidate_steps):,}\n"
        f"Maximum epochs per step: {config.epochs:,}\n"
        f"Epoch report interval: {config.epoch_report_interval:,}",
        enabled=config.verbose,
    )

    for position, step_index in enumerate(candidate_steps, start=1):
        step_started = time.perf_counter()
        _log(
            f"\nExercise step {step_index:03d} | "
            f"progress {position}/{len(candidate_steps)} | "
            "collecting samples...",
            enabled=config.verbose,
        )

        train_x, train_y = _collect_step_samples(
            training_batches,
            training_states,
            step_index=step_index,
        )
        val_x, val_y = _collect_step_samples(
            validation_batches,
            validation_states,
            step_index=step_index,
        )

        _log(
            f"Exercise step {step_index:03d} | "
            f"raw training samples: {len(train_x):,} | "
            f"raw validation samples: {len(val_x):,}",
            enabled=config.verbose,
        )

        if (
            len(train_x) < config.minimum_samples_per_step
            or len(val_x) < 2
        ):
            _log(
                f"Exercise step {step_index:03d} skipped | "
                "insufficient in-the-money samples.",
                enabled=config.verbose,
            )
            continue

        raw_train_count = len(train_x)
        raw_val_count = len(val_x)
        train_x, train_y = _subsample(
            train_x,
            train_y,
            maximum=config.maximum_samples_per_step,
            seed=config.seed + step_index,
        )
        val_x, val_y = _subsample(
            val_x,
            val_y,
            maximum=max(config.maximum_samples_per_step // 4, 2),
            seed=config.seed + 10_000 + step_index,
        )

        if len(train_x) != raw_train_count or len(val_x) != raw_val_count:
            _log(
                f"Exercise step {step_index:03d} | "
                f"samples after cap: {len(train_x):,} training, "
                f"{len(val_x):,} validation",
                enabled=config.verbose,
            )

        standardizer = FeatureStandardizer.fit(train_x)
        initial_state = (
            later_state
            if config.warm_start_from_later_step
            else None
        )
        model, step_history = _fit_one_network(
            train_x,
            train_y,
            val_x,
            val_y,
            config=config,
            standardizer=standardizer,
            initial_state_dict=initial_state,
            step_index=step_index,
        )

        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        later_state = state_dict
        policy_steps[step_index] = NeuralContinuationStep(
            step_index=step_index,
            standardizer=standardizer,
            network_config=config.network,
            state_dict=state_dict,
            n_training_samples=len(train_x),
            n_validation_samples=len(val_x),
        )
        history_rows.extend(step_history)

        _log(
            f"Exercise step {step_index:03d} | "
            "updating training backward states...",
            enabled=config.verbose,
        )
        _update_backward_states(
            training_batches,
            training_states,
            step_index=step_index,
            model=model,
            standardizer=standardizer,
            device=device,
        )

        _log(
            f"Exercise step {step_index:03d} | "
            "updating validation backward states...",
            enabled=config.verbose,
        )
        _update_backward_states(
            validation_batches,
            validation_states,
            step_index=step_index,
            model=model,
            standardizer=standardizer,
            device=device,
        )

        _log(
            f"Exercise step {step_index:03d} complete | "
            f"elapsed: {time.perf_counter() - step_started:.2f}s | "
            f"trained networks: {len(policy_steps):,}",
            enabled=config.verbose,
        )

    _log(
        "\nNeural Longstaff-Schwartz training complete\n"
        f"Continuation networks trained: {len(policy_steps):,}\n"
        f"History rows: {len(history_rows):,}\n"
        f"Total runtime: {time.perf_counter() - total_started:.2f}s",
        enabled=config.verbose,
    )

    return (
        NeuralLSMPolicy(n_steps=n_steps, steps=policy_steps),
        pd.DataFrame(history_rows),
    )


def evaluate_neural_lsm_policy(
    policy: NeuralLSMPolicy,
    batch: ContractPathBatch,
    *,
    device: str | torch.device = "cpu",
    confidence_level: float = 0.95,
) -> LSMPriceResult:
    """Evaluate a neural stopping policy on independent paths."""

    if batch.n_steps != policy.n_steps:
        raise ValueError("Path step count does not match neural policy.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    device = torch.device(device)
    contract = batch.contract
    n_paths = batch.n_paths
    active = np.ones(n_paths, dtype=bool)
    exercise_indices = np.full(
        n_paths,
        policy.n_steps,
        dtype=np.int32,
    )
    payoffs = np.maximum(
        contract.strike - batch.paths[:, -1],
        0.0,
    )

    for step_index in range(1, policy.n_steps):
        policy_step = policy.steps.get(step_index)
        if policy_step is None:
            continue

        candidate_indices = np.flatnonzero(active)
        if candidate_indices.size == 0:
            break

        spots = batch.paths[candidate_indices, step_index]
        intrinsic = np.maximum(contract.strike - spots, 0.0)
        itm = intrinsic > 0.0
        if not np.any(itm):
            continue

        itm_indices = candidate_indices[itm]
        time_remaining = contract.time_to_maturity * (
            1.0 - step_index / policy.n_steps
        )
        features = build_continuation_features(
            spots[itm],
            contract=contract,
            time_remaining=time_remaining,
        )
        model = policy_step.build_model(device)
        continuation = _predict_normalized_continuation(
            model,
            policy_step.standardizer,
            features,
            device=device,
        ) * contract.strike
        exercise = intrinsic[itm] > continuation
        exercise_path_indices = itm_indices[exercise]

        if exercise_path_indices.size:
            payoffs[exercise_path_indices] = np.maximum(
                contract.strike
                - batch.paths[exercise_path_indices, step_index],
                0.0,
            )
            exercise_indices[exercise_path_indices] = step_index
            active[exercise_path_indices] = False

    delta_t = contract.time_to_maturity / policy.n_steps
    discounted_payoffs = payoffs * np.exp(
        -contract.risk_free_rate * delta_t * exercise_indices
    )
    intrinsic_zero = max(contract.strike - contract.spot, 0.0)
    continuation_zero = float(np.mean(discounted_payoffs))
    time_zero_exercise = (
        intrinsic_zero >= continuation_zero
        and intrinsic_zero > 0.0
    )

    if time_zero_exercise:
        discounted_payoffs = np.full(
            n_paths,
            intrinsic_zero,
            dtype=np.float64,
        )
        exercise_indices = np.zeros(n_paths, dtype=np.int32)

    price = float(np.mean(discounted_payoffs))
    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / math.sqrt(n_paths)
    )
    z_value = 1.959963984540054
    exercise_times = exercise_indices * delta_t

    return LSMPriceResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_low=price - z_value * standard_error,
        confidence_interval_high=price + z_value * standard_error,
        n_paths=n_paths,
        discounted_payoffs=discounted_payoffs,
        exercise_indices=exercise_indices,
        exercise_times=exercise_times.astype(np.float64),
        exercised_early_rate=float(
            np.mean(exercise_indices < policy.n_steps)
        ),
        intrinsic_at_time_zero=float(intrinsic_zero),
        time_zero_exercise=time_zero_exercise,
    )


def save_neural_lsm_policy(policy: NeuralLSMPolicy, path: str) -> None:
    policy.save(path)


def load_neural_lsm_policy(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> NeuralLSMPolicy:
    return NeuralLSMPolicy.load(path, map_location=map_location)


__all__ = [
    "NeuralLSMTrainingConfig",
    "evaluate_neural_lsm_policy",
    "fit_neural_lsm_policy",
    "load_neural_lsm_policy",
    "save_neural_lsm_policy",
    "set_lsm_seed",
    "validate_contract_separation",
]
