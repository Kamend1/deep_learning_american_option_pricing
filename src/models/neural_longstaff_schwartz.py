"""Neural continuation-value models for amortized Longstaff–Schwartz."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.pricing.simulation import GBMContract


@dataclass(frozen=True)
class ContinuationNetworkConfig:
    """Architecture of a continuation-value network."""

    input_dim: int = 5
    hidden_dims: tuple[int, ...] = (64, 64, 32)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class ContinuationValueNetwork(nn.Module):
    """Small MLP returning a non-negative normalized continuation value."""

    def __init__(self, config: ContinuationNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContinuationNetworkConfig()
        layers: list[nn.Module] = []
        previous = self.config.input_dim
        for width in self.config.hidden_dims:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.SiLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return shape ``(batch,)`` normalized continuation estimates."""

        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected features with shape (batch, {self.config.input_dim}); "
                f"received {tuple(features.shape)}."
            )
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Serializable feature standardization parameters."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional array.")
        if not np.isfinite(features).all():
            raise ValueError("features contains non-finite values.")
        mean = features.mean(axis=0)
        scale = features.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        return ((features - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
        )


@dataclass(frozen=True)
class ContractPathBatch:
    """Paths and metadata for one contract."""

    contract: GBMContract
    paths: np.ndarray

    def __post_init__(self) -> None:
        paths = np.asarray(self.paths)
        if paths.ndim != 2 or paths.shape[1] < 2:
            raise ValueError("paths must be a two-dimensional path matrix.")
        if paths.shape[0] < 2:
            raise ValueError("at least two paths are required.")
        if not np.isfinite(paths).all() or np.any(paths <= 0.0):
            raise ValueError("paths must contain finite positive values.")
        if not np.allclose(paths[:, 0], self.contract.spot):
            raise ValueError("the first path column must equal contract spot.")

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.paths.shape[1] - 1)


@dataclass
class NeuralContinuationStep:
    """Serialized network and scaler for one exercise index."""

    step_index: int
    standardizer: FeatureStandardizer
    network_config: ContinuationNetworkConfig
    state_dict: dict[str, torch.Tensor]
    n_training_samples: int
    n_validation_samples: int

    def build_model(self, device: str | torch.device = "cpu") -> ContinuationValueNetwork:
        model = ContinuationValueNetwork(self.network_config)
        model.load_state_dict(self.state_dict)
        return model.to(device).eval()


@dataclass
class NeuralLSMPolicy:
    """Time-indexed amortized neural continuation policy."""

    n_steps: int
    steps: dict[int, NeuralContinuationStep]
    feature_names: tuple[str, ...] = (
        "log_moneyness",
        "time_remaining",
        "risk_free_rate",
        "dividend_yield",
        "volatility",
    )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_steps": self.n_steps,
            "feature_names": self.feature_names,
            "steps": {
                index: {
                    "step_index": item.step_index,
                    "standardizer": item.standardizer.to_dict(),
                    "network_config": asdict(item.network_config),
                    "state_dict": {
                        key: value.detach().cpu() for key, value in item.state_dict.items()
                    },
                    "n_training_samples": item.n_training_samples,
                    "n_validation_samples": item.n_validation_samples,
                }
                for index, item in self.steps.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "NeuralLSMPolicy":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        steps: dict[int, NeuralContinuationStep] = {}
        for raw_index, item in payload["steps"].items():
            index = int(raw_index)
            config_payload = item["network_config"]
            config_payload["hidden_dims"] = tuple(config_payload["hidden_dims"])
            steps[index] = NeuralContinuationStep(
                step_index=index,
                standardizer=FeatureStandardizer.from_dict(item["standardizer"]),
                network_config=ContinuationNetworkConfig(**config_payload),
                state_dict=item["state_dict"],
                n_training_samples=int(item["n_training_samples"]),
                n_validation_samples=int(item["n_validation_samples"]),
            )
        return cls(
            n_steps=int(payload["n_steps"]),
            steps=steps,
            feature_names=tuple(payload["feature_names"]),
        )


def build_continuation_features(
    spot_values: np.ndarray,
    *,
    contract: GBMContract,
    time_remaining: float,
) -> np.ndarray:
    """Build the five neural continuation features."""

    spot_values = np.asarray(spot_values, dtype=np.float64).reshape(-1)
    if np.any(spot_values <= 0.0) or not np.isfinite(spot_values).all():
        raise ValueError("spot_values must contain finite positive values.")
    if time_remaining < 0.0:
        raise ValueError("time_remaining cannot be negative.")
    return np.column_stack(
        [
            np.log(spot_values / contract.strike),
            np.full_like(spot_values, time_remaining),
            np.full_like(spot_values, contract.risk_free_rate),
            np.full_like(spot_values, contract.dividend_yield),
            np.full_like(spot_values, contract.volatility),
        ]
    )


__all__ = [
    "ContinuationNetworkConfig",
    "ContinuationValueNetwork",
    "ContractPathBatch",
    "FeatureStandardizer",
    "NeuralContinuationStep",
    "NeuralLSMPolicy",
    "build_continuation_features",
]
