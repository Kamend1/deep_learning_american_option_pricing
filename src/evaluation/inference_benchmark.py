"""Reusable raw forward-pass benchmarks for PyTorch models."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def benchmark_forward_pass(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    repeats: int = 5,
    warmup: int = 0,
) -> dict[str, float | int]:
    """Measure median model-forward latency on one in-memory tensor."""

    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor.")
    if features.ndim == 0 or features.shape[0] == 0:
        raise ValueError("features must contain observations.")

    resolved_device = (
        torch.device(device)
        if device is not None
        else features.device
    )
    model = model.to(resolved_device)
    feature_tensor = features.to(resolved_device)
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(feature_tensor)

        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

        timings: list[float] = []

        for _ in range(repeats):
            started = time.perf_counter()
            model(feature_tensor)

            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)

            timings.append(
                time.perf_counter() - started
            )

    median_seconds = float(np.median(timings))
    observations = int(feature_tensor.shape[0])

    return {
        "observations": observations,
        "median_seconds": median_seconds,
        "observations_per_second": float(
            observations / median_seconds
        ),
    }


__all__ = ["benchmark_forward_pass"]
