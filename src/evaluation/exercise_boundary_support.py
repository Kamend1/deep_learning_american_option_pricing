"""Notebook-level support for exercise-boundary analysis.

The functions here are reusable evaluation utilities. All models, feature
transformers, column definitions, and devices are passed explicitly so the
module does not depend on notebook globals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.pricing.binomial_tree import crr_option_diagnostics
from src.pricing.black_scholes import black_scholes_put_price


def boundary_decision_diagnostics(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    threshold: float,
    bands: Sequence[float] = (0.001, 0.005, 0.010),
) -> pd.DataFrame:
    """Add class counts and economic regret to boundary-band metrics."""

    required = [
        "boundary_distance_normalized",
        "exercise_now",
        "intrinsic_value",
        "continuation_value",
        probability_column,
    ]
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Frame is missing columns: {missing}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must lie strictly between zero and one."
        )

    rows: list[dict[str, float | int]] = []

    for raw_band in bands:
        band = float(raw_band)
        if band <= 0.0:
            raise ValueError("Boundary bands must be positive.")

        subset = frame.loc[
            frame["boundary_distance_normalized"] <= band
        ]

        actual = subset[
            "exercise_now"
        ].astype(bool).to_numpy()
        predicted = (
            subset[probability_column].to_numpy(
                dtype=np.float64
            )
            >= threshold
        )
        wrong = actual != predicted

        exercise_count = int(actual.sum())
        continuation_count = int(
            len(actual) - exercise_count
        )
        false_exercise = int(
            ((~actual) & predicted).sum()
        )
        missed_exercise = int(
            (actual & (~predicted)).sum()
        )

        decision_gap = np.abs(
            subset["intrinsic_value"].to_numpy(
                dtype=np.float64
            )
            - subset["continuation_value"].to_numpy(
                dtype=np.float64
            )
        )
        regret = np.where(wrong, decision_gap, 0.0)
        wrong_regret = regret[wrong]

        rows.append(
            {
                "boundary_band": band,
                "continuation_observations": continuation_count,
                "exercise_observations": exercise_count,
                "false_exercise_count": false_exercise,
                "missed_exercise_count": missed_exercise,
                "decision_errors": int(wrong.sum()),
                "mean_regret_all": (
                    float(regret.mean())
                    if len(regret)
                    else np.nan
                ),
                "mean_regret_when_wrong": (
                    float(wrong_regret.mean())
                    if len(wrong_regret)
                    else 0.0
                ),
                "maximum_regret": (
                    float(wrong_regret.max())
                    if len(wrong_regret)
                    else 0.0
                ),
                "total_regret": float(regret.sum()),
            }
        )

    return pd.DataFrame(rows)


def build_boundary_slice(
    *,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    feature_scaler: Any,
    feature_columns: Sequence[str],
    exercise_classifier: nn.Module,
    multitask_model: nn.Module,
    device: str | torch.device,
    strike: float = 100.0,
    steps: int = 250,
    points: int = 241,
) -> pd.DataFrame:
    """Build one CRR boundary slice and attach neural probabilities."""

    if strike <= 0.0:
        raise ValueError("strike must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if points < 2:
        raise ValueError("points must be at least two.")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    resolved_device = torch.device(device)
    moneyness_grid = np.linspace(0.40, 1.20, points)
    rows: list[dict[str, float | bool]] = []

    for moneyness in moneyness_grid:
        spot = strike * moneyness
        diagnostic = crr_option_diagnostics(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            steps=steps,
            option_type="put",
            exercise_style="american",
        )
        european = black_scholes_put_price(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )
        intrinsic = max(strike - spot, 0.0)
        continuation = diagnostic.continuation_value
        financial_floor = max(european, intrinsic)
        signed_margin = (
            intrinsic - continuation
        ) / strike

        rows.append(
            {
                "moneyness": float(moneyness),
                "log_moneyness": float(np.log(moneyness)),
                "time_to_maturity": float(time_to_maturity),
                "risk_free_rate": float(risk_free_rate),
                "dividend_yield": float(dividend_yield),
                "volatility": float(volatility),
                "intrinsic_value": float(intrinsic),
                "continuation_value": float(continuation),
                "signed_boundary_margin": float(signed_margin),
                "boundary_distance_normalized": float(
                    abs(signed_margin)
                ),
                "exercise_now": bool(
                    diagnostic.exercise_now
                ),
                "normalized_european_price": float(
                    european / strike
                ),
                "normalized_intrinsic_value": float(
                    intrinsic / strike
                ),
                "normalized_financial_floor": float(
                    financial_floor / strike
                ),
                "normalized_american_price": float(
                    diagnostic.price / strike
                ),
                "normalized_floor_residual": float(
                    max(
                        diagnostic.price - financial_floor,
                        0.0,
                    )
                    / strike
                ),
            }
        )

    frame = pd.DataFrame(rows)
    feature_values = feature_scaler.transform(
        frame.loc[
            :,
            list(feature_columns),
        ].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    features = torch.from_numpy(
        feature_values
    ).to(resolved_device)

    exercise_classifier = exercise_classifier.to(
        resolved_device
    ).eval()
    multitask_model = multitask_model.to(
        resolved_device
    ).eval()

    with torch.inference_mode():
        classifier_probability = torch.sigmoid(
            exercise_classifier(features)
        ).cpu().numpy().reshape(-1)

        _, exercise_logits = multitask_model(features)
        multitask_probability = torch.sigmoid(
            exercise_logits
        ).cpu().numpy().reshape(-1)

    frame[
        "classifier_probability"
    ] = classifier_probability
    frame[
        "multitask_probability"
    ] = multitask_probability

    return frame


__all__ = [
    "boundary_decision_diagnostics",
    "build_boundary_slice",
]
