"""Synthetic American put dataset generation.

The generator fixes strike at a configurable positive value and samples the
spot/strike ratio. This uses the homogeneity of vanilla option prices and avoids
introducing a redundant absolute scale into the first deep-learning dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing.black_scholes import black_scholes_put_price
from src.pricing.binomial_tree import crr_option_diagnostics


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class ParameterRanges:
    """Parameter domain used for synthetic American put generation."""

    moneyness: tuple[float, float] = (0.50, 1.50)
    time_to_maturity: tuple[float, float] = (7.0 / 365.0, 2.0)
    volatility: tuple[float, float] = (0.05, 0.80)
    risk_free_rate: tuple[float, float] = (0.00, 0.10)
    dividend_yield: tuple[float, float] = (0.00, 0.08)

    def __post_init__(self) -> None:
        for name in (
            "moneyness",
            "time_to_maturity",
            "volatility",
            "risk_free_rate",
            "dividend_yield",
        ):
            lower, upper = getattr(self, name)
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{name} bounds must be finite.")
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be below upper bound.")
        if self.moneyness[0] <= 0.0:
            raise ValueError("Moneyness must remain positive.")
        if self.time_to_maturity[0] <= 0.0:
            raise ValueError("Minimum time to maturity must be positive.")
        if self.volatility[0] <= 0.0:
            raise ValueError("Minimum volatility must be positive.")


def _latin_hypercube(
    *,
    n_samples: int,
    n_dimensions: int,
    seed: int,
) -> np.ndarray:
    """Generate a NumPy-only randomized Latin hypercube in [0, 1]."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_dimensions <= 0:
        raise ValueError("n_dimensions must be positive.")

    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, n_dimensions), dtype=np.float64)
    for column in range(n_dimensions):
        points = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        samples[:, column] = rng.permutation(points)
    return samples


def _scale_unit_interval(
    unit_values: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    lower, upper = bounds
    return lower + unit_values * (upper - lower)


def generate_american_put_dataset(
    *,
    n_samples: int,
    tree_steps: int,
    seed: int = 42,
    strike: float = 100.0,
    ranges: ParameterRanges | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Generate a synthetic American put pricing dataset.

    Latin hypercube sampling improves coverage of the five-dimensional parameter
    domain relative to independent random draws of the same pilot size.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if tree_steps <= 0:
        raise ValueError("tree_steps must be positive.")
    if strike <= 0.0:
        raise ValueError("strike must be positive.")

    domain = ranges or ParameterRanges()
    unit = _latin_hypercube(
        n_samples=n_samples,
        n_dimensions=5,
        seed=seed,
    )

    moneyness = _scale_unit_interval(unit[:, 0], domain.moneyness)
    maturities = _scale_unit_interval(unit[:, 1], domain.time_to_maturity)
    volatilities = _scale_unit_interval(unit[:, 2], domain.volatility)
    rates = _scale_unit_interval(unit[:, 3], domain.risk_free_rate)
    dividends = _scale_unit_interval(unit[:, 4], domain.dividend_yield)

    records: list[dict[str, float | int | bool]] = []
    for index in range(n_samples):
        spot = float(strike * moneyness[index])
        time_to_maturity = float(maturities[index])
        volatility = float(volatilities[index])
        risk_free_rate = float(rates[index])
        dividend_yield = float(dividends[index])

        parameters = {
            "spot": spot,
            "strike": float(strike),
            "time_to_maturity": time_to_maturity,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "volatility": volatility,
        }

        european_price = black_scholes_put_price(**parameters)
        american = crr_option_diagnostics(
            **parameters,
            steps=tree_steps,
            option_type="put",
            exercise_style="american",
        )

        # A finite CRR tree can oscillate slightly around the analytical
        # European value. The true American contract cannot be worth less than
        # either the European contract or intrinsic value, so the supervised
        # label receives a transparent no-arbitrage floor repair. The raw tree
        # value and adjustment are retained for auditability.
        raw_american_price = american.price
        american_price = max(
            raw_american_price,
            european_price,
            american.intrinsic_value,
        )
        floor_adjustment = american_price - raw_american_price
        premium = american_price - european_price

        records.append(
            {
                "sample_id": index,
                "spot": spot,
                "strike": float(strike),
                "moneyness": spot / strike,
                "log_moneyness": float(np.log(spot / strike)),
                "time_to_maturity": time_to_maturity,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "volatility": volatility,
                "intrinsic_value": american.intrinsic_value,
                "continuation_value": american.continuation_value,
                "european_price": european_price,
                "raw_american_price": raw_american_price,
                "american_price": american_price,
                "pricing_floor_adjustment": floor_adjustment,
                "early_exercise_premium": premium,
                "normalized_european_price": european_price / strike,
                "normalized_american_price": american_price / strike,
                "normalized_early_exercise_premium": premium / strike,
                "exercise_now": american.exercise_now,
                "tree_steps": int(tree_steps),
            }
        )

        if progress_callback is not None:
            progress_callback(index + 1, n_samples)

    return pd.DataFrame.from_records(records)


def save_generated_dataset(
    dataset: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save a generated dataset as CSV or Parquet based on the file suffix."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        dataset.to_csv(output_path, index=False)
    elif suffix == ".parquet":
        try:
            dataset.to_parquet(output_path, index=False)
        except ImportError as error:
            raise ImportError(
                "Saving Parquet requires pyarrow or fastparquet. "
                "Use a .csv path or install pyarrow."
            ) from error
    else:
        raise ValueError("Dataset path must end in .csv or .parquet.")

    return output_path


__all__ = [
    "ParameterRanges",
    "generate_american_put_dataset",
    "save_generated_dataset",
]
