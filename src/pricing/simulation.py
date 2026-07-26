"""Risk-neutral geometric Brownian motion simulation utilities.

The functions in this module are deliberately independent of the neural-network
implementation. They support both classical and neural Least-Squares Monte Carlo
experiments and make common-random-number comparisons straightforward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class GBMContract:
    """Vanilla-option contract and risk-neutral process parameters."""

    contract_id: str
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float

    def __post_init__(self) -> None:
        values = {
            "spot": self.spot,
            "strike": self.strike,
            "time_to_maturity": self.time_to_maturity,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "volatility": self.volatility,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value!r}.")
        if not self.contract_id:
            raise ValueError("contract_id cannot be empty.")
        if self.spot <= 0.0:
            raise ValueError("spot must be greater than zero.")
        if self.strike <= 0.0:
            raise ValueError("strike must be greater than zero.")
        if self.time_to_maturity <= 0.0:
            raise ValueError("time_to_maturity must be greater than zero.")
        if self.volatility < 0.0:
            raise ValueError("volatility cannot be negative.")

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-compatible contract representation."""

        return asdict(self)


@dataclass(frozen=True)
class GBMMomentValidation:
    """Empirical and theoretical terminal-distribution diagnostics."""

    empirical_mean: float
    theoretical_mean: float
    mean_relative_error: float
    empirical_variance: float
    theoretical_variance: float
    variance_relative_error: float
    n_paths: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible result."""

        return asdict(self)


def generate_antithetic_normals(
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Generate standard-normal shocks with antithetic pairing.

    For even ``n_paths``, the second half is exactly the negative of the first.
    For odd ``n_paths``, one additional independent path is appended.
    """

    if n_paths <= 0:
        raise ValueError("n_paths must be greater than zero.")
    if n_steps <= 0:
        raise ValueError("n_steps must be greater than zero.")

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    base = rng.standard_normal((half, n_steps)).astype(dtype, copy=False)
    shocks = np.concatenate([base, -base], axis=0)

    if n_paths % 2:
        extra = rng.standard_normal((1, n_steps)).astype(dtype, copy=False)
        shocks = np.concatenate([shocks, extra], axis=0)

    return shocks


def simulate_gbm_paths(
    *,
    spot: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
    antithetic: bool = True,
    dtype: np.dtype | type = np.float64,
    normal_shocks: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate risk-neutral geometric Brownian motion paths.

    Returns an array with shape ``(n_paths, n_steps + 1)``. Column zero contains
    the initial spot value. Supplying ``normal_shocks`` enables exact common
    random numbers across competing pricing policies.
    """

    numeric = {
        "spot": spot,
        "time_to_maturity": time_to_maturity,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "volatility": volatility,
    }
    for name, value in numeric.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}.")
    if spot <= 0.0:
        raise ValueError("spot must be greater than zero.")
    if time_to_maturity <= 0.0:
        raise ValueError("time_to_maturity must be greater than zero.")
    if volatility < 0.0:
        raise ValueError("volatility cannot be negative.")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be greater than zero.")

    if normal_shocks is None:
        if antithetic:
            normal_shocks = generate_antithetic_normals(
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                dtype=dtype,
            )
        else:
            rng = np.random.default_rng(seed)
            normal_shocks = rng.standard_normal((n_paths, n_steps)).astype(
                dtype, copy=False
            )
    else:
        normal_shocks = np.asarray(normal_shocks, dtype=dtype)
        if normal_shocks.shape != (n_paths, n_steps):
            raise ValueError(
                "normal_shocks must have shape "
                f"({n_paths}, {n_steps}); received {normal_shocks.shape}."
            )
        if not np.isfinite(normal_shocks).all():
            raise ValueError("normal_shocks contains non-finite values.")

    delta_t = time_to_maturity / n_steps
    drift = (
        risk_free_rate - dividend_yield - 0.5 * volatility**2
    ) * delta_t
    diffusion = volatility * math.sqrt(delta_t)
    log_increments = drift + diffusion * normal_shocks
    cumulative_log_returns = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1), dtype=dtype)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(cumulative_log_returns)
    return paths


def simulate_contract_paths(
    contract: GBMContract,
    *,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
    antithetic: bool = True,
    dtype: np.dtype | type = np.float64,
    normal_shocks: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate paths for a :class:`GBMContract`."""

    return simulate_gbm_paths(
        spot=contract.spot,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        dividend_yield=contract.dividend_yield,
        volatility=contract.volatility,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
        dtype=dtype,
        normal_shocks=normal_shocks,
    )


def theoretical_terminal_moments(
    contract: GBMContract,
) -> tuple[float, float]:
    """Return risk-neutral terminal mean and variance for GBM."""

    mean = contract.spot * math.exp(
        (contract.risk_free_rate - contract.dividend_yield)
        * contract.time_to_maturity
    )
    variance = (
        contract.spot**2
        * math.exp(
            2.0
            * (contract.risk_free_rate - contract.dividend_yield)
            * contract.time_to_maturity
        )
        * (math.exp(contract.volatility**2 * contract.time_to_maturity) - 1.0)
    )
    return mean, variance


def validate_simulated_moments(
    paths: np.ndarray,
    contract: GBMContract,
) -> GBMMomentValidation:
    """Compare empirical terminal moments with their GBM values."""

    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be a two-dimensional path matrix.")
    if not np.isfinite(paths).all() or np.any(paths <= 0.0):
        raise ValueError("paths must contain finite positive values.")
    if not np.allclose(paths[:, 0], contract.spot):
        raise ValueError("the first path column does not match contract spot.")

    terminal = paths[:, -1]
    empirical_mean = float(np.mean(terminal))
    empirical_variance = float(np.var(terminal, ddof=1))
    theoretical_mean, theoretical_variance = theoretical_terminal_moments(contract)

    mean_relative_error = abs(empirical_mean - theoretical_mean) / max(
        abs(theoretical_mean), 1e-15
    )
    variance_relative_error = abs(
        empirical_variance - theoretical_variance
    ) / max(abs(theoretical_variance), 1e-15)

    return GBMMomentValidation(
        empirical_mean=empirical_mean,
        theoretical_mean=theoretical_mean,
        mean_relative_error=float(mean_relative_error),
        empirical_variance=empirical_variance,
        theoretical_variance=theoretical_variance,
        variance_relative_error=float(variance_relative_error),
        n_paths=int(paths.shape[0]),
    )


def sample_contracts_latin_hypercube(
    *,
    n_contracts: int,
    parameter_ranges: Mapping[str, tuple[float, float]],
    seed: int,
    prefix: str,
    strike: float = 100.0,
) -> list[GBMContract]:
    """Sample a reproducible contract grid with Latin hypercube sampling.

    Required ranges are ``moneyness``, ``time_to_maturity``,
    ``risk_free_rate``, ``dividend_yield``, and ``volatility``.
    """

    if n_contracts <= 0:
        raise ValueError("n_contracts must be greater than zero.")
    if strike <= 0.0:
        raise ValueError("strike must be greater than zero.")

    names = [
        "moneyness",
        "time_to_maturity",
        "risk_free_rate",
        "dividend_yield",
        "volatility",
    ]
    missing = [name for name in names if name not in parameter_ranges]
    if missing:
        raise ValueError(f"Missing parameter ranges: {missing}.")

    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        low, high = parameter_ranges[name]
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            raise ValueError(f"Invalid range for {name}: {(low, high)}.")
        lower.append(float(low))
        upper.append(float(high))

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit_sample = sampler.random(n=n_contracts)
    scaled = qmc.scale(unit_sample, lower, upper)

    contracts: list[GBMContract] = []
    for index, row in enumerate(scaled):
        values = dict(zip(names, row, strict=True))
        contracts.append(
            GBMContract(
                contract_id=f"{prefix}_{index:04d}",
                spot=float(values["moneyness"] * strike),
                strike=float(strike),
                time_to_maturity=float(values["time_to_maturity"]),
                risk_free_rate=float(values["risk_free_rate"]),
                dividend_yield=float(values["dividend_yield"]),
                volatility=float(values["volatility"]),
            )
        )
    return contracts


__all__ = [
    "GBMContract",
    "GBMMomentValidation",
    "generate_antithetic_normals",
    "sample_contracts_latin_hypercube",
    "simulate_contract_paths",
    "simulate_gbm_paths",
    "theoretical_terminal_moments",
    "validate_simulated_moments",
]
