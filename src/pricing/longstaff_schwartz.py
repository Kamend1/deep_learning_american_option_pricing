"""Classical Least-Squares Monte Carlo for American put options.

The implementation separates policy fitting from policy valuation. A continuation
policy is fitted on one set of paths and priced on independent paths, which avoids
reporting the optimistic value obtained by evaluating a stopping policy on the
same paths used to estimate it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Literal

import numpy as np
import pandas as pd


BasisName = Literal["polynomial", "laguerre"]


@dataclass(frozen=True)
class ContinuationRegression:
    """Regression coefficients for one exercise date."""

    step_index: int
    coefficients: np.ndarray
    basis: BasisName
    degree: int
    strike: float
    n_observations: int
    condition_number: float

    def predict(self, spot_values: np.ndarray) -> np.ndarray:
        """Estimate non-negative continuation values."""

        matrix = _basis_matrix(
            np.asarray(spot_values, dtype=np.float64) / self.strike,
            basis=self.basis,
            degree=self.degree,
        )
        values = matrix @ self.coefficients
        return np.maximum(values, 0.0)


@dataclass(frozen=True)
class LongstaffSchwartzPolicy:
    """Contract-specific classical LSM stopping policy."""

    strike: float
    time_to_maturity: float
    risk_free_rate: float
    n_steps: int
    basis: BasisName
    degree: int
    regressions: dict[int, ContinuationRegression] = field(default_factory=dict)

    @property
    def delta_t(self) -> float:
        return self.time_to_maturity / self.n_steps


@dataclass(frozen=True)
class LSMPriceResult:
    """Price, uncertainty, and stopping-policy outputs."""

    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    n_paths: int
    discounted_payoffs: np.ndarray
    exercise_indices: np.ndarray
    exercise_times: np.ndarray
    exercised_early_rate: float
    intrinsic_at_time_zero: float
    time_zero_exercise: bool

    def summary(self) -> dict[str, float | int | bool]:
        """Return scalar result fields."""

        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "confidence_interval_low": self.confidence_interval_low,
            "confidence_interval_high": self.confidence_interval_high,
            "n_paths": self.n_paths,
            "exercised_early_rate": self.exercised_early_rate,
            "intrinsic_at_time_zero": self.intrinsic_at_time_zero,
            "time_zero_exercise": self.time_zero_exercise,
        }


@dataclass(frozen=True)
class LSMExperimentResult:
    """Combined fitted policy and independent valuation result."""

    policy: LongstaffSchwartzPolicy
    valuation: LSMPriceResult
    fit_diagnostics: pd.DataFrame


def _validate_paths(paths: np.ndarray, *, n_steps: int | None = None) -> np.ndarray:
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be a two-dimensional matrix.")
    if paths.shape[0] < 2:
        raise ValueError("at least two paths are required.")
    if n_steps is not None and paths.shape[1] != n_steps + 1:
        raise ValueError(
            f"Expected {n_steps + 1} path columns; received {paths.shape[1]}."
        )
    if not np.isfinite(paths).all() or np.any(paths <= 0.0):
        raise ValueError("paths must contain finite positive values.")
    return paths


def _put_intrinsic(spot_values: np.ndarray | float, strike: float) -> np.ndarray:
    return np.maximum(strike - np.asarray(spot_values, dtype=np.float64), 0.0)


def _basis_matrix(
    normalized_spot: np.ndarray,
    *,
    basis: BasisName,
    degree: int,
) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree cannot be negative.")
    x = np.asarray(normalized_spot, dtype=np.float64).reshape(-1)
    if basis == "polynomial":
        return np.polynomial.polynomial.polyvander(x, degree)
    if basis == "laguerre":
        raw = np.polynomial.laguerre.lagvander(x, degree)
        return raw * np.exp(-0.5 * x)[:, None]
    raise ValueError("basis must be 'polynomial' or 'laguerre'.")


def _fit_regression(
    spot_values: np.ndarray,
    targets: np.ndarray,
    *,
    strike: float,
    basis: BasisName,
    degree: int,
    ridge: float,
    step_index: int,
) -> ContinuationRegression:
    n_observations = len(spot_values)
    effective_degree = min(degree, max(n_observations - 1, 0))
    matrix = _basis_matrix(
        np.asarray(spot_values, dtype=np.float64) / strike,
        basis=basis,
        degree=effective_degree,
    )
    targets = np.asarray(targets, dtype=np.float64)

    gram = matrix.T @ matrix
    penalty = np.eye(gram.shape[0], dtype=np.float64) * max(ridge, 0.0)
    if penalty.size:
        penalty[0, 0] = 0.0
    rhs = matrix.T @ targets

    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, targets, rcond=None)[0]

    condition_number = float(np.linalg.cond(matrix)) if matrix.size else math.inf
    return ContinuationRegression(
        step_index=step_index,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        basis=basis,
        degree=effective_degree,
        strike=strike,
        n_observations=n_observations,
        condition_number=condition_number,
    )


def fit_longstaff_schwartz_policy(
    training_paths: np.ndarray,
    *,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    basis: BasisName = "polynomial",
    degree: int = 2,
    ridge: float = 1e-10,
    minimum_regression_observations: int = 8,
) -> tuple[LongstaffSchwartzPolicy, pd.DataFrame]:
    """Fit a classical American-put stopping policy by backward induction."""

    if strike <= 0.0:
        raise ValueError("strike must be greater than zero.")
    if time_to_maturity <= 0.0:
        raise ValueError("time_to_maturity must be greater than zero.")
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if degree < 0:
        raise ValueError("degree cannot be negative.")
    if minimum_regression_observations < 2:
        raise ValueError("minimum_regression_observations must be at least two.")

    training_paths = _validate_paths(training_paths)
    n_paths, n_columns = training_paths.shape
    n_steps = n_columns - 1
    delta_t = time_to_maturity / n_steps

    cashflows = _put_intrinsic(training_paths[:, -1], strike)
    exercise_indices = np.full(n_paths, n_steps, dtype=np.int32)
    regressions: dict[int, ContinuationRegression] = {}
    diagnostics: list[dict[str, float | int | str]] = []

    for step_index in range(n_steps - 1, 0, -1):
        spots = training_paths[:, step_index]
        intrinsic = _put_intrinsic(spots, strike)
        in_the_money = intrinsic > 0.0
        count = int(np.sum(in_the_money))

        if count < minimum_regression_observations:
            diagnostics.append(
                {
                    "step_index": step_index,
                    "n_regression_observations": count,
                    "exercise_count": 0,
                    "basis": basis,
                    "effective_degree": 0,
                    "condition_number": math.nan,
                    "status": "insufficient_observations",
                }
            )
            continue

        future_discount = np.exp(
            -risk_free_rate
            * delta_t
            * (exercise_indices[in_the_money] - step_index)
        )
        targets = cashflows[in_the_money] * future_discount
        regression = _fit_regression(
            spots[in_the_money],
            targets,
            strike=strike,
            basis=basis,
            degree=degree,
            ridge=ridge,
            step_index=step_index,
        )
        regressions[step_index] = regression

        continuation = regression.predict(spots[in_the_money])
        exercise_local = intrinsic[in_the_money] > continuation
        exercise_path_indices = np.flatnonzero(in_the_money)[exercise_local]

        cashflows[exercise_path_indices] = intrinsic[exercise_path_indices]
        exercise_indices[exercise_path_indices] = step_index

        diagnostics.append(
            {
                "step_index": step_index,
                "n_regression_observations": count,
                "exercise_count": int(np.sum(exercise_local)),
                "basis": basis,
                "effective_degree": regression.degree,
                "condition_number": regression.condition_number,
                "status": "fitted",
            }
        )

    policy = LongstaffSchwartzPolicy(
        strike=float(strike),
        time_to_maturity=float(time_to_maturity),
        risk_free_rate=float(risk_free_rate),
        n_steps=n_steps,
        basis=basis,
        degree=degree,
        regressions=regressions,
    )
    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(
        "step_index", ignore_index=True
    )
    return policy, diagnostics_frame


def evaluate_longstaff_schwartz_policy(
    valuation_paths: np.ndarray,
    policy: LongstaffSchwartzPolicy,
    *,
    confidence_level: float = 0.95,
) -> LSMPriceResult:
    """Evaluate a fitted stopping policy on independent paths."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")
    valuation_paths = _validate_paths(valuation_paths, n_steps=policy.n_steps)
    n_paths = valuation_paths.shape[0]
    delta_t = policy.delta_t

    active = np.ones(n_paths, dtype=bool)
    exercise_indices = np.full(n_paths, policy.n_steps, dtype=np.int32)
    payoffs = _put_intrinsic(valuation_paths[:, -1], policy.strike)

    for step_index in range(1, policy.n_steps):
        regression = policy.regressions.get(step_index)
        if regression is None:
            continue
        candidate_indices = np.flatnonzero(active)
        if candidate_indices.size == 0:
            break
        spots = valuation_paths[candidate_indices, step_index]
        intrinsic = _put_intrinsic(spots, policy.strike)
        itm = intrinsic > 0.0
        if not np.any(itm):
            continue
        itm_path_indices = candidate_indices[itm]
        continuation = regression.predict(spots[itm])
        exercise = intrinsic[itm] > continuation
        exercise_path_indices = itm_path_indices[exercise]
        if exercise_path_indices.size:
            payoffs[exercise_path_indices] = _put_intrinsic(
                valuation_paths[exercise_path_indices, step_index], policy.strike
            )
            exercise_indices[exercise_path_indices] = step_index
            active[exercise_path_indices] = False

    discounted_payoffs = payoffs * np.exp(
        -policy.risk_free_rate * delta_t * exercise_indices
    )
    intrinsic_zero = float(
        _put_intrinsic(valuation_paths[0, 0], policy.strike)
    )
    continuation_zero = float(np.mean(discounted_payoffs))
    time_zero_exercise = intrinsic_zero >= continuation_zero and intrinsic_zero > 0.0

    if time_zero_exercise:
        discounted_payoffs = np.full(n_paths, intrinsic_zero, dtype=np.float64)
        exercise_indices = np.zeros(n_paths, dtype=np.int32)

    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(n_paths))
    # 1.959963984540054 is the standard-normal 97.5% quantile.
    z_value = 1.959963984540054 if confidence_level == 0.95 else _normal_quantile(
        0.5 + confidence_level / 2.0
    )
    ci_low = price - z_value * standard_error
    ci_high = price + z_value * standard_error
    exercise_times = exercise_indices.astype(np.float64) * delta_t
    exercised_early_rate = float(np.mean(exercise_indices < policy.n_steps))

    return LSMPriceResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_low=float(ci_low),
        confidence_interval_high=float(ci_high),
        n_paths=n_paths,
        discounted_payoffs=discounted_payoffs,
        exercise_indices=exercise_indices,
        exercise_times=exercise_times,
        exercised_early_rate=exercised_early_rate,
        intrinsic_at_time_zero=intrinsic_zero,
        time_zero_exercise=time_zero_exercise,
    )


def _normal_quantile(probability: float) -> float:
    """Acklam-style inverse-normal approximation without an extra dependency."""

    # Coefficients from Peter J. Acklam's rational approximation.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be strictly between zero and one.")
    if probability < plow:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def longstaff_schwartz_put_price(
    training_paths: np.ndarray,
    valuation_paths: np.ndarray,
    *,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    basis: BasisName = "polynomial",
    degree: int = 2,
    ridge: float = 1e-10,
    minimum_regression_observations: int = 8,
) -> LSMExperimentResult:
    """Fit a policy and value it on an independent path sample."""

    policy, diagnostics = fit_longstaff_schwartz_policy(
        training_paths,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        basis=basis,
        degree=degree,
        ridge=ridge,
        minimum_regression_observations=minimum_regression_observations,
    )
    valuation = evaluate_longstaff_schwartz_policy(valuation_paths, policy)
    return LSMExperimentResult(
        policy=policy,
        valuation=valuation,
        fit_diagnostics=diagnostics,
    )


__all__ = [
    "BasisName",
    "ContinuationRegression",
    "LSMExperimentResult",
    "LSMPriceResult",
    "LongstaffSchwartzPolicy",
    "evaluate_longstaff_schwartz_policy",
    "fit_longstaff_schwartz_policy",
    "longstaff_schwartz_put_price",
]
