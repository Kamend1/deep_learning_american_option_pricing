"""Deployment selection and safety controls for Notebook 08.

The module deliberately separates model selection from final test and
out-of-domain evaluation.  The preferred integrated deployment candidate is
chosen only from validation quality, operational runtime, model size, and the
protected-price checks observed on validation data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DOMAIN_COLUMNS: tuple[str, ...] = (
    "moneyness",
    "time_to_maturity",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
)


@dataclass(frozen=True, slots=True)
class IntegratedDeploymentSelectionConfig:
    """Predefined tolerances for the in-domain deployment decision."""

    validation_price_relative_tolerance: float = 0.05
    validation_exercise_f1_tolerance: float = 0.001
    validation_disagreement_absolute_tolerance: float = 0.002
    require_smaller_model: bool = True
    require_faster_inference: bool = True
    require_zero_validation_price_violations: bool = True

    def __post_init__(self) -> None:
        if self.validation_price_relative_tolerance < 0.0:
            raise ValueError("validation_price_relative_tolerance cannot be negative")
        if self.validation_exercise_f1_tolerance < 0.0:
            raise ValueError("validation_exercise_f1_tolerance cannot be negative")
        if self.validation_disagreement_absolute_tolerance < 0.0:
            raise ValueError(
                "validation_disagreement_absolute_tolerance cannot be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def select_integrated_deployment_candidate(
    candidate_comparison: pd.DataFrame,
    *,
    scratch_candidate: str = "selected_scratch",
    warm_candidate: str = "warm_start",
    config: IntegratedDeploymentSelectionConfig | None = None,
) -> dict[str, Any]:
    """Choose the preferred in-domain integrated deployment candidate.

    The function intentionally has no test or OOD arguments.  Extra columns in
    ``candidate_comparison`` are ignored.  This makes it impossible for final
    test or out-of-domain metrics to affect the deployment selection.
    """

    cfg = config or IntegratedDeploymentSelectionConfig()
    required_columns = (
        "parameter_count",
        "validation_constrained_rmse",
        "validation_exercise_f1",
        "validation_disagreement_rate",
        "validation_protected_price_violations",
        "validation_median_inference_seconds",
    )
    missing_columns = [
        column for column in required_columns if column not in candidate_comparison.columns
    ]
    if missing_columns:
        raise ValueError(
            "Candidate comparison is missing deployment fields: "
            f"{missing_columns}"
        )
    missing_candidates = [
        name
        for name in (scratch_candidate, warm_candidate)
        if name not in candidate_comparison.index
    ]
    if missing_candidates:
        raise ValueError(f"Candidate comparison is missing rows: {missing_candidates}")

    scratch = candidate_comparison.loc[scratch_candidate]
    warm = candidate_comparison.loc[warm_candidate]

    scratch_parameters = int(scratch["parameter_count"])
    warm_parameters = int(warm["parameter_count"])
    scratch_rmse = _finite_float(
        scratch["validation_constrained_rmse"],
        name="scratch validation_constrained_rmse",
    )
    warm_rmse = _finite_float(
        warm["validation_constrained_rmse"],
        name="warm validation_constrained_rmse",
    )
    scratch_f1 = _finite_float(
        scratch["validation_exercise_f1"],
        name="scratch validation_exercise_f1",
    )
    warm_f1 = _finite_float(
        warm["validation_exercise_f1"],
        name="warm validation_exercise_f1",
    )
    scratch_disagreement = _finite_float(
        scratch["validation_disagreement_rate"],
        name="scratch validation_disagreement_rate",
    )
    warm_disagreement = _finite_float(
        warm["validation_disagreement_rate"],
        name="warm validation_disagreement_rate",
    )
    scratch_runtime = _finite_float(
        scratch["validation_median_inference_seconds"],
        name="scratch validation_median_inference_seconds",
    )
    warm_runtime = _finite_float(
        warm["validation_median_inference_seconds"],
        name="warm validation_median_inference_seconds",
    )
    warm_violations = int(warm["validation_protected_price_violations"])

    checks = {
        "smaller_parameter_count": (
            warm_parameters < scratch_parameters
            if cfg.require_smaller_model
            else True
        ),
        "validation_price_within_tolerance": (
            warm_rmse
            <= scratch_rmse * (1.0 + cfg.validation_price_relative_tolerance)
        ),
        "validation_exercise_f1_within_tolerance": (
            warm_f1 >= scratch_f1 - cfg.validation_exercise_f1_tolerance
        ),
        "validation_disagreement_within_tolerance": (
            warm_disagreement
            <= scratch_disagreement
            + cfg.validation_disagreement_absolute_tolerance
        ),
        "faster_validation_inference": (
            warm_runtime < scratch_runtime
            if cfg.require_faster_inference
            else True
        ),
        "zero_validation_protected_price_violations": (
            warm_violations == 0
            if cfg.require_zero_validation_price_violations
            else True
        ),
    }

    warm_selected = bool(all(checks.values()))
    preferred = warm_candidate if warm_selected else scratch_candidate

    return {
        "schema_version": 1,
        "selection_scope": "in_domain_combined_deployment",
        "preferred_integrated_candidate": preferred,
        "selected_scratch_candidate": scratch_candidate,
        "selection_rule": (
            "Choose warm-start only when all predefined validation and "
            "operational checks pass."
        ),
        "selection_config": cfg.to_dict(),
        "checks": {name: bool(value) for name, value in checks.items()},
        "all_checks_passed": warm_selected,
        "selection_evidence": [
            "parameter_count",
            "validation_constrained_rmse",
            "validation_exercise_f1",
            "validation_disagreement_rate",
            "validation_protected_price_violations",
            "validation_median_inference_seconds",
        ],
        "excluded_selection_evidence": [
            "held_out_test_metrics",
            "out_of_domain_metrics",
        ],
        "test_metrics_used_for_selection": False,
        "ood_metrics_used_for_selection": False,
    }


def build_union_domain_bounds(*range_specs: Any) -> dict[str, dict[str, float]]:
    """Build the union of the in-domain generation ranges.

    ``range_specs`` may be dataclasses or objects exposing the RangeSpec fields
    used by the production generator.
    """

    if not range_specs:
        raise ValueError("At least one range specification is required")
    source_names = {
        "moneyness": "moneyness",
        "time_to_maturity": "time_to_maturity",
        "volatility": "volatility",
        "risk_free_rate": "risk_free_rate",
        "dividend_yield": "dividend_yield",
    }
    bounds: dict[str, dict[str, float]] = {}
    for output_name, attribute_name in source_names.items():
        pairs: list[tuple[float, float]] = []
        for spec in range_specs:
            pair = getattr(spec, attribute_name)
            lower, upper = float(pair[0]), float(pair[1])
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid bounds for {attribute_name}: {pair}")
            pairs.append((lower, upper))
        bounds[output_name] = {
            "minimum": min(pair[0] for pair in pairs),
            "maximum": max(pair[1] for pair in pairs),
        }
    return bounds


def assess_integrated_model_domain(
    frame: pd.DataFrame,
    domain_bounds: Mapping[str, Mapping[str, float]],
    *,
    neural_path: str = "warm_start_integrated_model",
    fallback_path: str = "high_resolution_crr",
) -> pd.DataFrame:
    """Classify each contract as in-domain or numerical-fallback territory."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    working = frame.copy()
    if "moneyness" not in working.columns:
        if {"spot", "strike"}.issubset(working.columns):
            strike = pd.to_numeric(working["strike"], errors="raise")
            if (strike <= 0.0).any():
                raise ValueError("strike must remain positive")
            working["moneyness"] = (
                pd.to_numeric(working["spot"], errors="raise") / strike
            )
        elif "log_moneyness" in working.columns:
            working["moneyness"] = np.exp(
                pd.to_numeric(working["log_moneyness"], errors="raise")
            )

    missing = [column for column in DOMAIN_COLUMNS if column not in working.columns]
    if missing:
        raise ValueError(f"Domain assessment is missing columns: {missing}")

    violations: list[list[str]] = [[] for _ in range(len(working))]
    for column in DOMAIN_COLUMNS:
        if column not in domain_bounds:
            raise ValueError(f"Domain bounds are missing {column!r}")
        lower = float(domain_bounds[column]["minimum"])
        upper = float(domain_bounds[column]["maximum"])
        values = pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values < lower) | (values > upper)
        for index in np.flatnonzero(invalid):
            violations[int(index)].append(column)

    in_domain = np.array([not fields for fields in violations], dtype=bool)
    return pd.DataFrame(
        {
            "in_domain": in_domain,
            "out_of_domain_fields": [",".join(fields) for fields in violations],
            "recommended_path": np.where(in_domain, neural_path, fallback_path),
        },
        index=frame.index,
    )


def paired_price_error_evidence(
    true_values: Sequence[float] | np.ndarray,
    scratch_predictions: Sequence[float] | np.ndarray,
    warm_predictions: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Summarize paired absolute-error improvement with a bootstrap interval."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one")
    truth = np.asarray(true_values, dtype=np.float64).reshape(-1)
    scratch = np.asarray(scratch_predictions, dtype=np.float64).reshape(-1)
    warm = np.asarray(warm_predictions, dtype=np.float64).reshape(-1)
    if not (len(truth) == len(scratch) == len(warm)) or len(truth) == 0:
        raise ValueError("Paired arrays must have the same non-zero length")
    if not np.isfinite(np.column_stack([truth, scratch, warm])).all():
        raise ValueError("Paired arrays must be finite")

    scratch_error = np.abs(scratch - truth)
    warm_error = np.abs(warm - truth)
    improvement = scratch_error - warm_error

    rng = np.random.default_rng(seed)
    chunk_size = min(16, bootstrap_samples)
    boot_means: list[np.ndarray] = []
    remaining = bootstrap_samples
    while remaining > 0:
        size = min(chunk_size, remaining)
        indices = rng.integers(0, len(improvement), size=(size, len(improvement)))
        boot_means.append(improvement[indices].mean(axis=1))
        remaining -= size
    bootstrap = np.concatenate(boot_means)
    alpha = (1.0 - confidence_level) / 2.0

    return {
        "observations": int(len(improvement)),
        "scratch_mae": float(scratch_error.mean()),
        "warm_start_mae": float(warm_error.mean()),
        "mean_absolute_error_improvement": float(improvement.mean()),
        "relative_mae_improvement": float(
            improvement.mean() / scratch_error.mean()
            if scratch_error.mean() > 0.0
            else np.nan
        ),
        "warm_start_win_rate": float(np.mean(warm_error < scratch_error)),
        "scratch_win_rate": float(np.mean(scratch_error < warm_error)),
        "tie_rate": float(np.mean(np.isclose(scratch_error, warm_error))),
        "bootstrap_ci_lower": float(np.quantile(bootstrap, alpha)),
        "bootstrap_ci_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        "bootstrap_samples": int(bootstrap_samples),
        "confidence_level": float(confidence_level),
    }


def paired_exercise_decision_evidence(
    true_labels: Sequence[bool] | np.ndarray,
    scratch_probabilities: Sequence[float] | np.ndarray,
    warm_probabilities: Sequence[float] | np.ndarray,
    *,
    scratch_threshold: float,
    warm_threshold: float,
) -> dict[str, int | float]:
    """Compare candidate exercise decisions on the same observations."""

    labels = np.asarray(true_labels, dtype=bool).reshape(-1)
    scratch = np.asarray(scratch_probabilities, dtype=np.float64).reshape(-1)
    warm = np.asarray(warm_probabilities, dtype=np.float64).reshape(-1)
    if not (len(labels) == len(scratch) == len(warm)) or len(labels) == 0:
        raise ValueError("Paired arrays must have the same non-zero length")
    scratch_correct = (scratch >= scratch_threshold) == labels
    warm_correct = (warm >= warm_threshold) == labels
    warm_only = warm_correct & ~scratch_correct
    scratch_only = scratch_correct & ~warm_correct
    both = warm_correct & scratch_correct
    neither = ~warm_correct & ~scratch_correct
    return {
        "observations": int(len(labels)),
        "both_correct": int(both.sum()),
        "warm_start_only_correct": int(warm_only.sum()),
        "scratch_only_correct": int(scratch_only.sum()),
        "both_incorrect": int(neither.sum()),
        "warm_start_accuracy": float(warm_correct.mean()),
        "scratch_accuracy": float(scratch_correct.mean()),
        "net_additional_correct_warm_start": int(warm_only.sum() - scratch_only.sum()),
    }


def assert_deployment_selection_integrity(selection: Mapping[str, Any]) -> None:
    """Fail when final test or OOD evidence leaked into deployment selection."""

    if selection.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError("Deployment selection must not use held-out test metrics")
    if selection.get("ood_metrics_used_for_selection") is not False:
        raise RuntimeError("Deployment selection must not use OOD metrics")
    evidence = {str(value) for value in selection.get("selection_evidence", [])}
    forbidden_tokens = ("test", "ood", "out_of_domain")
    contaminated = sorted(
        value for value in evidence if any(token in value.lower() for token in forbidden_tokens)
    )
    if contaminated:
        raise RuntimeError(f"Deployment selection contains forbidden evidence: {contaminated}")


__all__ = [
    "DOMAIN_COLUMNS",
    "IntegratedDeploymentSelectionConfig",
    "assess_integrated_model_domain",
    "assert_deployment_selection_integrity",
    "build_union_domain_bounds",
    "paired_exercise_decision_evidence",
    "paired_price_error_evidence",
    "select_integrated_deployment_candidate",
]
