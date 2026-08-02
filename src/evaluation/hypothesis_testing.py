"""Predefined and reproducible H1-H6 decision rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import pandas as pd


ALLOWED_DECISIONS = (
    "Supported",
    "Partially supported",
    "Not supported",
    "Inconclusive",
)


@dataclass(frozen=True)
class HypothesisDecision:
    hypothesis: str
    decision: str
    primary_evidence: str
    secondary_evidence: str
    limitation: str


def _finite(evidence: Mapping[str, Any], *keys: str) -> bool:
    try:
        return all(math.isfinite(float(evidence[key])) for key in keys)
    except (KeyError, TypeError, ValueError):
        return False


def decide_h1(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "black_scholes_mae", "direct_mlp_mae"):
        return HypothesisDecision(
            "H1",
            "Inconclusive",
            "Required Black–Scholes and direct-MLP MAE values are unavailable.",
            "",
            "No result is inferred from missing evidence.",
        )
    baseline = float(evidence["black_scholes_mae"])
    model = float(evidence["direct_mlp_mae"])
    ratio = model / baseline if baseline > 0 else math.inf
    decision = (
        "Supported"
        if ratio <= 0.95
        else "Partially supported"
        if ratio < 1.0
        else "Not supported"
    )
    return HypothesisDecision(
        "H1",
        decision,
        f"Direct MLP MAE / Black–Scholes proxy MAE = {ratio:.4f}.",
        "Supported requires a ratio no greater than 0.95.",
        "The decision concerns the fixed in-domain test split.",
    )


def decide_h2(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_mlp_mae", "best_residual_mae"):
        return HypothesisDecision(
            "H2",
            "Inconclusive",
            "Required direct and residual-model MAE values are unavailable.",
            "",
            "No result is inferred from missing evidence.",
        )
    direct = float(evidence["direct_mlp_mae"])
    residual = float(evidence["best_residual_mae"])
    ratio = residual / direct if direct > 0 else math.inf
    decision = (
        "Supported"
        if ratio <= 0.98
        else "Partially supported"
        if ratio < 1.0
        else "Not supported"
    )
    return HypothesisDecision(
        "H2",
        decision,
        f"Best residual-model MAE / direct-model MAE = {ratio:.4f}.",
        "Supported requires a ratio no greater than 0.98.",
        "The residual configuration must have been selected on validation data.",
    )


def decide_h3(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_violation_rate", "constrained_violation_rate"):
        return HypothesisDecision(
            "H3",
            "Inconclusive",
            "Comparable financial-violation rates are unavailable.",
            "",
            "No result is inferred from missing evidence.",
        )
    direct = float(evidence["direct_violation_rate"])
    constrained = float(evidence["constrained_violation_rate"])
    if constrained == 0.0 and direct > 0.0:
        decision = "Supported"
    elif constrained <= direct:
        decision = "Partially supported"
    else:
        decision = "Not supported"
    return HypothesisDecision(
        "H3",
        decision,
        f"Direct violation rate={direct:.6g}; constrained rate={constrained:.6g}.",
        "Zero constrained violations with positive direct violations supports H3.",
        "Architectural lower-bound guarantees do not establish global monotonicity.",
    )


def decide_h4(evidence: Mapping[str, Any]) -> HypothesisDecision:
    required = (
        "price_only_boundary_f1",
        "multitask_boundary_f1",
        "price_only_boundary_error",
        "multitask_boundary_error",
    )
    if not _finite(evidence, *required):
        return HypothesisDecision(
            "H4",
            "Inconclusive",
            "Boundary classification or boundary-pricing evidence is unavailable.",
            "",
            "No result is inferred from missing evidence.",
        )

    baseline_f1 = float(evidence["price_only_boundary_f1"])
    multitask_f1 = float(evidence["multitask_boundary_f1"])
    baseline_error = float(evidence["price_only_boundary_error"])
    multitask_error = float(evidence["multitask_boundary_error"])

    required_f1_gain = float(evidence.get("required_h4_f1_gain", 0.02))
    required_error_improvement = float(
        evidence.get("required_h4_error_improvement", 0.0)
    )

    f1_gain = multitask_f1 - baseline_f1
    relative_error_improvement = (
        (baseline_error - multitask_error) / baseline_error
        if baseline_error > 0
        else -math.inf
    )

    f1_pass = f1_gain >= required_f1_gain
    error_pass = relative_error_improvement >= required_error_improvement

    if f1_pass and error_pass:
        decision = "Supported"
    elif f1_gain > 0.0 or relative_error_improvement > 0.0:
        decision = "Partially supported"
    else:
        decision = "Not supported"

    return HypothesisDecision(
        "H4",
        decision,
        (
            f"Boundary F1 gain={f1_gain:.6f}; relative boundary-error "
            f"improvement={relative_error_improvement:.4%}."
        ),
        (
            f"Required F1 gain={required_f1_gain:.6f}; required relative "
            f"error improvement={required_error_improvement:.4%}."
        ),
        "Boundary quality depends on the CRR label resolution and test-band definition.",
    )


def decide_h5(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "crr_seconds_per_option", "neural_seconds_per_option"):
        return HypothesisDecision(
            "H5",
            "Inconclusive",
            "Comparable CRR and neural marginal-runtime evidence is unavailable.",
            "",
            "Up-front neural training cost cannot substitute for marginal-runtime evidence.",
        )
    crr = float(evidence["crr_seconds_per_option"])
    neural = float(evidence["neural_seconds_per_option"])
    ratio = neural / crr if crr > 0 else math.inf
    decision = (
        "Supported"
        if ratio <= 0.50
        else "Partially supported"
        if ratio < 1.0
        else "Not supported"
    )
    return HypothesisDecision(
        "H5",
        decision,
        f"Neural LSM marginal runtime / CRR runtime = {ratio:.6f}.",
        "Supported requires a ratio no greater than 0.50.",
        "Data generation and policy-training cost are reported separately.",
    )


def decide_h6(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "in_domain_mae", "aggregate_ood_mae"):
        return HypothesisDecision(
            "H6",
            "Inconclusive",
            "Comparable in-domain and aggregate OOD MAE values are unavailable.",
            "",
            "No result is inferred from missing evidence.",
        )
    in_domain = float(evidence["in_domain_mae"])
    ood = float(evidence["aggregate_ood_mae"])
    ratio = ood / in_domain if in_domain > 0 else math.inf
    decision = (
        "Supported"
        if ratio >= 1.25
        else "Partially supported"
        if ratio > 1.0
        else "Not supported"
    )
    model = evidence.get("h6_model", "selected static model")
    return HypothesisDecision(
        "H6",
        decision,
        f"{model}: aggregate OOD MAE / in-domain MAE = {ratio:.4f}.",
        "Supported requires a deterioration ratio of at least 1.25.",
        "An aggregate result can conceal regime-specific differences.",
    )


def decide_all_hypotheses(evidence: Mapping[str, Any]) -> pd.DataFrame:
    """Apply H1-H6 rules to a single evidence mapping."""

    decisions = [
        decide_h1(evidence),
        decide_h2(evidence),
        decide_h3(evidence),
        decide_h4(evidence),
        decide_h5(evidence),
        decide_h6(evidence),
    ]
    frame = pd.DataFrame(asdict(decision) for decision in decisions)
    if not frame["decision"].isin(ALLOWED_DECISIONS).all():
        raise RuntimeError("Unexpected hypothesis decision")
    return frame


__all__ = [
    "ALLOWED_DECISIONS",
    "HypothesisDecision",
    "decide_all_hypotheses",
    "decide_h1",
    "decide_h2",
    "decide_h3",
    "decide_h4",
    "decide_h5",
    "decide_h6",
]
