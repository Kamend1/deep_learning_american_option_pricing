"""Predefined H1-H6 decision rules for the final evaluation."""

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
        return HypothesisDecision("H1", "Inconclusive", "Required MAE values unavailable.", "", "")
    baseline = float(evidence["black_scholes_mae"])
    model = float(evidence["direct_mlp_mae"])
    ratio = model / baseline if baseline > 0 else math.inf
    decision = "Supported" if ratio <= 0.95 else "Partially supported" if ratio < 1 else "Not supported"
    return HypothesisDecision(
        "H1",
        decision,
        f"Direct MLP MAE / Black–Scholes proxy MAE = {ratio:.4f}.",
        "Thresholds: <=0.95 supported; <1 partially supported.",
        "Decision concerns the predefined evaluation domain.",
    )


def decide_h2(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_mlp_mae", "best_residual_mae"):
        return HypothesisDecision("H2", "Inconclusive", "Required MAE values unavailable.", "", "")
    direct = float(evidence["direct_mlp_mae"])
    residual = float(evidence["best_residual_mae"])
    ratio = residual / direct if direct > 0 else math.inf
    decision = "Supported" if ratio <= 0.98 else "Partially supported" if ratio < 1 else "Not supported"
    return HypothesisDecision(
        "H2",
        decision,
        f"Residual-model MAE / direct-model MAE = {ratio:.4f}.",
        "Thresholds: <=0.98 supported; <1 partially supported.",
        "The selected residual configuration must be chosen on validation data.",
    )


def decide_h3(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_violation_rate", "constrained_violation_rate"):
        return HypothesisDecision("H3", "Inconclusive", "Violation rates unavailable.", "", "")
    direct = float(evidence["direct_violation_rate"])
    constrained = float(evidence["constrained_violation_rate"])
    if constrained == 0 and direct > 0:
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
        "Architectural guarantees do not establish monotonicity outside tested grids.",
    )


def decide_h4(evidence: Mapping[str, Any]) -> HypothesisDecision:
    required = (
        "price_only_boundary_f1",
        "multitask_boundary_f1",
        "price_only_boundary_error",
        "multitask_boundary_error",
    )
    if not _finite(evidence, *required):
        return HypothesisDecision("H4", "Inconclusive", "Boundary evidence unavailable.", "", "")
    f1_gain = float(evidence["multitask_boundary_f1"]) - float(evidence["price_only_boundary_f1"])
    error_improved = float(evidence["multitask_boundary_error"]) < float(evidence["price_only_boundary_error"])
    if f1_gain >= 0.02 and error_improved:
        decision = "Supported"
    elif f1_gain > 0 or error_improved:
        decision = "Partially supported"
    else:
        decision = "Not supported"
    return HypothesisDecision(
        "H4",
        decision,
        f"Boundary F1 gain={f1_gain:.4f}; boundary-location error improved={error_improved}.",
        "Supported requires both >=0.02 F1 gain and lower boundary error.",
        "Boundary quality depends on the CRR label resolution.",
    )


def decide_h5(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "crr_seconds_per_option", "neural_seconds_per_option"):
        return HypothesisDecision("H5", "Inconclusive", "Comparable runtime evidence unavailable.", "", "")
    crr = float(evidence["crr_seconds_per_option"])
    neural = float(evidence["neural_seconds_per_option"])
    ratio = neural / crr if crr > 0 else math.inf
    decision = "Supported" if ratio <= 0.5 else "Partially supported" if ratio < 1 else "Not supported"
    return HypothesisDecision(
        "H5",
        decision,
        f"Neural marginal runtime / CRR runtime = {ratio:.6f}.",
        "Thresholds: <=0.5 supported; <1 partially supported.",
        "Up-front data generation and training cost must be reported separately.",
    )


def decide_h6(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "in_domain_mae", "aggregate_ood_mae"):
        return HypothesisDecision("H6", "Inconclusive", "In-domain or OOD MAE unavailable.", "", "")
    in_domain = float(evidence["in_domain_mae"])
    ood = float(evidence["aggregate_ood_mae"])
    ratio = ood / in_domain if in_domain > 0 else math.inf
    decision = "Supported" if ratio >= 1.25 else "Partially supported" if ratio > 1 else "Not supported"
    return HypothesisDecision(
        "H6",
        decision,
        f"Aggregate OOD MAE / in-domain MAE = {ratio:.4f}.",
        "Thresholds: >=1.25 supported; >1 partially supported.",
        "Aggregated OOD results can conceal regime-specific differences.",
    )


def decide_all_hypotheses(evidence: Mapping[str, Any]) -> pd.DataFrame:
    """Apply all predefined decision rules."""

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
        raise RuntimeError("Unexpected hypothesis decision.")
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
