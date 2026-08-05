"""Predefined H1-H6 decision rules for the final project evaluation."""

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


@dataclass(frozen=True, slots=True)
class HypothesisDecision:
    hypothesis: str
    decision: str
    primary_evidence: str
    secondary_evidence: str
    threshold: str
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
            "Required common-test MAE values are unavailable.",
            "",
            "Direct MLP MAE / Black–Scholes proxy MAE <= 0.95.",
            "No decision is inferred from missing evidence.",
        )
    baseline = float(evidence["black_scholes_mae"])
    model = float(evidence["direct_mlp_mae"])
    ratio = model / baseline if baseline > 0.0 else math.inf
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
        f"Common-test MAE ratio = {ratio:.6f}.",
        f"Direct MLP MAE={model:.8g}; Black–Scholes proxy MAE={baseline:.8g}.",
        "Supported requires a ratio no greater than 0.95.",
        "The result concerns the fixed synthetic in-domain test set.",
    )


def decide_h2(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_mlp_mae", "selected_residual_mae"):
        return HypothesisDecision(
            "H2",
            "Inconclusive",
            "The direct or validation-selected residual MAE is unavailable.",
            "",
            "Selected residual MAE / Direct MLP MAE <= 0.98.",
            "The selected residual model must come from Notebook 05 validation selection.",
        )
    direct = float(evidence["direct_mlp_mae"])
    residual = float(evidence["selected_residual_mae"])
    ratio = residual / direct if direct > 0.0 else math.inf
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
        f"Selected residual/direct MAE ratio = {ratio:.6f}.",
        f"Selected residual={evidence.get('selected_residual_model')}; MAE={residual:.8g}.",
        "Supported requires a ratio no greater than 0.98.",
        "The comparison uses the same aligned test observations and validation-selected model.",
    )


def decide_h3(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "direct_violation_rate", "constrained_violation_rate"):
        return HypothesisDecision(
            "H3",
            "Inconclusive",
            "Comparable lower-bound violation rates are unavailable.",
            "",
            "Zero constrained violations with positive direct-model violations.",
            "Architectural lower bounds do not prove every possible monotonicity property.",
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
        f"Direct violation rate={direct:.8g}; constrained rate={constrained:.8g}.",
        "Rates are recomputed from the common Phase 4 prediction matrix.",
        "Supported requires zero constrained violations and positive direct violations.",
        "The decision covers non-negativity, European, intrinsic, and combined floor checks.",
    )


def decide_h4(evidence: Mapping[str, Any]) -> HypothesisDecision:
    required = (
        "classifier_boundary_f1",
        "multitask_boundary_f1",
        "price_only_boundary_mae",
        "multitask_boundary_mae",
    )
    if not _finite(evidence, *required):
        return HypothesisDecision(
            "H4",
            "Inconclusive",
            "Notebook 06 boundary evidence is incomplete.",
            "",
            "Multi-task classification must be non-inferior and boundary pricing must improve.",
            "Notebook 08 is supporting integration evidence, not the primary H4 experiment.",
        )

    classifier_f1 = float(evidence["classifier_boundary_f1"])
    multitask_f1 = float(evidence["multitask_boundary_f1"])
    price_only_mae = float(evidence["price_only_boundary_mae"])
    multitask_mae = float(evidence["multitask_boundary_mae"])
    allowed_f1_degradation = float(evidence.get("allowed_f1_degradation", 0.001))
    required_mae_improvement = float(evidence.get("required_mae_improvement", 0.01))

    classification_pass = multitask_f1 >= classifier_f1 - allowed_f1_degradation
    relative_mae_improvement = (
        (price_only_mae - multitask_mae) / price_only_mae
        if price_only_mae > 0.0
        else -math.inf
    )
    pricing_pass = relative_mae_improvement >= required_mae_improvement

    if classification_pass and pricing_pass:
        decision = "Supported"
    elif classification_pass or pricing_pass:
        decision = "Partially supported"
    else:
        decision = "Not supported"

    integrated_f1 = evidence.get("integrated_exercise_f1")
    specialist_f1 = evidence.get("specialist_exercise_f1")
    secondary = ""
    if _finite(
        {"integrated": integrated_f1, "specialist": specialist_f1},
        "integrated",
        "specialist",
    ):
        secondary = (
            "Notebook 08 integrated exercise F1="
            f"{float(integrated_f1):.6f}; specialist F1={float(specialist_f1):.6f}."
        )

    return HypothesisDecision(
        "H4",
        decision,
        (
            f"Notebook 06 multi-task F1 change={multitask_f1-classifier_f1:.6f}; "
            f"relative boundary-price MAE improvement={relative_mae_improvement:.4%}."
        ),
        secondary,
        (
            f"Allowed F1 degradation={allowed_f1_degradation:.6f}; required "
            f"boundary-price improvement={required_mae_improvement:.2%}."
        ),
        "Exercise labels and boundary distances are generated by the CRR reference model.",
    )


def decide_h5(evidence: Mapping[str, Any]) -> HypothesisDecision:
    if not _finite(evidence, "crr_seconds_per_option", "static_seconds_per_option"):
        return HypothesisDecision(
            "H5",
            "Inconclusive",
            "Comparable static-neural and CRR marginal runtimes are unavailable.",
            "",
            "Static neural seconds per option / CRR seconds per option <= 0.50.",
            "Training and data-generation costs are excluded from marginal inference.",
        )
    crr = float(evidence["crr_seconds_per_option"])
    static = float(evidence["static_seconds_per_option"])
    ratio = static / crr if crr > 0.0 else math.inf
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
        f"Selected static neural / CRR marginal-runtime ratio = {ratio:.8g}.",
        (
            f"Static model={evidence.get('h5_static_model')}; "
            f"static={static:.8g}s, CRR={crr:.8g}s per option."
        ),
        "Supported requires a ratio no greater than 0.50.",
        "Batch size, hardware, and implementation affect absolute runtime; LSM timing is reported separately.",
    )


def decide_h6(evidence: Mapping[str, Any]) -> HypothesisDecision:
    required = (
        "h6_eligible_models",
        "h6_models_at_or_above_1_25",
        "h6_models_above_1_0",
        "h6_minimum_aggregate_ratio",
    )
    if not _finite(evidence, *required):
        return HypothesisDecision(
            "H6",
            "Inconclusive",
            "Per-model aggregate OOD deterioration evidence is unavailable.",
            "",
            "Every eligible static neural pricing model must have aggregate OOD/in-domain MAE >= 1.25.",
            "The aggregate can conceal regime-specific improvements or failures.",
        )
    total = int(float(evidence["h6_eligible_models"]))
    material = int(float(evidence["h6_models_at_or_above_1_25"]))
    worse = int(float(evidence["h6_models_above_1_0"]))
    minimum = float(evidence["h6_minimum_aggregate_ratio"])
    if total <= 0:
        decision = "Inconclusive"
    elif material == total:
        decision = "Supported"
    elif worse == total or material / total >= 0.75:
        decision = "Partially supported"
    else:
        decision = "Not supported"
    return HypothesisDecision(
        "H6",
        decision,
        (
            f"Eligible models with aggregate ratio >=1.25: {material}/{total}; "
            f"models above 1.0: {worse}/{total}; minimum ratio={minimum:.6f}."
        ),
        str(evidence.get("h6_model_ratios", "")),
        "Supported requires every eligible model's aggregate OOD/in-domain MAE ratio to be at least 1.25.",
        "The decision applies to the four predefined synthetic OOD regimes, not arbitrary market data.",
    )


def decide_all_hypotheses(evidence: Mapping[str, Any]) -> pd.DataFrame:
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
