"""Strict project-readiness checks for Notebook 09 Phase 8."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_HYPOTHESES = {"H1", "H2", "H3", "H4", "H5", "H6"}
EXPECTED_RECOMMENDATION_TASKS = {
    "Most accurate static price",
    "Most accurate exercise decision",
    "One model for both price and exercise",
    "Path-based valuation",
    "Lowest absolute error outside the training range",
    "Repeated large-batch pricing",
}
EXPECTED_RUNTIME_FAMILIES = {
    "static neural inference",
    "numerical valuation",
    "path-based valuation",
    "up-front training",
}


def _check(name: str, valid: bool, details: str) -> dict[str, Any]:
    return {"check": name, "valid": bool(valid), "details": str(details)}


def _boolean_status(
    frame: pd.DataFrame,
    *,
    label: str,
    accepted_columns: tuple[str, ...],
) -> tuple[pd.Series, str]:
    """Return the first available boolean status column and its name.

    Different audit tables intentionally use different semantic names.
    Row-level artifact and package checks use ``valid``; shared-field
    comparisons use ``matches``. The final gate must preserve those
    upstream schemas rather than requiring every table to be renamed.
    """

    for column in accepted_columns:
        if column in frame.columns:
            return frame[column].fillna(False).astype(bool), column
    raise KeyError(
        f"{label} must contain one of {list(accepted_columns)}. "
        f"Available columns: {frame.columns.tolist()}"
    )


def build_pre_export_readiness_audit(
    *,
    artifact_audit: pd.DataFrame,
    package_coherence: pd.DataFrame,
    static_prediction_alignment: pd.DataFrame,
    static_field_alignment: pd.DataFrame,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
    task_recommendations: pd.DataFrame,
    integrated_model_tradeoff: pd.DataFrame,
    project_findings: pd.DataFrame,
    project_limitations: pd.DataFrame,
    final_results_summary: Mapping[str, Any],
    chart_paths: Mapping[str, Path],
) -> pd.DataFrame:
    """Validate all evidence and final interpretation before writing exports."""

    checks: list[dict[str, Any]] = []

    required = artifact_audit.loc[artifact_audit["required_for_final"].astype(bool)]
    invalid_required = required.loc[~required["valid"].astype(bool)]
    checks.append(
        _check(
            "required_artifacts_valid",
            invalid_required.empty,
            f"invalid_required={len(invalid_required)}",
        )
    )

    invalid_packages = package_coherence.loc[~package_coherence["valid"].astype(bool)]
    checks.append(
        _check(
            "package_coherence_valid",
            invalid_packages.empty,
            f"invalid_checks={len(invalid_packages)}",
        )
    )

    invalid_alignment = static_prediction_alignment.loc[
        ~static_prediction_alignment["valid"].astype(bool)
    ]
    checks.append(
        _check(
            "static_prediction_alignment_valid",
            invalid_alignment.empty,
            f"invalid_rows={len(invalid_alignment)}",
        )
    )

    field_status, field_status_column = _boolean_status(
        static_field_alignment,
        label="static_field_alignment",
        accepted_columns=("matches", "valid"),
    )
    invalid_fields = static_field_alignment.loc[~field_status]
    checks.append(
        _check(
            "static_field_alignment_valid",
            invalid_fields.empty,
            (
                f"status_column={field_status_column}; "
                f"invalid_rows={len(invalid_fields)}"
            ),
        )
    )

    pricing_valid = (
        not static_model_metrics.empty
        and {"model_id", "normalized_mae", "pricing_rank"}.issubset(
            static_model_metrics.columns
        )
        and pd.to_numeric(
            static_model_metrics["normalized_mae"], errors="coerce"
        ).notna().all()
    )
    checks.append(
        _check(
            "static_pricing_complete",
            pricing_valid,
            f"models={len(static_model_metrics)}",
        )
    )

    constrained = static_financial_consistency.loc[
        static_financial_consistency.get("financially_constrained", False).astype(bool)
    ]
    constrained_zero = (
        not constrained.empty
        and pd.to_numeric(
            constrained["below_financial_floor_rate"], errors="coerce"
        ).fillna(np.inf).eq(0.0).all()
    )
    checks.append(
        _check(
            "constrained_prices_have_zero_floor_violations",
            constrained_zero,
            f"constrained_models={len(constrained)}",
        )
    )

    exercise_ids = set(exercise_model_metrics.get("model_id", pd.Series(dtype=str)).astype(str))
    exercise_valid = len(exercise_model_metrics) >= 6 and {
        "exercise_only_classifier",
        "multitask_exercise_head",
        "integrated_warm_start_exercise_head",
        "integrated_warm_start_continuation_path",
        "integrated_scratch_exercise_head",
        "integrated_scratch_continuation_path",
    }.issubset(exercise_ids)
    checks.append(
        _check(
            "exercise_paths_complete",
            exercise_valid,
            f"models={sorted(exercise_ids)}",
        )
    )

    eligible_ood = static_ood_model_summary.loc[
        static_ood_model_summary.get("h6_eligible", False).astype(bool)
    ]
    ood_valid = (
        not eligible_ood.empty
        and pd.to_numeric(eligible_ood["regimes"], errors="coerce").eq(4).all()
        and pd.to_numeric(
            eligible_ood["aggregate_ood_to_in_domain_ratio"], errors="coerce"
        ).notna().all()
    )
    checks.append(
        _check(
            "ood_evidence_complete",
            ood_valid,
            f"eligible_models={len(eligible_ood)}",
        )
    )

    runtime_families = set(runtime_comparison["benchmark_family"].astype(str))
    checks.append(
        _check(
            "runtime_families_separated",
            EXPECTED_RUNTIME_FAMILIES.issubset(runtime_families),
            f"families={sorted(runtime_families)}",
        )
    )

    hypotheses = set(hypothesis_decisions["hypothesis"].astype(str))
    inconclusive = hypothesis_decisions.loc[
        hypothesis_decisions["decision"].astype(str).eq("Inconclusive")
    ]
    checks.append(
        _check(
            "hypotheses_complete",
            hypotheses == EXPECTED_HYPOTHESES and inconclusive.empty,
            f"hypotheses={sorted(hypotheses)}; inconclusive={len(inconclusive)}",
        )
    )

    tasks = set(task_recommendations["task"].astype(str))
    checks.append(
        _check(
            "task_recommendations_complete",
            EXPECTED_RECOMMENDATION_TASKS.issubset(tasks),
            f"tasks={sorted(tasks)}",
        )
    )

    tradeoff_dimensions = set(integrated_model_tradeoff["dimension"].astype(str))
    expected_tradeoffs = {
        "Pricing error",
        "Exercise F1",
        "Financial lower-bound violations",
        "Inference throughput",
        "Continuation-implied decision F1",
    }
    checks.append(
        _check(
            "integrated_tradeoff_complete",
            expected_tradeoffs.issubset(tradeoff_dimensions),
            f"dimensions={sorted(tradeoff_dimensions)}",
        )
    )

    checks.append(
        _check(
            "interpretation_tables_nonempty",
            not project_findings.empty and not project_limitations.empty,
            f"findings={len(project_findings)}; limitations={len(project_limitations)}",
        )
    )

    summary_valid = (
        final_results_summary.get("status") == "complete"
        and final_results_summary.get("universal_preferred_model") is None
        and bool(final_results_summary.get("overall_answer"))
    )
    checks.append(
        _check(
            "final_summary_task_specific",
            summary_valid,
            (
                f"status={final_results_summary.get('status')}; "
                f"universal={final_results_summary.get('universal_preferred_model')}"
            ),
        )
    )

    missing_charts = [
        name
        for name, path in chart_paths.items()
        if not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    checks.append(
        _check(
            "final_charts_written",
            len(chart_paths) >= 5 and not missing_charts,
            f"charts={len(chart_paths)}; missing={missing_charts}",
        )
    )

    return pd.DataFrame(checks)


def build_post_export_readiness_audit(
    export_manifest: pd.DataFrame,
    *,
    required_relative_paths: Iterable[str],
    export_verification: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = set(map(str, required_relative_paths))
    present = set(export_manifest.get("relative_path", pd.Series(dtype=str)).astype(str))
    missing = sorted(required.difference(present))
    nonpositive = export_manifest.loc[
        pd.to_numeric(export_manifest.get("bytes"), errors="coerce").fillna(0).le(0)
    ]
    missing_hash = export_manifest.loc[
        export_manifest.get("sha256", pd.Series(dtype=str)).astype(str).str.len().ne(64)
    ]
    verification_valid = (
        export_verification is None
        or (not export_verification.empty and export_verification["valid"].astype(bool).all())
    )
    verification_invalid = (
        0
        if export_verification is None
        else int((~export_verification["valid"].astype(bool)).sum())
    )
    return pd.DataFrame(
        [
            _check(
                "required_final_exports_present",
                not missing,
                f"missing={missing}",
            ),
            _check(
                "final_exports_nonempty",
                nonpositive.empty,
                f"empty_files={len(nonpositive)}",
            ),
            _check(
                "final_export_hashes_complete",
                missing_hash.empty,
                f"invalid_hashes={len(missing_hash)}",
            ),
            _check(
                "final_export_files_match_manifest",
                verification_valid,
                f"invalid_files={verification_invalid}",
            ),
        ]
    )


def assert_phase_7_8_ready(readiness_audit: pd.DataFrame) -> None:
    if readiness_audit.empty:
        raise ValueError("Final readiness audit is empty.")
    invalid = readiness_audit.loc[~readiness_audit["valid"].astype(bool)]
    if invalid.empty:
        return
    details = "; ".join(
        f"{row.check}: {row.details}"
        for row in invalid.itertuples(index=False)
    )
    raise RuntimeError("Phases 7–8 readiness failed: " + details)


__all__ = [
    "EXPECTED_HYPOTHESES",
    "EXPECTED_RECOMMENDATION_TASKS",
    "EXPECTED_RUNTIME_FAMILIES",
    "assert_phase_7_8_ready",
    "build_post_export_readiness_audit",
    "build_pre_export_readiness_audit",
]
