"""Validate the final Notebook 09 evidence package without rerunning it."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.final_reporting import verify_export_manifest
from src.evaluation.final_validation import assert_phase_7_8_ready


REQUIRED_BUSINESS_CASE_FILES = {
    "runtime_scaling.csv",
    "accuracy_speed_tradeoff.csv",
    "runtime_curves.csv",
    "operational_crossover.csv",
    "upfront_cost_inventory.csv",
    "upfront_cost_scenarios.csv",
    "lifecycle_break_even.csv",
    "business_case_scenarios.csv",
    "business_case_recommendations.csv",
    "business_case_readiness_audit.csv",
    "runtime_environment.json",
    "research_question_7_summary.json",
    "research_question_7.md",
    "charts/business_runtime_scaling.png",
    "charts/business_speedup_vs_crr.png",
    "charts/business_lifecycle_break_even.png",
    "charts/business_workload_scenarios.png",
}


def main() -> None:
    output = PROJECT_ROOT / "artifacts" / "final_evaluation" / "final"
    audit_path = output / "final_readiness_audit.csv"
    manifest_path = output / "final_export_manifest.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing final readiness audit: {audit_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing final export manifest: {manifest_path}")

    audit = pd.read_csv(audit_path)
    manifest = pd.read_csv(manifest_path)
    assert_phase_7_8_ready(audit)

    manifest_paths = set(manifest["relative_path"].astype(str))
    missing_business = sorted(REQUIRED_BUSINESS_CASE_FILES - manifest_paths)
    if missing_business:
        raise RuntimeError(
            "Final export is missing required business-case evidence:\n"
            + "\n".join(missing_business)
        )

    business_audit = pd.read_csv(output / "business_case_readiness_audit.csv")
    if business_audit.empty or not business_audit["valid"].astype(bool).all():
        invalid = business_audit.loc[~business_audit["valid"].astype(bool)]
        raise RuntimeError(
            "Business-case readiness failed:\n" + invalid.to_string(index=False)
        )

    verification = verify_export_manifest(output, manifest)
    if verification.empty or not verification["valid"].all():
        invalid = verification.loc[~verification["valid"]]
        raise RuntimeError(
            "Final export verification failed:\n" + invalid.to_string(index=False)
        )

    print("Final evaluation validation: PASS")
    print(f"Readiness checks: {len(audit)}")
    print(f"Business-case checks: {len(business_audit)}")
    print(f"Verified exported files: {len(verification)}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
