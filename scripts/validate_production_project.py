"""Validate whether the production project is ready for Notebook 09.

Usage
-----
python scripts/validate_production_project.py
python scripts/validate_production_project.py --allow-missing
python scripts/validate_production_project.py --deep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.artifact_registry import audit_artifacts


EXPECTED_TOTAL_OBSERVATIONS = 1_450_000


def _manifest_total(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in (
        "total_observations",
        "total_rows",
        "configured_total_observations",
        "expected_total_observations",
    ):
        if key in payload:
            try:
                return int(payload[key])
            except (TypeError, ValueError):
                pass
    for key in ("components", "datasets", "dataset_components"):
        components = payload.get(key)
        if isinstance(components, dict):
            total = 0
            found = False
            for value in components.values():
                if isinstance(value, dict):
                    for count_key in ("observations", "rows", "size", "count"):
                        if count_key in value:
                            total += int(value[count_key])
                            found = True
                            break
                elif isinstance(value, (int, float)):
                    total += int(value)
                    found = True
            if found:
                return total
    return None


def validate_project(project_root: Path, *, deep: bool = False) -> tuple[pd.DataFrame, list[str]]:
    audit = audit_artifacts(project_root)
    errors: list[str] = []

    required_invalid = audit[
        audit["required_for_final"] & (~audit["valid"])
    ]
    for row in required_invalid.itertuples(index=False):
        errors.append(f"Missing or invalid required artifact: {row.name}")

    manifest_rows = audit[
        (audit["name"] == "production_dataset_manifest") & audit["valid"]
    ]
    if not manifest_rows.empty:
        manifest_path = Path(manifest_rows.iloc[0]["resolved_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        total = _manifest_total(payload)
        if total is not None and total != EXPECTED_TOTAL_OBSERVATIONS:
            errors.append(
                f"Production manifest reports {total:,} observations; "
                f"expected {EXPECTED_TOTAL_OBSERVATIONS:,}."
            )

    if deep:
        parquet_root = project_root / "data" / "generated"
        parquet_files = sorted(parquet_root.rglob("*.parquet")) if parquet_root.exists() else []
        if not parquet_files:
            errors.append("Deep validation found no generated Parquet chunks.")
        else:
            try:
                import pyarrow.parquet as pq
                total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
                if total_rows != EXPECTED_TOTAL_OBSERVATIONS:
                    errors.append(
                        f"Generated Parquet chunks contain {total_rows:,} rows; "
                        f"expected {EXPECTED_TOTAL_OBSERVATIONS:,}."
                    )
            except Exception as exc:
                errors.append(f"Could not inspect Parquet metadata: {exc}")

    return audit, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_evaluation" / "production_validation.csv",
    )
    args = parser.parse_args()

    audit, errors = validate_project(PROJECT_ROOT, deep=args.deep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)

    print(audit[["name", "required_for_final", "found", "valid", "notes"]].to_string(index=False))
    if errors:
        print("\nValidation issues:")
        for issue in errors:
            print(f"- {issue}")
    else:
        print("\nProduction project validation passed.")

    return 0 if not errors or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
