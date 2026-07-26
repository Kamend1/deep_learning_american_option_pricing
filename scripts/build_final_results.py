"""Build final Notebook 09 input tables from available artifacts.

The script is intentionally safe before the production runs. When metrics are
missing, it creates an auditable pending-state output rather than fabricating
results.

Usage
-----
python scripts/build_final_results.py
python scripts/build_final_results.py --strict
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

from src.evaluation.artifact_registry import (
    audit_artifacts,
    default_artifact_registry,
    resolve_artifact_path,
)
from src.evaluation.final_project_evaluation import (
    build_consolidated_pricing_table,
    metric_inventory_from_json_files,
)
from src.evaluation.final_reporting import export_final_evaluation
from src.evaluation.hypothesis_testing import decide_all_hypotheses


def _available_json_paths(project_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in default_artifact_registry():
        path = resolve_artifact_path(project_root, spec)
        if path is not None and path.suffix.lower() == ".json":
            paths[spec.name] = path
    return paths


def _load_evidence(project_root: Path) -> dict[str, object]:
    path = project_root / "artifacts" / "final_evaluation" / "hypothesis_evidence.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_evaluation",
    )
    args = parser.parse_args()

    audit = audit_artifacts(PROJECT_ROOT)
    inventory = metric_inventory_from_json_files(_available_json_paths(PROJECT_ROOT))
    evidence = _load_evidence(PROJECT_ROOT)
    decisions = decide_all_hypotheses(evidence)

    metrics_by_model: dict[str, dict[str, object]] = {}
    if not inventory.empty:
        for source, group in inventory.groupby("source", sort=True):
            metrics_by_model[source] = dict(zip(group["metric"], group["value"]))
    pricing_table = build_consolidated_pricing_table(metrics_by_model)

    valid_required = audit.loc[audit["required_for_final"], "valid"]
    status = (
        "READY_FOR_FINAL_WRITEUP"
        if len(valid_required) > 0 and bool(valid_required.all())
        else "PENDING_ARTIFACTS"
    )
    summary = {
        "status": status,
        "required_artifacts": int(valid_required.size),
        "valid_required_artifacts": int(valid_required.sum()),
        "metric_inventory_rows": int(len(inventory)),
    }

    export_final_evaluation(
        args.output_dir,
        tables={
            "metric_inventory": inventory,
            "consolidated_model_metrics": pricing_table,
        },
        hypothesis_decisions=decisions,
        artifact_audit=audit,
        summary=summary,
    )

    print(f"Final evaluation inputs written to: {args.output_dir}")
    print(f"Status: {status}")
    if args.strict and status != "READY_FOR_FINAL_WRITEUP":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
