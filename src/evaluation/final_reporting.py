"""Strict final exports for Notebook 09 Phases 7 and 8."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FINAL_TABLES = (
    "task_recommendations",
    "integrated_model_tradeoff",
    "project_findings",
    "project_limitations",
    "hypothesis_decisions",
    "static_model_metrics",
    "static_financial_consistency",
    "exercise_model_metrics",
    "exercise_boundary_metrics",
    "static_ood_model_summary",
    "lsm_heldout_pricing",
    "runtime_comparison",
    "runtime_scaling",
    "accuracy_speed_tradeoff",
    "runtime_curves",
    "operational_crossover",
    "upfront_cost_inventory",
    "upfront_cost_scenarios",
    "lifecycle_break_even",
    "business_case_scenarios",
    "business_case_recommendations",
    "business_case_readiness_audit",
)

REQUIRED_CHARTS = (
    "static_pricing_mae",
    "exercise_f1",
    "ood_deterioration",
    "runtime_comparison",
    "lsm_heldout_mae",
    "business_runtime_scaling",
    "business_speedup_vs_crr",
    "business_lifecycle_break_even",
    "business_workload_scenarios",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_table(name: str, table: pd.DataFrame) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Final export {name!r} is not a DataFrame.")
    if table.empty:
        raise ValueError(f"Final export {name!r} is empty.")
    if table.dropna(axis=1, how="all").empty:
        raise ValueError(f"Final export {name!r} contains only missing values.")


def _frame_markdown(table: pd.DataFrame, *, max_rows: int = 20) -> str:
    view = table.head(max_rows)
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return "```text\n" + view.to_string(index=False) + "\n```"


def build_final_writeup_inputs(
    *,
    final_conclusion_markdown: str,
    task_recommendations: pd.DataFrame,
    integrated_model_tradeoff: pd.DataFrame,
    project_findings: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
    final_results_summary: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Final Write-up Inputs",
            "",
            f"Status: **{final_results_summary.get('status', 'unknown')}**",
            "",
            final_conclusion_markdown.strip(),
            "",
            "## Task-specific recommendations",
            "",
            _frame_markdown(task_recommendations),
            "",
            "## Integrated-model trade-off",
            "",
            _frame_markdown(integrated_model_tradeoff),
            "",
            "## Main project findings",
            "",
            _frame_markdown(project_findings),
            "",
            "## Hypothesis decisions",
            "",
            _frame_markdown(hypothesis_decisions),
            "",
        ]
    )


def export_final_project(
    output_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    final_results_summary: Mapping[str, Any],
    final_conclusion_markdown: str,
    chart_paths: Mapping[str, Path],
    readiness_audit: pd.DataFrame,
) -> dict[str, Any]:
    """Write the final evidence package and return all paths except the manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    missing_tables = sorted(set(REQUIRED_FINAL_TABLES).difference(tables))
    if missing_tables:
        raise KeyError("Missing required final tables: " + ", ".join(missing_tables))
    for name in REQUIRED_FINAL_TABLES:
        _validate_table(name, tables[name])

    missing_charts = sorted(set(REQUIRED_CHARTS).difference(chart_paths))
    if missing_charts:
        raise KeyError("Missing required final charts: " + ", ".join(missing_charts))
    for name in REQUIRED_CHARTS:
        path = Path(chart_paths[name])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required final chart is missing or empty: {name}: {path}")

    if final_results_summary.get("status") != "complete":
        raise ValueError("Final results summary must have status='complete'.")
    if not final_conclusion_markdown.strip():
        raise ValueError("Final conclusion markdown is empty.")
    _validate_table("final_readiness_audit", readiness_audit)

    paths: dict[str, str] = {}
    for name, table in tables.items():
        _validate_table(name, table)
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)

    summary_path = write_json(
        output / "final_results_summary.json",
        dict(final_results_summary),
    )
    paths["final_results_summary"] = str(summary_path)

    decisions_path = write_json(
        output / "hypothesis_decisions.json",
        tables["hypothesis_decisions"].to_dict(orient="records"),
    )
    paths["hypothesis_decisions_json"] = str(decisions_path)

    conclusion_path = output / "final_project_conclusion.md"
    conclusion_path.write_text(final_conclusion_markdown.strip() + "\n", encoding="utf-8")
    paths["final_project_conclusion"] = str(conclusion_path)

    writeup_path = output / "final_writeup_inputs.md"
    writeup_path.write_text(
        build_final_writeup_inputs(
            final_conclusion_markdown=final_conclusion_markdown,
            task_recommendations=tables["task_recommendations"],
            integrated_model_tradeoff=tables["integrated_model_tradeoff"],
            project_findings=tables["project_findings"],
            hypothesis_decisions=tables["hypothesis_decisions"],
            final_results_summary=final_results_summary,
        ),
        encoding="utf-8",
    )
    paths["final_writeup_inputs"] = str(writeup_path)

    audit_path = output / "final_readiness_audit.csv"
    readiness_audit.to_csv(audit_path, index=False)
    paths["final_readiness_audit"] = str(audit_path)

    for name, path in chart_paths.items():
        paths[f"chart_{name}"] = str(Path(path))

    return {
        "paths": paths,
        "output_dir": str(output),
    }


def build_export_manifest(output_dir: Path) -> pd.DataFrame:
    """Inventory every final output except the manifest itself."""

    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    if not output.is_dir():
        return pd.DataFrame(columns=["relative_path", "bytes", "sha256"])
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"final_export_manifest.csv", "final_export_manifest.json"}:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows)



def verify_export_manifest(
    output_dir: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Verify that each manifest entry still exists with the recorded size and hash."""

    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        relative = str(record.get("relative_path"))
        path = output / relative
        exists = path.is_file()
        actual_bytes = int(path.stat().st_size) if exists else None
        actual_sha256 = file_sha256(path) if exists else None
        expected_bytes = int(record.get("bytes")) if record.get("bytes") is not None else None
        expected_sha256 = str(record.get("sha256"))
        rows.append(
            {
                "relative_path": relative,
                "exists": exists,
                "bytes_match": exists and actual_bytes == expected_bytes,
                "sha256_match": exists and actual_sha256 == expected_sha256,
                "valid": (
                    exists
                    and actual_bytes == expected_bytes
                    and actual_sha256 == expected_sha256
                ),
            }
        )
    return pd.DataFrame(rows)


def write_export_manifest(output_dir: Path) -> pd.DataFrame:
    output = Path(output_dir)
    manifest = build_export_manifest(output)
    if manifest.empty:
        raise ValueError("Final export manifest would be empty.")
    manifest.to_csv(output / "final_export_manifest.csv", index=False)
    write_json(
        output / "final_export_manifest.json",
        manifest.to_dict(orient="records"),
    )
    return manifest


__all__ = [
    "REQUIRED_CHARTS",
    "REQUIRED_FINAL_TABLES",
    "build_export_manifest",
    "build_final_writeup_inputs",
    "export_final_project",
    "file_sha256",
    "verify_export_manifest",
    "write_export_manifest",
    "write_json",
]
