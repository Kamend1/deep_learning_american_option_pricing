"""Strict final-result export and academic-writeup handoff utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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


def _validate_export_table(name: str, table: pd.DataFrame, *, strict: bool) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Export {name!r} is not a DataFrame")
    if strict and table.empty:
        raise ValueError(f"Export table {name!r} is empty")
    if strict and len(table.columns) == 0:
        raise ValueError(f"Export table {name!r} has no columns")
    if strict and not table.empty:
        substantive = table.dropna(axis=1, how="all")
        if substantive.empty:
            raise ValueError(f"Export table {name!r} contains only missing values")


def export_final_evaluation(
    output_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    hypothesis_decisions: pd.DataFrame,
    artifact_audit: pd.DataFrame,
    summary: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Export validated final tables, decisions, audit, summary, and write-up handoff."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    required_audit = artifact_audit.loc[artifact_audit["required_for_final"]]
    invalid_required = required_audit.loc[~required_audit["valid"]]
    if strict and not invalid_required.empty:
        names = ", ".join(invalid_required["name"].astype(str))
        raise RuntimeError(f"Cannot export final evaluation; invalid artifacts: {names}")

    if strict and hypothesis_decisions.empty:
        raise ValueError("Hypothesis decisions are empty")
    if strict and hypothesis_decisions["decision"].eq("Inconclusive").any():
        unresolved = ", ".join(
            hypothesis_decisions.loc[
                hypothesis_decisions["decision"].eq("Inconclusive"),
                "hypothesis",
            ].astype(str)
        )
        raise RuntimeError(
            "Cannot export strict final evaluation with inconclusive hypotheses: "
            + unresolved
        )

    paths: dict[str, str] = {}
    for name, table in tables.items():
        _validate_export_table(name, table, strict=strict)
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)

    decisions_path = output / "hypothesis_decisions.json"
    write_json(decisions_path, hypothesis_decisions.to_dict(orient="records"))
    paths["hypothesis_decisions"] = str(decisions_path)

    audit_path = output / "artifact_audit.csv"
    artifact_audit.to_csv(audit_path, index=False)
    paths["artifact_audit"] = str(audit_path)

    summary_path = output / "final_results_summary.json"
    write_json(summary_path, dict(summary or {}))
    paths["final_results_summary"] = str(summary_path)

    writeup_path = output / "final_writeup_inputs.md"
    writeup_path.write_text(
        build_writeup_handoff(
            tables=tables,
            hypothesis_decisions=hypothesis_decisions,
            artifact_audit=artifact_audit,
            summary=summary or {},
        ),
        encoding="utf-8",
    )
    paths["final_writeup_inputs"] = str(writeup_path)
    return paths


def build_writeup_handoff(
    *,
    tables: Mapping[str, pd.DataFrame],
    hypothesis_decisions: pd.DataFrame,
    artifact_audit: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    """Create a concise Markdown handoff based only on exported evidence."""

    required = artifact_audit.loc[artifact_audit["required_for_final"]]
    valid_count = int(required["valid"].sum()) if not required.empty else 0
    total_count = int(len(required))

    lines = [
        "# Final Write-up Inputs",
        "",
        "## Production readiness",
        "",
        f"- Required valid artifacts: **{valid_count}/{total_count}**",
        f"- Final evaluation status: **{summary.get('status', 'UNKNOWN')}**",
        "",
        "## Hypothesis decisions",
        "",
    ]

    if hypothesis_decisions.empty:
        lines.append("No hypothesis decisions were exported.")
    else:
        lines.append(hypothesis_decisions.to_markdown(index=False))

    lines.extend(["", "## Exported evidence tables", ""])
    for name, table in tables.items():
        lines.extend(
            [
                f"### {name.replace('_', ' ').title()}",
                "",
                f"Rows: {len(table)}; columns: {len(table.columns)}.",
                "",
            ]
        )
        if not table.empty:
            lines.append(table.head(10).to_markdown(index=False))
            lines.append("")

    lines.extend(
        [
            "## Interpretation controls",
            "",
            "- Keep static normalized-price results separate from contract-level raw-price LSM results.",
            "- Separate up-front training cost from marginal inference or valuation cost.",
            "- Report financial lower-bound guarantees separately from empirical monotonicity checks.",
            "- Discuss OOD performance by regime before quoting an aggregate deterioration ratio.",
            "- Map literature citations manually from the supplied papers; no citation is inferred by code.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "build_writeup_handoff",
    "export_final_evaluation",
    "write_json",
]
