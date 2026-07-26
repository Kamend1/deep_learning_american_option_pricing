"""Final result export and academic-writeup handoff utilities."""

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


def export_final_evaluation(
    output_dir: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    hypothesis_decisions: pd.DataFrame,
    artifact_audit: pd.DataFrame,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Export final tables, audit, decisions, and summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=True)
        paths[name] = str(path)

    decisions_path = output / "hypothesis_decisions.json"
    write_json(
        decisions_path,
        hypothesis_decisions.to_dict(orient="records"),
    )
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
    """Create a Markdown handoff for the final academic-writing phase."""

    required = artifact_audit[artifact_audit["required_for_final"]]
    valid_count = int(required["valid"].sum()) if not required.empty else 0
    total_count = int(len(required))
    lines = [
        "# Final Write-up Inputs",
        "",
        "## Production readiness",
        "",
        f"- Required valid artifacts: **{valid_count}/{total_count}**",
        f"- Final evaluation status: **{summary.get('status', 'PENDING')}**",
        "",
        "## Hypothesis decisions",
        "",
    ]
    if hypothesis_decisions.empty:
        lines.append("PENDING — no hypothesis decisions are available.")
    else:
        lines.append(hypothesis_decisions.to_markdown(index=False))

    lines.extend(["", "## Exported comparison tables", ""])
    if not tables:
        lines.append("PENDING — no comparison tables are available.")
    else:
        for name, table in tables.items():
            lines.extend(
                [
                    f"### {name.replace('_', ' ').title()}",
                    "",
                    f"Rows: {len(table)}; columns: {len(table.columns)}.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Required interpretation",
            "",
            "- Explain the dominant pricing result without relying only on aggregate RMSE.",
            "- Discuss early-exercise-premium performance and near-zero target imbalance.",
            "- Explain boundary-region errors and false exercise versus missed exercise decisions.",
            "- Separate financial lower-bound guarantees from empirically tested monotonicity.",
            "- Compare interpolation and out-of-domain deterioration.",
            "- Separate up-front computational cost from marginal inference cost.",
            "- Compare findings with all ten supplied papers and the foundational literature.",
            "- State what the project demonstrates, suggests, and does not establish.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "build_writeup_handoff",
    "export_final_evaluation",
    "write_json",
]
