<!-- Generated from notebooks/09_final_evaluation.ipynb. The notebook is the executable source of truth. -->

# Final Evaluation, Hypothesis Decisions, and Project Synthesis

**Notebook 09**

This notebook is the final analytical layer of the project. It does not train
new models. It audits the complete artifact chain, aligns the results of all
static and simulation-based methods, produces consolidated comparison tables,
applies predefined H1-H6 decision rules, and creates the handoff for the final
academic paper.

The notebook is intentionally executable before the expensive production runs.
Unavailable artifacts are shown as **PENDING**. No placeholder value is treated
as an empirical result.

## Final research question

> Can deep learning provide an accurate, financially coherent, and
> computationally efficient surrogate-pricing framework for American put
> options, while also learning the early-exercise decision and remaining
> transparent about out-of-domain limitations?

The final answer must distinguish:

- what the completed experiments demonstrate;
- what the evidence merely suggests;
- what cannot be established from synthetic model-generated prices.

## 1. Environment and final-evaluation design

```python
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR

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

DESIGN_PATH = PROJECT_ROOT / "data" / "manifests" / "final_evaluation_design.json"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_evaluation"

design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
design
```

### Section conclusion — final design

**Complete after execution:** Confirm that the design manifest matches the
actual model registry, output directory, hypothesis definitions, and production
dataset size. Record any justified deviation before interpreting results.

## 2. Artifact audit

```python
artifact_audit = audit_artifacts(PROJECT_ROOT)
artifact_audit[
    [
        "name",
        "category",
        "required_for_final",
        "found",
        "valid",
        "resolved_path",
        "notes",
    ]
]
```

```python
required_audit = artifact_audit[artifact_audit["required_for_final"]]
valid_required = int(required_audit["valid"].sum())
total_required = int(len(required_audit))

print(f"Valid required artifacts: {valid_required}/{total_required}")
if valid_required < total_required:
    print("PENDING — Notebook 09 is running in skeleton mode.")
else:
    print("All registered required artifacts passed the basic audit.")
```

### Section conclusion — artifact readiness

**Complete after execution:** Identify all missing, stale, or invalid artifacts.
Do not continue to final hypothesis decisions until the critical production
artifacts are complete and aligned.

## 3. Metric inventory

```python
json_paths = {}
for spec in default_artifact_registry():
    path = resolve_artifact_path(PROJECT_ROOT, spec)
    if path is not None and path.suffix.lower() == ".json":
        json_paths[spec.name] = path

metric_inventory = metric_inventory_from_json_files(json_paths)
metric_inventory.head(30)
```

The inventory is deliberately long-form. It exposes the actual metric keys saved
by earlier notebooks and prevents Notebook 09 from silently assuming that all
model outputs use identical naming conventions.

## 4. Consolidated in-domain pricing comparison

```python
metrics_by_model = {}
if not metric_inventory.empty:
    for source, group in metric_inventory.groupby("source", sort=True):
        metrics_by_model[source] = dict(zip(group["metric"], group["value"]))

consolidated_pricing = build_consolidated_pricing_table(metrics_by_model)
consolidated_pricing
```

### Interpretation placeholder

Discuss:

- the strongest model by MAE and RMSE;
- whether the ranking changes under maximum error or tolerance-band coverage;
- whether the integrated model improves on the best standalone residual model;
- whether any improvement is economically material rather than merely
  statistically detectable.

## 5. Financial-consistency comparison

```python
financial_consistency = pd.DataFrame(
    columns=[
        "negative_price_violations",
        "below_intrinsic_violations",
        "below_european_violations",
        "spot_monotonicity_violation_rate",
        "volatility_monotonicity_violation_rate",
        "decision_disagreement_rate",
    ]
)
financial_consistency.index.name = "model"
financial_consistency
```

### Interpretation placeholder

Populate this table from the prediction artifacts. Distinguish between:

- architectural guarantees, such as the constrained lower bound;
- empirical properties, such as monotonicity on tested grids;
- internal contradictions between price, continuation, and classification heads.

## 6. Exercise-boundary comparison

```python
boundary_comparison = pd.DataFrame(
    columns=[
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "false_exercise_rate",
        "missed_exercise_rate",
        "boundary_location_error",
        "near_boundary_price_mae",
    ]
)
boundary_comparison.index.name = "model"
boundary_comparison
```

### Interpretation placeholder

Compare CRR, Step 6, the final integrated classifier head, the integrated
continuation-implied decision, classical LSM, and neural LSM. Explain whether
errors are symmetric: a false early exercise and a missed exercise opportunity
can have different economic implications.

## 7. Out-of-domain deterioration

```python
ood_results = pd.DataFrame(
    columns=[
        "model",
        "regime",
        "in_domain_mae",
        "ood_mae",
        "ood_deterioration",
    ]
)
ood_results
```

### Interpretation placeholder

Report each OOD regime separately. Do not use one aggregate ratio to conceal
material differences between extreme volatility, moneyness, maturity, and
rate/dividend combinations.

## 8. Runtime and computational efficiency

```python
runtime_comparison = pd.DataFrame(
    columns=[
        "model",
        "device",
        "batch_size",
        "observations",
        "seconds",
        "seconds_per_observation",
        "observations_per_second",
        "up_front_training_seconds",
    ]
)
runtime_comparison
```

### Interpretation placeholder

Separate:

- analytical Black–Scholes runtime;
- CRR numerical-pricing runtime;
- classical LSM simulation and regression;
- neural LSM policy training and marginal valuation;
- static neural-model training;
- marginal CPU and GPU inference.

The speed claim must not ignore the up-front label-generation and training cost.

## 9. Static-model ablation

```python
static_ablation = pd.DataFrame(
    columns=[
        "direct_learning",
        "residual_learning",
        "nonnegative_output",
        "financial_floor",
        "exercise_head",
        "continuation_head",
        "consistency_penalty",
        "shared_backbone",
        "test_mae",
        "boundary_f1",
        "financial_violation_rate",
    ]
)
static_ablation.index.name = "model"
static_ablation
```

### Interpretation placeholder

Use the standalone models as controlled ablations. The final integrated model
should not be credited for an improvement unless the comparison isolates which
design feature added value.

## 10. Classical and neural Longstaff–Schwartz

```python
lsm_comparison = pd.DataFrame(
    columns=[
        "method",
        "contracts",
        "mae_vs_crr",
        "rmse_vs_crr",
        "ci_coverage",
        "policy_agreement",
        "boundary_error",
        "training_seconds",
        "valuation_seconds",
    ]
)
lsm_comparison
```

### Interpretation placeholder

Explain the distinct role of neural LSM. It is a path-based policy-learning
method, not another head of the static integrated model. Compare accuracy,
uncertainty, policy stability, and computational burden.

## 11. Predefined H1-H6 decisions

```python
EVIDENCE_PATH = FINAL_OUTPUT_DIR / "hypothesis_evidence.json"

if EVIDENCE_PATH.exists():
    hypothesis_evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
else:
    hypothesis_evidence = {}
    print(
        "PENDING — create hypothesis_evidence.json from the final aligned "
        "tables before formal decisions."
    )

hypothesis_decisions = decide_all_hypotheses(hypothesis_evidence)
hypothesis_decisions
```

### Academic judgment placeholder

The automated rules provide consistency with the pre-registered design, but the
written decision must also discuss:

- economic materiality;
- uncertainty and Monte Carlo error;
- segmented and OOD evidence;
- conflicting metrics;
- limitations of the synthetic benchmark.

## 12. Literature synthesis matrix

```python
literature_synthesis = pd.DataFrame(
    columns=[
        "project_finding",
        "citation_key",
        "agreement_or_disagreement",
        "possible_explanation",
        "scope_limitation",
    ]
)
literature_synthesis
```

Populate the matrix with all ten supplied papers and the foundational option
pricing sources. Every comparison must be tied to a reported result from this
project.

## 13. Export final-evaluation inputs

```python
tables_to_export = {
    "consolidated_model_metrics": consolidated_pricing,
    "metric_inventory": metric_inventory,
    "financial_consistency_table": financial_consistency,
    "boundary_comparison": boundary_comparison,
    "ood_results": ood_results,
    "runtime_comparison": runtime_comparison,
    "static_model_ablation": static_ablation,
    "lsm_comparison": lsm_comparison,
    "literature_synthesis": literature_synthesis,
}

status = (
    "READY_FOR_FINAL_WRITEUP"
    if total_required > 0 and valid_required == total_required
    else "PENDING_ARTIFACTS"
)

exported_paths = export_final_evaluation(
    FINAL_OUTPUT_DIR,
    tables=tables_to_export,
    hypothesis_decisions=hypothesis_decisions,
    artifact_audit=artifact_audit,
    summary={
        "status": status,
        "valid_required_artifacts": valid_required,
        "required_artifacts": total_required,
        "expected_total_observations": design["expected_total_observations"],
    },
)

pd.Series(exported_paths, name="path").to_frame()
```

# Final project conclusion — skeleton

## Research objective

Summarize the original research question and explain the progression from
Black–Scholes to CRR, static neural surrogates, explicit boundary learning,
neural Longstaff–Schwartz, and the integrated multi-head model.

## Principal empirical findings

Insert the main numerical findings. Identify the strongest static model, the
strongest simulation-based model, and the relevant benchmarks.

## Pricing accuracy

Discuss in-domain pricing performance, segmented error, tail error, and
economic materiality.

## Early-exercise premium

Explain whether residual learning improved the direct-price baseline and how
the near-zero premium distribution affected training.

## Exercise boundary

Discuss classification, continuation-value estimation, near-boundary pricing,
and boundary-location error.

## Financial consistency

Separate hard architectural guarantees from empirical monotonicity and
cross-head consistency.

## Out-of-domain robustness

Explain each OOD regime and decide whether extrapolation risk is material.

## Computational efficiency

Separate up-front computation from marginal valuation cost. State the device,
batch size, and measurement protocol.

## Comparison with prior literature

Compare the findings with all ten supplied papers and the foundational sources.
Explain both agreements and disagreements.

## Hypothesis decisions

Present H1-H6 with primary evidence, secondary evidence, and limitations.

## Practical implications

Explain where a trained neural surrogate could be useful and where numerical
pricing remains necessary.

## Limitations

Address synthetic labels, constant volatility, continuous dividends, model
risk, finite-tree approximation, Monte Carlo noise, and generalization.

## Future research

Consider implied-volatility surfaces, discrete dividends, stochastic volatility,
real option quotes, stronger arbitrage constraints, and deployment benchmarking.

## Final answer to the research question

Provide one direct, evidence-based conclusion. Distinguish what the data
demonstrate, suggest, and do not establish.

# Notebook conclusion and project handoff

## Artifacts produced

List all final comparison tables, hypothesis decisions, figure manifests, and
write-up inputs.

## Remaining work

After the production runs, the remaining work should consist only of:

1. filling section-level interpretation cells;
2. completing notebook conclusions and handoffs in Notebooks 01–08;
3. refreshing Markdown twins;
4. finalizing this Notebook 09 conclusion;
5. updating README with actual results;
6. producing the final academic paper or PDF.
