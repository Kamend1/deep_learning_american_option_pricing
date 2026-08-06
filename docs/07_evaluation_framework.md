# Evaluation Framework

## Purpose

The project evaluates more than aggregate predictive error. A useful American put surrogate must be accurate, financially coherent, informative near the exercise boundary, robust within its declared scope, and operationally faster for relevant workloads.

## Common-test principle

Static models from Notebooks 04, 05, 06, and 08 are compared on the same 187,811 held-out observations.

Notebook 09 aligns every prediction table by `sample_id` and verifies the same true target before calculating cross-model comparisons.

The path-based Notebook 07 sample is separate and is not inserted into the static leaderboard.

## Pricing metrics

Primary metrics include:

### Mean absolute error

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\widehat y_i|.
$$

### Root mean squared error

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\widehat y_i)^2
}.
$$

### Median absolute error

Robust summary of the typical error.

### Maximum absolute error

Tail diagnostic that prevents a small average from hiding an extreme miss.

### Coefficient of determination

$$
R^2 =
1-
\frac{
\sum_i(y_i-\widehat y_i)^2
}{
\sum_i(y_i-\overline y)^2
}.
$$

### Tolerance-band coverage

The share of observations with absolute error below predefined normalized or currency thresholds.

### Why MAPE is not primary

When the true option value is close to zero, percentage error is unstable and can dominate the average. Absolute and normalized-by-strike measures are more meaningful for this dataset.

## Price units

Most models train in normalized units:

$$
\widetilde V=\frac{V}{K}.
$$

Final reporting includes both normalized error and currency error through multiplication by strike.

Since the production strike is 100, a normalized error of 0.0001 corresponds to 0.01 currency units.

## Segmented error analysis

Aggregate metrics are segmented by:

- moneyness;
- maturity;
- volatility;
- risk-free rate;
- dividend yield;
- exercise versus continuation region;
- distance from the exercise boundary;
- dataset component.

The segmentation answers where a model fails, not only whether its average is small.

## Financial consistency

File: `src/evaluation/financial_checks.py`

### Hard architectural guarantees

The constrained residual outputs guarantee:

$$
\widehat V_A\geq 0,
$$

$$
\widehat V_A\geq I,
$$

$$
\widehat V_A\geq V_E.
$$

These are consequences of reconstruction, not empirical luck.

### Empirical lower-bound checks

The direct and diagnostic heads are tested for:

- negative price;
- price below intrinsic;
- price below European value;
- price below the combined financial floor.

Violation rates and maximum violation magnitudes are reported.

### Monotonicity

Numerical grids test expected directional behavior, including:

- put value decreases as spot increases;
- put value increases as strike increases;
- put value generally increases with volatility.

These are empirical checks. The current architecture does not guarantee every monotonicity relation.

### Integrated consistency

The four-head model is evaluated for:

- direct-price versus constrained-price gap;
- exercise-head versus continuation-implied decision gap;
- contradictory decisions;
- continuation value inconsistent with the selected action.

## Classification metrics

File: `src/evaluation/classification_metrics.py`

Reported metrics include:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- confusion matrix;
- probability calibration.

For exercise as the positive class:

$$
Precision=
\frac{TP}{TP+FP},
$$

$$
Recall=
\frac{TP}{TP+FN},
$$

$$
F1=
\frac{
2\cdot Precision\cdot Recall
}{
Precision+Recall
}.
$$

Accuracy alone is insufficient because the exercise class is imbalanced.

## Threshold selection

Classification heads output probabilities. The operating threshold is selected on validation data, typically by maximum F1.

The test threshold is frozen before test evaluation.

Notebook 08 maintains separate thresholds for:

- direct exercise head;
- continuation-implied decision path.

## Exercise-boundary evaluation

Files:

- `src/evaluation/exercise_boundary.py`
- `src/evaluation/exercise_boundary_support.py`

### Boundary bands

Observations are grouped by absolute normalized boundary distance:

$$
|D|=
\left|
\frac{I-C}{K}
\right|.
$$

Narrow bands focus on states where the exercise decision is most difficult.

### Boundary location

For controlled state grids, the estimated boundary is compared with the CRR label boundary.

### Boundary price error

Pricing MAE is reported inside boundary bands. A model can have excellent global MAE while failing near the stopping transition.

### Economic error types

A false exercise and a missed exercise opportunity are not interpreted as identical business events. Final tables retain both error directions where available.

## Out-of-domain evaluation

The final evaluation reports each OOD regime separately and also computes aggregate summaries.

For model $m$:

$$
R_m^{OOD} =
\frac{
MAE_m^{OOD}
}{
MAE_m^{ID}
}.
$$

A ratio above 1 indicates deterioration relative to the common in-domain test.

H6 uses predefined eligibility and materiality rules rather than selecting only the most dramatic regime.

### Interpretation

OOD tests are stress evidence. They do not imply that the broad in-domain production range is narrow. They establish that the surrogate should have explicit domain checks and a numerical fallback.

## Static model comparison

Files:

- `src/evaluation/final_static_comparison.py`
- `src/evaluation/model_comparison.py`

The comparison includes:

- Black–Scholes proxy;
- direct MLP;
- selected constrained residual model;
- Notebook 06 price output;
- Notebook 08 warm-start constrained price;
- Notebook 08 scratch constrained price;
- diagnostic heads where relevant.

Pairwise observation-level wins can be calculated because the prediction rows are aligned.

## Longstaff–Schwartz evaluation

File: `src/evaluation/lsm_comparison.py`

The path-based branch reports:

- held-out contract MAE and RMSE versus CRR;
- paired errors;
- bootstrap comparisons;
- standard errors;
- confidence-interval coverage;
- exercise-policy agreement;
- stopping-time distributions;
- OOD contract results;
- multi-seed robustness;
- training and valuation runtime.

Classical and neural LSM share matched simulation budgets and independent valuation paths.

## Artifact and lineage audit

Files:

- `src/evaluation/artifact_registry.py`
- `src/evaluation/final_artifact_adapters.py`
- `src/evaluation/final_lineage_audit.py`

The audit establishes:

- required files exist;
- result packages declare valid profiles;
- manifests and checkpoints agree;
- dependency fingerprints are visible;
- static predictions share IDs and targets;
- common state fields agree within absolute and relative tolerances.

These checks establish comparability, not model superiority.

## Hypothesis decisions

Notebook 09 applies fixed rules.

| Hypothesis | Technical question |
|---|---|
| H1 | Does the direct MLP materially improve on the European proxy? |
| H2 | Does selected residual learning improve on direct price learning? |
| H3 | Do constraints reduce lower-bound violations? |
| H4 | Does Notebook 06 joint learning improve boundary classification and pricing? |
| H5 | Is marginal neural inference substantially faster than CRR? |
| H6 | Do eligible models deteriorate materially on predefined OOD sets? |

Automated rules preserve consistency. The written conclusion also discusses economic materiality and limitations.

## Final result exports

Notebook 09 exports:

- aligned model metrics;
- financial consistency;
- classification and boundary tables;
- OOD summaries;
- LSM comparison;
- runtime scaling;
- operational crossover;
- lifecycle break-even;
- workload scenarios;
- hypothesis decisions;
- task recommendations;
- charts;
- final readiness audit;
- export manifest.

The export manifest is rebuilt after the final audit and verified for presence and integrity.

## Failure modes

Final readiness fails when:

- required artifacts are missing;
- a package is incomplete;
- predictions are not aligned;
- true targets disagree;
- mandatory metric tables are empty;
- hypothesis decisions are missing;
- required charts are outside the final package;
- the export manifest does not match written files.

## Related notebooks

- Notebooks 04–08 produce local evaluation packages
- Notebook 09 performs the only final cross-model comparison
