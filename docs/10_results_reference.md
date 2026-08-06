# Final Results Reference

## Purpose

This page records the frozen headline results used by Notebook 09. It is a technical lookup, not a replacement for the full interpretation.

The authoritative sources are the final artifact packages and the executed Notebook 09.

## Common static test

- Observations: **187,811**
- All static predictions aligned by `sample_id`
- Same true normalized American price verified within final tolerances

## Static pricing

| Method | Price MAE | Relative interpretation |
|---|---:|---|
| European Black–Scholes proxy | 1.339839 | Omits early-exercise value |
| Direct MLP | 0.078167 | Major improvement over proxy |
| Constrained floor residual MLP | 0.010187 | Best static price |
| Integrated warm-start constrained price | 0.029391 | Combined deployment compromise |

Derived comparisons:

- Direct/proxy MAE ratio: **0.058341**
- Selected residual/direct MAE ratio: **0.130322**
- Integrated/specialist pricing-error ratio: approximately **2.89**

## Financial consistency

| Method | Combined financial-floor violation rate |
|---|---:|
| Direct MLP | 30.812892% |
| Constrained floor residual MLP | 0% |
| Integrated constrained price | 0% |

The zero rate for constrained outputs is primarily an architectural property of reconstruction.

## Exercise classification

| Decision model | F1 |
|---|---:|
| Integrated warm-start exercise head | 0.996603 |
| Exercise-only specialist | 0.996393 |
| Notebook 06 multi-task exercise head | 0.995242 |
| Integrated continuation-implied decision | 0.985347 |

The integrated head exceeds the specialist by approximately **0.000209**. The difference is operationally negligible. The specialist remains appropriate for exercise-only deployment; the integrated head is useful when price and decision are required together.

## Longstaff–Schwartz

| Method | Held-out pricing MAE | 95% interval coverage |
|---|---:|---:|
| Classical LSM | 0.039594 | 68% |
| Neural LSM | 0.087197 | 42% |

The neural policy has approximately 2.2 times the held-out pricing MAE and weaker interval coverage.

## Out-of-domain deterioration

Formal H6 evidence:

- eligible models: 7;
- aggregate OOD/in-domain ratio at least 1.25: 7 of 7;
- aggregate ratio above 1.0: 7 of 7;
- minimum aggregate ratio: **8.127309**.

The constrained residual model retains the lowest absolute OOD error among the principal static candidates, but still deteriorates materially relative to its very low in-domain error.

## Runtime scaling

At one million valuations:

| Method | Seconds | Approximate speedup vs project CRR |
|---|---:|---:|
| Project high-resolution Numba CRR | 17.877584 | 1.00× |
| Notebook 05 constrained residual | 2.399223 | 7.45× |
| Notebook 08 integrated | 2.909232 | 6.15× |

Conservative measured warm crossover versus project CRR:

- Notebook 05: 1,000 valuations;
- Notebook 08: 1,000 valuations.

Fitted warm-curve crossover:

- Notebook 05: approximately 1 valuation;
- Notebook 08: approximately 246 valuations.

## One-billion-valuation annual workload

Assumption:

$$
4{,}000{,}000
\text{ valuations/day}
\times
250
\text{ days}
=
1{,}000{,}000{,}000.
$$

Approximate annual computation saved versus project Numba CRR:

| Deployment | Hours saved |
|---|---:|
| Notebook 05 price-only | 4.3 |
| Notebook 08 combined | 4.2 |

The more important operational effect is latency: a four-million-valuation batch falls from roughly 71 seconds to approximately 10–12 seconds.

## Lifecycle break-even

Against project Numba CRR:

| Deployment | Lower label-generation scenario | Higher scenario |
|---|---:|---:|
| Notebook 05 price-only | 272,201,560 | 5,745,812,652 |
| Notebook 08 combined | 428,521,964 | 6,079,061,481 |

Units are cumulative valuations.

## Hypothesis decisions

| Hypothesis | Decision | Primary statistic |
|---|---|---|
| H1 | Supported | Direct/proxy MAE ratio = 0.058341 |
| H2 | Supported | Residual/direct MAE ratio = 0.130322 |
| H3 | Supported | Direct violation = 0.30812892; constrained = 0 |
| H4 | Not supported | Multi-task F1 change = -0.001152; boundary-price improvement = -51.6000% |
| H5 | Supported | Formal selected neural/CRR marginal-runtime ratio = $3.7567719\times10^{-6}$ |
| H6 | Supported | 7/7 eligible models deteriorate materially; minimum ratio = 8.127309 |

The formal H5 statistic comes from the consolidated hypothesis artifact. The dedicated business-case benchmark gives the more conservative operational speedups and crossover figures above.

## Task-specific recommendations

| Task | Preferred method |
|---|---|
| Most accurate static price | Constrained floor residual MLP |
| Exercise-only deployment | Exercise-only specialist |
| Highest measured exercise F1 | Integrated warm-start exercise head |
| One model for price and exercise | Notebook 08 warm-start integrated model |
| Path-based valuation | Classical Longstaff–Schwartz |
| One-off, changing, or extreme OOD pricing | Numerical CRR, finite difference, or QuantLib fallback |

## Artifact sources

| Evidence | Artifact family |
|---|---|
| Direct model | `artifacts/direct_mlp/final_metrics.json` |
| Residual models | `artifacts/premium_models/final_metrics.json` |
| Exercise and multi-task | `artifacts/multitask_model/final_metrics.json` |
| LSM | `artifacts/neural_lsm/final_metrics.json` |
| Integrated model | `artifacts/final_multihead/final_metrics.json` |
| Common-test predictions | notebook-specific `test_predictions.parquet` |
| Final consolidated outputs | `artifacts/final_evaluation/final/` |
