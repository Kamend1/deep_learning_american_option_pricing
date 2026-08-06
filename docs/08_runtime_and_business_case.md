# Runtime and Business Case

## Purpose

A neural surrogate is not valuable merely because one forward pass is fast. The business case must separate:

1. cold setup cost;
2. warm marginal inference;
3. batch scaling;
4. numerical alternatives;
5. label generation and training;
6. lifecycle break-even;
7. realistic workload frequency.

## Benchmark implementation

Files:

- `src/evaluation/business_case_benchmark.py`
- `src/evaluation/final_business_case.py`

Configuration: `RuntimeScalingConfig`

Default batch grid:

```text
1
10
100
1,000
10,000
100,000
1,000,000
```

The benchmark uses deterministic in-domain contracts generated from the production ranges.

## Methods

### Project Numba CRR

- 250 steps;
- production batch pricer;
- price and root exercise decision;
- exact measurement up to a configured limit;
- larger rows explicitly marked as extrapolated when required.

### Notebook 05 constrained residual model

- canonical selected checkpoint;
- shared feature scaler;
- protected normalized price;
- price-only output.

### Notebook 08 integrated model

- preferred warm-start deployment checkpoint;
- integrated scaler;
- protected price;
- exercise probability.

### QuantLib methods

Optional benchmarks:

- binomial CRR;
- finite-difference American put.

QuantLib rows are labeled optional and use smaller exact limits because the per-contract Python-facing calls are expensive.

## Cold and warm timing

### Cold timing

Includes method setup such as:

- model construction;
- checkpoint loading;
- scaler loading;
- QuantLib object initialization.

Cold timing matters for one-off command execution and serverless-style workloads.

### Warm timing

Reuses the loaded method state and measures repeated calculation.

Warm timing is the relevant basis for a long-running pricing or risk service.

CUDA measurements synchronize before and after timing to prevent asynchronous execution from understating runtime.

## Measurement protocol

For each method and batch:

1. build a deterministic input frame;
2. perform configured warm-up runs;
3. execute repeated timings;
4. record median runtime;
5. calculate seconds per observation;
6. calculate observations per second;
7. label exact versus extrapolated measurements.

Median timing reduces sensitivity to isolated operating-system interruptions.

## Runtime scaling

At one million valuations on the measured machine:

| Method | Approximate seconds | Approximate speedup vs project CRR |
|---|---:|---:|
| Project Numba CRR | 17.88 | 1.0× |
| Notebook 05 constrained residual | 2.40 | 7.45× |
| Notebook 08 integrated | 2.91 | 6.15× |

The integrated model is slower than the specialist because it computes several heads, but it returns both price and exercise information.

## Crossover definitions

### Fitted curve crossover

A scaling curve estimates where neural runtime becomes lower than numerical runtime.

Warm fitted crossover against project CRR:

- Notebook 05: approximately 1 valuation;
- Notebook 08: approximately 246 valuations.

### Smallest measured neural win

The actual benchmark grid is more conservative. Both neural models first beat project CRR at a measured warm batch of approximately 1,000 valuations.

### Operational materiality

A method may be technically faster while saving only milliseconds. Materiality depends on batch size and repetition.

The project therefore treats millions of repeated valuations as the relevant operational scale.

## Example annual workload

At 4 million valuations per operating day over 250 days:

$$
4{,}000{,}000\times250
=
1{,}000{,}000{,}000
$$

annual valuations.

Relative to the optimized project CRR, this corresponds approximately to:

| Model | Annual computation saved |
|---|---:|
| Notebook 05 specialist | 4.3 hours |
| Notebook 08 integrated | 4.2 hours |

The annual hour total is not the only value. A four-million-valuation job falls from roughly 71 seconds to approximately 10–12 seconds, reducing decision latency.

Relevant environments include:

- options market making;
- high-frequency and algorithmic trading;
- large brokerage pricing;
- institutional portfolio revaluation;
- real-time risk limits;
- scenario and stress grids;
- repeated Greeks or sensitivity workflows built on repricing.

## Up-front cost

The surrogate pays an initial cost:

- numerical label generation;
- feature preprocessing;
- model training;
- candidate rejection;
- validation;
- deployment preparation.

The production label-generation time was not available as one authoritative historical artifact, so the final business case uses explicit scenarios from 0.5 to 24 hours.

Minimum reproducible deployment and the full research programme are treated separately.

## Lifecycle break-even

Against project Numba CRR, cumulative break-even ranges are:

| Deployment | Lower scenario | Higher scenario |
|---|---:|---:|
| Notebook 05 price-only | 272,201,560 | 5,745,812,652 valuations |
| Notebook 08 combined | 428,521,964 | 6,079,061,481 valuations |

At approximately one billion annual valuations, the lower-cost Notebook 05 case recovers within the first year.

Against standard QuantLib engines, break-even is materially lower because each avoided numerical valuation is slower.

## Workload scenarios

Examples evaluated in Notebook 09 include:

- one-off 100-contract portfolio;
- daily 10,000-contract portfolio;
- 10,000 contracts under 100 scenarios;
- 10,000 contracts under 1,000 scenarios;
- intraday ten-million-valuation grids.

The conclusion is:

- isolated and small workloads do not justify the surrogate;
- one-million-valuation repeated jobs begin to produce visible savings;
- ten-million-valuation grids produce clearly material savings;
- very high-frequency repeated evaluation provides the strongest operational case.

## Accuracy-runtime trade-off

The runtime benefit is purchased with approximation error and domain restrictions.

The selected Notebook 05 model has the best static pricing accuracy. Notebook 08 accepts higher price error and lower throughput in exchange for an exercise output from the same model.

Numerical methods remain preferable when:

- the payoff changes;
- the stochastic model changes;
- discrete dividends are introduced;
- inputs leave the trained range;
- independent validation is required;
- workload volume is low.

## Deployment control

A production surrogate should apply:

1. schema validation;
2. feature-domain validation;
3. scaler and checkpoint fingerprint validation;
4. protected price reconstruction;
5. numerical fallback;
6. periodic benchmark and drift checks.

Notebook 08 exports domain bounds and a deployment policy to support these controls.

## Limitations

Runtime evidence is specific to:

- hardware;
- Python and library versions;
- CPU/GPU device;
- batch size;
- process state;
- data-transfer pattern;
- implementation choices.

The benchmark does not claim a universal speedup. It documents the measured environment and separates exact from extrapolated rows.

## Related notebook

- Notebook 09, business-case sections
