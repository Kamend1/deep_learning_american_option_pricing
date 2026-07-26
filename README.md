<div align="center">

# Deep Learning for American Option Pricing

### Learning the Early-Exercise Premium and Exercise Boundary

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Research%20Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)](#project-status)

**SoftUni Deep Learning Final Project**

</div>

---

## Project overview

This project investigates the use of deep neural networks as surrogate pricing models for **American put options**.

Unlike European options, American options may be exercised before maturity. Their valuation therefore includes an **optimal-stopping problem**: at every admissible exercise date, the holder must compare the immediate exercise payoff with the expected discounted continuation value.

The project does not attempt to replace a closed-form solution with an unnecessarily complex neural network. Instead, it focuses on a problem for which no general closed-form solution exists and where repeated numerical valuation can become computationally expensive.

The core research question is:

> Can a financially structured neural network learn American put prices, the early-exercise premium, and the exercise boundary with sufficient accuracy, financial consistency, and computational efficiency to function as a practical surrogate for traditional numerical methods?

The repository is designed as an **academic computational-finance paper implemented through reproducible notebooks**. Markdown discussion, mathematical reasoning, citations, interpretation, and limitations are treated as first-class project components. Reusable implementation remains in `src/`, while notebooks present and evaluate the research.

---

## Research objectives

The project evaluates whether deep learning can:

- approximate high-resolution American put prices;
- learn the early-exercise premium over the corresponding European option;
- identify the exercise-versus-continuation region;
- respect essential financial lower bounds and monotonicity properties;
- generalize across different moneyness, maturity, volatility, rate, and dividend regimes;
- provide faster batched inference than repeated numerical pricing;
- remain robust when tested outside the parameter domain used for training.

The project explicitly distinguishes between:

- **surrogate numerical pricing**, which is the scope of this work; and
- **market-price forecasting**, which is not the scope of this work.

The neural networks learn a pricing function generated under a clearly defined numerical model. They do not predict future underlying prices or claim to discover an objectively correct market price.

---

## Financial foundation

### European put benchmark

Under the Black–Scholes–Merton framework with continuous dividend yield, the European put value is

\[
P_E = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1),
\]

where

\[
d_1 =
\frac{
\ln(S/K) + \left(r-q+\frac{1}{2}\sigma^2\right)T
}{
\sigma\sqrt{T}
},
\qquad
d_2=d_1-\sigma\sqrt{T}.
\]

The Black–Scholes price is used as:

1. an analytical European benchmark;
2. a convergence target for the European CRR tree;
3. a lower bound for the corresponding American option;
4. the known component of the early-exercise-premium model.

### American optimal stopping

At time \(t\), the American put holder compares immediate exercise value

\[
I(S_t)=\max(K-S_t,0)
\]

with continuation value

\[
C(S_t,t)=
\mathbb{E}^{\mathbb{Q}}
\left[
e^{-r\Delta t}
V_A(S_{t+\Delta t},t+\Delta t)
\mid S_t
\right].
\]

The American value is

\[
V_A(S_t,t)=\max\left(I(S_t),C(S_t,t)\right).
\]

### Early-exercise premium

The American option can be decomposed as

\[
V_A = V_E + EEP,
\]

where

\[
EEP = V_A - V_E \geq 0.
\]

The main proposed neural model predicts only this residual component:

\[
\widehat{V}_A
=
V_{BS}
+
\operatorname{Softplus}(g_\theta(x)).
\]

This embeds the non-negative early-exercise-premium condition directly in the model architecture.

---

## Research questions

1. Can a conventional multilayer perceptron accurately approximate high-resolution American put prices?
2. Does predicting the early-exercise premium improve performance relative to predicting the complete price?
3. Do financially motivated output constraints reduce economically impossible predictions?
4. Can a multi-task model jointly learn price and the exercise-versus-continuation decision?
5. How do errors vary across moneyness, maturity, volatility, interest-rate, and dividend-yield regimes?
6. How materially does performance deteriorate outside the training domain?
7. Does neural inference provide a meaningful speed advantage over numerical valuation?

---

## Predefined hypotheses

The hypotheses are specified before neural-network training to reduce retrospective interpretation.

- **H1 — Direct pricing approximation:** A direct MLP will outperform the European Black–Scholes value used as an American-option proxy.
- **H2 — Premium decomposition:** A model trained on the early-exercise premium will outperform a model trained on the full American price.
- **H3 — Financial constraints:** A constrained premium model will produce fewer lower-bound violations than an unconstrained direct-price model.
- **H4 — Multi-task learning:** A joint price-and-exercise model will estimate the exercise boundary more accurately than a price-only model.
- **H5 — Computational acceleration:** Batched neural inference will be substantially faster than repeated high-resolution CRR valuation.
- **H6 — Out-of-domain deterioration:** All neural models will perform materially worse outside the training domain.

---

## Methodology

### 1. Numerical pricing foundation

The project uses:

- **Black–Scholes–Merton** for European analytical prices;
- **Cox–Ross–Rubinstein binomial trees** for European and American option pricing;
- **QuantLib** as an optional independent validation engine;
- **Least-Squares Monte Carlo** as a later extension.

The production tree resolution is selected through a documented convergence and runtime study rather than fixed arbitrarily.

### 2. Synthetic dataset generation

The configured production design contains exactly **1,450,000 observations**:

| Dataset component | Observations |
|---|---:|
| Core domain | 1,000,000 |
| Boundary-focused sample | 250,000 |
| OOD high volatility | 50,000 |
| OOD extreme moneyness | 50,000 |
| OOD long maturity | 50,000 |
| OOD rate/dividend combinations | 50,000 |
| **Total** | **1,450,000** |

The generation pipeline uses randomized Latin hypercube sampling and a Numba-accelerated CRR implementation.

The core input vector is

\[
x =
\left[
\log(S/K),\;
T,\;
r,\;
q,\;
\sigma
\right].
\]

The primary normalized target is

\[
y=\frac{V_A}{K}.
\]

Generated records include:

- spot and strike;
- moneyness and log-moneyness;
- time to maturity;
- risk-free rate;
- dividend yield;
- volatility;
- intrinsic value;
- continuation value;
- European price;
- raw and validated American prices;
- early-exercise premium;
- normalized targets;
- exercise decision;
- CRR step count;
- any pricing-floor adjustment.

### 3. Chunked and restartable generation

The production dataset is not written as one monolithic file. It is generated in deterministic Parquet chunks.

This design provides:

- bounded memory usage;
- restartability after interruption;
- component-level validation;
- transparent progress tracking;
- scalable downstream loading.

The generation script records a production manifest containing:

- component sizes;
- random seeds;
- parameter ranges;
- chunk sizes;
- CRR step count;
- output paths;
- validation results;
- generation timestamps.

### 4. Dataset design and split policy

The final dataset contains three experimental layers:

- **Core domain** — standard training, validation, and in-domain test observations;
- **Boundary-focused sample** — additional observations near the exercise/continuation transition;
- **Out-of-domain sets** — parameter regimes excluded from training and reserved for robustness testing.

The in-domain split is approximately:

- 70% training;
- 15% validation;
- 15% test.

All feature preprocessing is fitted only on the training observations.

### 5. Direct MLP baseline

The first neural model is a direct normalized-price regressor.

Input features:

```python
FEATURE_COLUMNS = [
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
]
```

Target:

```python
TARGET_COLUMN = "normalized_american_price"
```

The baseline architecture is:

```text
Input(5)
→ Linear(128)
→ BatchNorm
→ SiLU
→ Linear(128)
→ BatchNorm
→ SiLU
→ Linear(64)
→ SiLU
→ Linear(32)
→ SiLU
→ Linear(1)
→ Softplus
```

The output activation enforces non-negative normalized prices. The direct MLP is the control model that later premium and constrained architectures must outperform.

### 6. Comparative model framework

The project compares:

1. European Black–Scholes proxy;
2. direct MLP price model;
3. early-exercise-premium MLP;
4. financially constrained premium MLP;
5. multi-task price and exercise model;
6. classical Least-Squares Monte Carlo;
7. neural Least-Squares Monte Carlo;
8. final integrated static multi-head model.

### 7. Final integrated static model

The final static architecture uses one shared option-state representation and four specialized heads:

```text
Input(5)
→ Shared backbone
   ├── Financial-floor residual head
   ├── Direct American-price head
   ├── Continuation-value head
   └── Exercise-classification head
```

The authoritative price is reconstructed as

\[
\widehat V_A
=
\max(V_E,I)
+
\operatorname{Softplus}(\widehat R_F).
\]

The continuation and exercise heads provide two independent paths to the stopping decision. Their disagreement is measured and penalized during training. The integrated static model remains separate from neural LSM because neural LSM consumes simulated paths and time-indexed states rather than one static contract vector.

---

## Evaluation framework

Aggregate loss alone is not sufficient for this project.

### Pricing metrics

- mean absolute error;
- root mean squared error;
- normalized MAE;
- median absolute error;
- maximum absolute error;
- coefficient of determination;
- percentage of observations within predefined error bands.

MAPE is not used as the primary measure because percentage errors become unstable when the true option value is close to zero.

### Financial-consistency tests

Predicted American put values should satisfy:

\[
\widehat{V}_A \geq 0,
\]

\[
\widehat{V}_A \geq \max(K-S,0),
\]

\[
\widehat{V}_A \geq V_E.
\]

The project also tests expected directional properties:

- put value decreases as spot increases;
- put value increases as strike increases;
- put value generally increases as volatility increases;
- American value does not fall below the equivalent European value.

### Exercise-boundary evaluation

The exercise component is assessed through:

- precision;
- recall;
- F1-score;
- confusion matrix;
- boundary-location error;
- performance near the exercise/continuation transition.

### Computational benchmarking

The project compares:

- numerical-pricing runtime;
- neural inference runtime;
- batched portfolio valuation speed;
- training and data-generation cost as separate up-front investments.

---

## Repository architecture

```text
deep_learning_american_option_pricing/
├── notebooks/
│   ├── 01_option_pricing_foundations.ipynb
│   ├── 02_american_option_data_generation.ipynb
│   ├── 03_dataset_analysis_and_validation.ipynb
│   ├── 04_direct_mlp_pricer.ipynb
│   ├── 05_early_exercise_premium_model.ipynb
│   ├── 06_exercise_boundary_analysis.ipynb
│   ├── 07_neural_longstaff_schwartz.ipynb
│   ├── 08_final_multihead_model.ipynb
│   └── 09_final_evaluation.ipynb
│
├── docs/
│   ├── 01_option_pricing_foundations.md
│   ├── 02_american_option_data_generation.md
│   ├── 03_dataset_analysis_and_validation.md
│   ├── 04_direct_mlp_pricer.md
│   ├── 05_early_exercise_premium_model.md
│   ├── 06_exercise_boundary_analysis.md
│   ├── 07_neural_longstaff_schwartz.md
│   ├── 08_final_multihead_model.md
│   ├── 09_final_evaluation.md
│   ├── RESULTS_REQUIRED.md
│   └── FINAL_WRITEUP_CHECKLIST.md
│
├── scripts/
│   ├── generate_production_dataset.py
│   ├── train_final_multihead.py
│   ├── validate_production_project.py
│   └── build_final_results.py
│
├── src/
│   ├── pricing/
│   │   ├── black_scholes.py
│   │   ├── binomial_tree.py
│   │   ├── validation.py
│   │   ├── simulation.py
│   │   └── longstaff_schwartz.py
│   │
│   ├── data/
│   │   ├── generation.py
│   │   ├── production_generation.py
│   │   ├── dataset_validation.py
│   │   ├── splitting.py
│   │   ├── multihead_targets.py
│   │   └── torch_datasets.py
│   │
│   ├── models/
│   │   ├── direct_pricer.py
│   │   ├── premium_pricer.py
│   │   ├── multitask_pricer.py
│   │   ├── neural_longstaff_schwartz.py
│   │   └── integrated_multihead_pricer.py
│   │
│   ├── training/
│   │   ├── loops.py
│   │   ├── checkpointing.py
│   │   ├── losses.py
│   │   ├── multitask_losses.py
│   │   ├── multitask_loops.py
│   │   ├── lsm_training.py
│   │   ├── multihead_losses.py
│   │   └── multihead_loops.py
│   │
│   └── evaluation/
│       ├── regression_metrics.py
│       ├── financial_checks.py
│       ├── model_comparison.py
│       ├── classification_metrics.py
│       ├── exercise_boundary.py
│       ├── lsm_comparison.py
│       ├── internal_consistency.py
│       ├── integrated_model_comparison.py
│       ├── artifact_registry.py
│       ├── final_project_evaluation.py
│       ├── hypothesis_testing.py
│       └── final_reporting.py
│
├── tests/
│   ├── integration/
│   │   ├── test_static_multihead_pipeline.py
│   │   └── test_full_project_pipeline.py
│   ├── test_black_scholes.py
│   ├── test_binomial_tree.py
│   ├── test_data_generation.py
│   ├── test_data_splitting.py
│   ├── test_torch_datasets.py
│   ├── test_direct_pricer.py
│   ├── test_training_pipeline.py
│   ├── test_premium_pricer.py
│   ├── test_multitask_pricer.py
│   ├── test_gbm_simulation.py
│   ├── test_longstaff_schwartz.py
│   ├── test_neural_longstaff_schwartz.py
│   ├── test_lsm_comparison.py
│   ├── test_integrated_multihead_pricer.py
│   ├── test_multihead_losses.py
│   ├── test_internal_consistency.py
│   ├── test_integrated_model_comparison.py
│   └── test_final_project_evaluation.py
│
├── data/
│   ├── generated/          # ignored by Git
│   ├── manifests/          # tracked metadata and split definitions
│   └── sample/             # optional small reproducible examples
│
├── artifacts/              # ignored model checkpoints and training outputs
├── references/
│   ├── references.bib
│   ├── literature_matrix.md
│   └── citation_audit.md
│
├── pytest.ini
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Notebook 09 and the final artifact-audit, aggregation, and write-up skeleton are implemented. Empirical tables remain pending until the production runs are completed.

---

## Notebook workflow

The notebooks form one sequential academic workflow.

### Notebook 01 — Theoretical foundations and numerical motivation

- connects the project to earlier Black–Scholes work;
- introduces optimal stopping;
- visualizes intrinsic, European, and American values;
- defines the early-exercise premium;
- establishes the literature context;
- states the research questions and hypotheses.

### Notebook 02 — American option data generation

- validates pricing engines;
- studies CRR convergence;
- benchmarks runtime versus accuracy;
- selects production tree resolution;
- generates and validates the pilot dataset;
- records pricing-floor adjustments transparently.

### Notebook 03 — Dataset analysis and validation

- audits schema and data quality;
- analyzes parameter and target distributions;
- measures exercise-region representation;
- evaluates the Black–Scholes proxy baseline;
- freezes deterministic train, validation, and test splits;
- defines out-of-domain regimes.

### Notebook 04 — Direct MLP pricer

- loads the frozen production split;
- verifies features, targets, and dataloaders;
- trains the direct MLP benchmark;
- compares MSE and Smooth L1 loss;
- saves the best checkpoint and feature scaler;
- reports in-domain metrics;
- evaluates segmented errors;
- measures financial-consistency violations;
- evaluates out-of-domain deterioration;
- benchmarks neural inference against CRR pricing.

### Notebook 05 — Early-exercise-premium models

- compares direct-price and residual target formulations;
- trains unconstrained, non-negative-premium, and financial-floor residual models;
- evaluates weighted and boundary-aware losses;
- tests financial lower-bound violations;
- evaluates H2 and H3.

### Notebook 06 — Exercise-boundary multi-task model

- validates exercise labels and class balance;
- trains an exercise-only classifier;
- trains a shared price-and-exercise network;
- reconstructs exercise boundaries;
- evaluates boundary-region pricing and H4.

### Notebook 07 — Classical and neural Longstaff–Schwartz

- validates risk-neutral GBM simulation;
- compares polynomial and Laguerre classical LSM policies;
- studies path-count and exercise-date convergence;
- trains amortized neural continuation networks;
- evaluates policies on independent held-out paths and contracts;
- reports OOD, multi-seed, stopping-policy, and runtime results.

### Notebook 08 — Final integrated static multi-head model

- trains one shared backbone with residual, direct-price, continuation, and exercise heads;
- compares balanced, pricing-focused, and decision-focused loss configurations;
- evaluates scratch and optional Step 6 warm-start training;
- quantifies internal price and exercise consistency;
- reconstructs the exercise boundary;
- saves the authoritative constrained-price checkpoint.

### Notebook 09 — Final evaluation and project synthesis

- audits all required production artifacts;
- aligns static and simulation-based model results;
- consolidates pricing, boundary, consistency, OOD, and runtime tables;
- applies predefined H1–H6 decision rules;
- provides placeholders for literature synthesis and final conclusions;
- exports the final write-up inputs and project-level result manifests;
- runs safely in placeholder mode before the expensive production experiments.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kamend1/deep_learning_american_option_pricing.git
cd deep_learning_american_option_pricing
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional independent validation:

```bash
pip install QuantLib
```

### PyTorch and CUDA

CPU execution is fully supported.

For GPU training, install the appropriate CUDA-enabled PyTorch build for the local environment before installing the remaining project dependencies.

Numba accelerates CRR label generation, while PyArrow provides chunked Parquet storage.

---

## Running the tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

Run only the pricing tests:

```bash
python -m pytest -q \
    tests/test_black_scholes.py \
    tests/test_binomial_tree.py
```

Run only the dataset tests:

```bash
python -m pytest -q \
    tests/test_data_generation.py \
    tests/test_data_splitting.py
```

Run the direct-model pipeline tests:

```bash
python -m pytest -q \
    tests/test_torch_datasets.py \
    tests/test_direct_pricer.py \
    tests/test_training_pipeline.py
```

Run the simulation and Longstaff–Schwartz tests:

```bash
python -m pytest -q \
    tests/test_gbm_simulation.py \
    tests/test_longstaff_schwartz.py \
    tests/test_neural_longstaff_schwartz.py \
    tests/test_lsm_comparison.py
```

Run the final integrated-model tests:

```bash
python -m pytest -q \
    tests/test_integrated_multihead_pricer.py \
    tests/test_multihead_losses.py \
    tests/test_internal_consistency.py \
    tests/test_integrated_model_comparison.py
```

Run the miniature end-to-end integration pipeline:

```bash
python -m pytest -q -m integration
```

Run tests explicitly marked as slow:

```bash
python -m pytest -q -m slow
```

The default `pytest` configuration excludes `integration` and `slow` tests so routine validation remains fast. Full production generation and full model training are not ordinary unit-test workloads.

The pricing tests should pass before generating large datasets. A neural network trained on an incorrect pricing engine will learn the implementation error rather than correct it.

---

## Production dataset generation

Generate the full 1.45 million observation design from the repository root:

```bash
python scripts/generate_production_dataset.py
```

The script:

- generates each component separately;
- writes deterministic Parquet chunks;
- skips completed chunks when restarted;
- validates component outputs;
- records the production manifest;
- avoids loading the entire dataset into memory.

The generated files are intentionally excluded from Git.

Train one final multi-head configuration non-interactively:

```bash
python scripts/train_final_multihead.py --config balanced
```

Available presets are `balanced`, `pricing_focused`, and `decision_focused`. A Step 6-compatible warm-start run can be launched with:

```bash
python scripts/train_final_multihead.py \
    --config decision_focused \
    --architecture step6_compatible \
    --warm-start-checkpoint artifacts/multitask_model/best_multitask_pricer.pt
```

---


## Final production validation and report build

Before the final academic write-up, validate the complete artifact chain:

```bash
python scripts/validate_production_project.py --deep
```

Build the consolidated Notebook 09 input tables:

```bash
python scripts/build_final_results.py --strict
```

During the architecture-only phase, both scripts support a safe pending state:

```bash
python scripts/validate_production_project.py --allow-missing
python scripts/build_final_results.py
```

Notebook 09 never fabricates unavailable results. Missing evidence is marked as
`PENDING` or produces an `Inconclusive` hypothesis decision.

---

## Execution order

Run the notebooks sequentially:

```text
01_option_pricing_foundations.ipynb
02_american_option_data_generation.ipynb
03_dataset_analysis_and_validation.ipynb
04_direct_mlp_pricer.ipynb
05_early_exercise_premium_model.ipynb
06_exercise_boundary_analysis.ipynb
07_neural_longstaff_schwartz.ipynb
08_final_multihead_model.ipynb
09_final_evaluation.ipynb
```

At the current stage, Notebooks 01–08 and their supporting implementation modules are prepared. Full-profile data generation, model training, out-of-domain evaluation, and Notebook 09 still need to be executed before the final conclusions are locked.

---

## Generated data and Git policy

Large generated datasets, checkpoints, and experiment outputs are intentionally excluded from Git.

Typical ignored paths include:

```text
data/generated/
data/processed/
models/
checkpoints/
artifacts/
runs/
```

Small metadata files remain tracked:

```text
data/manifests/
references/
docs/
```

The production dataset is stored as multiple Parquet chunks rather than one monolithic file. This keeps generation restartable, constrains memory use, and supports scalable training-data loading.

Dataset manifests record:

- generation configuration;
- random seeds;
- pricing-engine settings;
- parameter ranges;
- tree resolution;
- component sizes;
- chunk sizes;
- split counts;
- validation results.

---

## Academic writing and citation policy

The project is structured as an academic paper, not only as a software demonstration.

Each notebook includes:

- motivation and research context;
- mathematical definitions;
- literature citations;
- methodological justification;
- code and outputs;
- interpretation;
- comparison with prior work;
- limitations;
- conclusions.

The notebook remains the executable source of truth. Markdown twins under `docs/` make the research easier to read directly on GitHub.

The bibliography is maintained under:

```text
references/references.bib
references/literature_matrix.md
references/citation_audit.md
```

The literature review covers direct supervised pricing, recurrent architectures, American-option optimal stopping, PDE-based neural methods, and volatility-surface representation.

---

## Project status

### Completed implementation

- project architecture and academic research design;
- Black–Scholes call and put pricing;
- European and American CRR pricing;
- pricing diagnostics and financial validation;
- convergence and runtime framework;
- synthetic pilot-data generation;
- dataset-quality auditing;
- deterministic split design;
- out-of-domain framework;
- 1.45 million observation production-data design;
- chunked, restartable, Numba-accelerated generation pipeline;
- PyTorch Dataset and DataLoader pipeline;
- direct MLP architecture;
- reusable training and checkpointing utilities;
- regression metrics and financial-consistency checks;
- direct, premium, constrained residual, exercise-classification, and multi-task architectures;
- GBM simulation with antithetic variates and moment validation;
- classical Longstaff–Schwartz with independent policy and valuation paths;
- amortized neural continuation-value policy by exercise index;
- LSM pricing, confidence-interval, stopping-policy, OOD, and runtime comparisons;
- final integrated four-head static architecture;
- multi-objective loss presets and internal-consistency penalties;
- scratch and Step 6-compatible warm-start support;
- non-interactive final-model training script;
- tiered pytest configuration with unit and integration markers;
- notebooks and Markdown documentation for Steps 1–8;
- unit and integration tests for pricing, data, neural models, exercise boundaries, simulation, LSM, and the final static pipeline.

### Pending execution and empirical validation

- full production dataset generation;
- direct MLP training on the full in-domain dataset;
- final in-domain and out-of-domain metrics;
- financial-violation analysis from trained-model predictions;
- CPU and GPU inference-speed benchmark.

### Completed final architecture

- Notebook 09 final-evaluation skeleton and Markdown twin;
- artifact registry with safe pending-state handling;
- production validation and final-results build scripts;
- predefined H1–H6 decision framework;
- full-project integration smoke test;
- final results and academic-writeup checklists.

### Planned execution and writing

- final-profile execution of all static and neural LSM experiments;
- final integrated-model loss and warm-start ablations;
- cross-model evaluation on aligned static and LSM contract grids;
- evidence-based H1–H6 decisions in Notebook 09;
- section conclusions and notebook-to-notebook handoffs;
- consolidated academic paper and conclusions.

---

## Scope limitations

The core project deliberately excludes several realistic market features:

- transaction-level American option data;
- bid–ask microstructure;
- stochastic volatility;
- jumps;
- discrete dividends;
- transaction costs;
- multiple correlated underlyings;
- full implied-volatility surfaces;
- reinforcement learning;
- transformer architectures.

These are not treated as irrelevant. They are excluded to keep the central experiment identifiable and reproducible.

Synthetic labels inherit the assumptions and approximation errors of the pricing engine. Strong interpolation performance will not be interpreted as evidence of reliable extrapolation or real-market pricing superiority.

---

## Selected references

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach.
- Merton, R. C. (1973). Theory of rational option pricing.
- Ding, L., Lu, E., & Cheung, K. (2025). Deep learning option pricing with market implied volatility surfaces.
- Elbayed, Z., & Qadi El Idrissi, A. (2025). Deep learning in financial modeling: Predicting European put option prices with neural networks.
- Ke, A., & Yang, A. (2019). Option pricing with deep learning.
- Pimentel, R., et al. (2026). Option pricing with deep learning: A long short-term memory approach.
- Pu, V. R. H. (2021). Pricing options using deep neural networks from a practical perspective.
- Zouaoui, H., & Naas, M.-N. (2023). Option pricing using deep learning based on LSTM-GRU neural networks.

The complete and audited bibliography will be maintained in `references/references.bib`.

---

## Connection to previous work

This project extends the option-pricing and hedging work developed in:

[Mathematical Foundations of Portfolio Optimization and Hedging](https://github.com/Kamend1/math_for_developers_final_project)

The earlier project focused on Black–Scholes pricing, protective puts, Greeks, and convexity. The current project advances from closed-form European valuation to American optimal stopping and deep-learning surrogate models.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Kamen Dimitrov**

Finance professional, CFA charterholder, and SoftUni AI and Machine Learning Upskill Program participant.

Project repository:

[github.com/Kamend1/deep_learning_american_option_pricing](https://github.com/Kamend1/deep_learning_american_option_pricing)
