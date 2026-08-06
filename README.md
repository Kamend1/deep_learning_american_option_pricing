<div align="center">

# Deep Learning for American Put Option Pricing

### When does a neural surrogate add value to a numerical pricing problem?

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-9%20Research%20Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![Data](https://img.shields.io/badge/Data-DVC%20%2B%20Cloudflare%20R2-13ADC7)](#data-and-artifact-access)
[![License](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-2EA44F)](#project-status)

**SoftUni Deep Learning Final Project**

</div>

---

## Project overview

This project investigates whether deep learning has a justified role in **American put option pricing**.

The objective is not to use a neural network merely because this is a deep-learning project. American options already have established numerical pricing methods. The relevant question is whether a trained neural surrogate can add enough value through:

- lower pricing error than a simple European proxy;
- financially valid outputs;
- accurate early-exercise decisions;
- faster repeated valuation;
- and transparent control of model risk.

The central research question is:

> **Can deep learning provide an accurate, financially coherent, and computationally efficient surrogate for American put pricing, and under what operating conditions does that surrogate make practical sense?**

The project loosely reproduces and extends questions raised in the deep-learning option-pricing literature. It combines direct supervised pricing, residual learning, exercise classification, neural Longstaff–Schwartz, and an integrated multi-head model under one controlled dataset and one final evaluation framework.

The repository is structured as an **academic computational-finance study implemented through reproducible notebooks**. The notebooks present the argument and evidence, `src/` contains reusable implementation, `tests/` verifies numerical and software contracts, and `docs/` explains the technical system.

---

## Final answer

A financially structured neural network can serve as a highly accurate and computationally efficient surrogate for American put pricing **inside a validated parameter domain and at sufficiently high repeated volume**.

The strongest result does not come from the largest model. It comes from changing the learning problem. The selected Notebook 05 model predicts only the non-negative residual above the known financial floor

$$
F = \max(V_E, I),
$$

and reconstructs the American value as

$$
\widehat V_A =
F
+
K\operatorname{Softplus}(g_\theta(x)).
$$

This constrained floor-residual MLP produces the lowest static pricing error and eliminates the lower-bound violations observed in the direct MLP.

The project does **not** produce one model that is best at everything. It establishes a division of work:

| Task | Preferred method | Reason |
|---|---|---|
| Most accurate static price | **Constrained floor-residual MLP** | Lowest aligned test MAE and zero financial-floor violations |
| Exercise-only deployment | **Exercise-only classifier** | Narrow specialist model with effectively the same classification quality as the integrated head |
| Highest measured exercise F1 | **Integrated warm-start exercise head** | Marginally highest F1, but only by 0.000209 |
| One model for price and exercise | **Notebook 08 warm-start integrated model** | Returns a protected price and exercise recommendation together |
| Path-based valuation | **Classical Longstaff–Schwartz** | More accurate, faster, and better calibrated than neural LSM in this experiment |
| One-off, changing, or stress-regime pricing | **Numerical method** | No training cost, immediate adaptability, and no extrapolation dependence |

The numerical method is therefore not replaced. It remains necessary to generate labels, validate the surrogate, price changed contracts and models, and provide a fallback outside the learned domain.

---

## Headline empirical results

All static models are compared on the same **187,811-observation** test set, aligned by `sample_id` and verified to contain identical targets.

### Static pricing

| Method | Price MAE | Interpretation |
|---|---:|---|
| European Black–Scholes proxy | 1.339839 | Does not capture the American early-exercise value |
| Direct MLP | 0.078167 | Major improvement from direct function approximation |
| **Constrained floor-residual MLP** | **0.010187** | Best static pricing result |
| Integrated warm-start constrained price | 0.029391 | Combined deployment compromise |

The direct MLP reduces MAE relative to the European proxy by approximately **94.2%**. The constrained floor-residual model then reduces MAE by a further **87.0%** relative to the direct MLP.

The integrated model remains strong, but its pricing error is approximately **2.89 times** that of the Notebook 05 specialist.

### Financial consistency

| Method | Combined financial-floor violation rate |
|---|---:|
| Direct MLP | 30.812892% |
| Constrained floor-residual MLP | 0% |
| Integrated constrained price | 0% |

This is a key result. Average loss alone hides economically impossible outputs. The constrained reconstruction removes that defect by construction.

### Exercise decision

| Decision model | F1 |
|---|---:|
| Integrated warm-start exercise head | 0.996603 |
| Exercise-only specialist | 0.996393 |
| Notebook 06 multi-task exercise head | 0.995242 |
| Integrated continuation-implied decision | 0.985347 |

The integrated exercise head has the highest measured F1, but the difference from the specialist is only **0.000209**. The specialist remains the rational exercise-only deployment. The integrated head earns its role when price and exercise must be produced together.

### Classical and neural Longstaff–Schwartz

| Method | Held-out pricing MAE | 95% interval coverage |
|---|---:|---:|
| **Classical Longstaff–Schwartz** | **0.039594** | **68%** |
| Neural Longstaff–Schwartz | 0.087197 | 42% |

The neural continuation policy has approximately **2.2 times** the held-out pricing error, weaker interval coverage, and higher computational burden. Adding a neural network to a numerical method did not improve the method in this design.

### Out-of-domain robustness

All **7 of 7** eligible static neural models deteriorate materially outside their training range. The minimum aggregate OOD-to-in-domain error ratio is **8.127309**.

The validated operating domain is intentionally broad:

| Variable | In-domain range |
|---|---:|
| Moneyness $S/K$ | 0.50 to 1.50 |
| Time to maturity | 7 days to 2 years |
| Volatility | 5% to 80% |
| Risk-free rate | 0% to 10% |
| Continuous dividend yield | 0% to 8% |

The OOD experiments extend into extreme moneyness, volatility up to 120%, maturities up to four years, and exceptional rate/dividend combinations. These are best interpreted as stress regimes requiring numerical fallback rather than normal deployment conditions.

---

## Does deep learning make business sense here?

The answer is conditional.

For an isolated option or a small portfolio, numerical pricing remains the sensible choice. It is already available, adapts immediately when assumptions change, and does not require synthetic data or training.

The neural surrogate becomes useful when the **same in-domain pricing map is evaluated repeatedly at large scale**.

### Measured runtime at one million valuations

| Method | Runtime | Speedup vs project CRR |
|---|---:|---:|
| Project high-resolution Numba CRR | 17.877584 seconds | 1.00× |
| Notebook 05 constrained residual | 2.399223 seconds | 7.45× |
| Notebook 08 integrated | 2.909232 seconds | 6.15× |

The conservative measured warm crossover is approximately **1,000 valuations** for both neural deployments. That is only the point where neural inference becomes faster; the absolute saving is still small.

The practical difference becomes visible in repeated million-valuation jobs and material in repeated ten-million-valuation grids.

### Example: one billion valuations annually

Assume four million calculations per operating day:

$$4{,}000{,}000 \times 250 = 1{,}000{,}000{,}000$$

annual valuations.

Relative to the already optimized project Numba CRR implementation:

| Deployment | Approximate annual computation saved |
|---|---:|
| Notebook 05 price-only model | 4.3 hours |
| Notebook 08 price-and-exercise model | 4.2 hours |

More importantly, one four-million-valuation batch falls from roughly **71 seconds** to approximately **10–12 seconds**.

The credible operating environments are therefore:

- options market making;
- algorithmic and high-frequency trading workflows;
- large brokerage pricing systems;
- institutional portfolio revaluation;
- real-time risk limits;
- large scenario and stress grids;
- repeated sensitivity and Greeks calculations based on repricing.

The benchmark measures **batch throughput**, not exchange-level microsecond latency. Its value in low-latency environments is the ability to refresh large pricing and risk surfaces several times faster, not a claim that the Python implementation itself is a complete high-frequency trading system.

### Lifecycle break-even

Training is not free. The surrogate must recover the cost of numerical label generation, model training, validation, and deployment.

Against the optimized project Numba CRR, the estimated cumulative break-even ranges are:

| Deployment | Lower-cost scenario | Higher-cost scenario |
|---|---:|---:|
| Notebook 05 price-only | 272,201,560 | 5,745,812,652 valuations |
| Notebook 08 combined | 428,521,964 | 6,079,061,481 valuations |

Against slower standard QuantLib engines, break-even is materially lower because each avoided numerical valuation is more expensive.

> **Practical conclusion:** use numerical pricing for one-off, low-volume, changing, or extreme-regime work. Use a validated neural surrogate when large portfolios or scenario grids require the same in-domain pricing function to be evaluated repeatedly at a scale of millions or billions of valuations.

---

## Hypothesis decisions

The six hypotheses were defined before the final consolidated evaluation.

| Hypothesis | Decision | Primary evidence |
|---|---|---|
| **H1 — Direct pricing approximation** | **Supported** | Direct/proxy MAE ratio = 0.058341 |
| **H2 — Premium decomposition** | **Supported** | Selected residual/direct MAE ratio = 0.130322 |
| **H3 — Financial constraints** | **Supported** | Direct violation rate = 0.30812892; constrained rate = 0 |
| **H4 — Multi-task learning** | **Not supported** | Multi-task F1 change = -0.001152; relative boundary-price MAE improvement = -51.6000% |
| **H5 — Computational acceleration** | **Supported** | Formal marginal-runtime rule passes; dedicated scaling benchmark shows 7.45× and 6.15× speedups at one million valuations |
| **H6 — OOD deterioration** | **Supported** | 7/7 eligible models deteriorate materially; minimum aggregate ratio = 8.127309 |

The hypotheses support a narrow conclusion: **deep learning is useful as a structured, high-volume surrogate, not as a universal replacement for numerical pricing.**

---

## Financial and mathematical foundation

### European benchmark

Under Black–Scholes–Merton with continuous dividend yield, the European put value is

$$
P_E = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1),
$$

where

$$
d_1 =
\frac{
\ln(S/K)
+
\left(r-q+\frac{1}{2}\sigma^2\right)T
}{
\sigma\sqrt{T}
},
\qquad
d_2=d_1-\sigma\sqrt{T}.
$$

The European price is used as:

1. an analytical benchmark;
2. a convergence target for the European CRR tree;
3. a lower bound for the American option;
4. a known component in residual learning.

### American optimal stopping

At time $t$, the option holder compares immediate exercise

$$
I(S_t)=\max(K-S_t,0)
$$

with continuation value

$$
C(S_t,t) =
\mathbb{E}^{\mathbb{Q}}
\left[
e^{-r\Delta t}
V_A(S_{t+\Delta t},t+\Delta t)
\mid S_t
\right].
$$

The American value is

$$
V_A(S_t,t) =
\max\left(I(S_t),C(S_t,t)\right).
$$

### Static neural input

Every static model receives one five-variable contract state:

$$
x =
\left[
\log(S/K),
T,
r,
q,
\sigma
\right].
$$

This is not a sequence. For that reason, the project does not add an LSTM, GRU, or Transformer merely to increase architectural complexity. Recurrent architectures are appropriate when the input has a genuine temporal structure. Here, a feed-forward network is the correct control architecture for a static pricing surface.

---

## Experimental design

### Synthetic dataset

A clean public dataset containing millions of American option contracts, complete numerical reference values, continuation values, exercise labels, and controlled stress regimes is not readily available in the required form.

The project therefore generates a synthetic dataset from a validated **250-step CRR American put pricer**.

| Component | Observations | Role |
|---|---:|---|
| Core domain | 1,000,000 | Main interpolation domain |
| Boundary-focused sample | 250,000 | Improves representation near the exercise transition |
| OOD high volatility | 50,000 | Volatility above the training maximum |
| OOD extreme moneyness | 50,000 | Deep ITM and deep OTM contracts |
| OOD long maturity | 50,000 | Maturities beyond two years |
| OOD rate/dividend | 50,000 | Exceptional rate and dividend combinations |
| **Total** | **1,450,000** | |

The generation pipeline uses:

- deterministic randomized Latin hypercube sampling;
- a Numba-accelerated CRR batch pricer;
- chunked Parquet output;
- restartable component generation;
- fixed global `sample_id` values;
- raw and financially repaired prices;
- SHA-256 component fingerprints;
- a production manifest recording the full configuration.

### Generated targets

Each record includes:

- American price;
- European price;
- intrinsic value;
- root continuation value;
- early-exercise premium;
- exercise-now label;
- normalized price and residual targets;
- boundary distance;
- CRR step count;
- any pricing-floor adjustment.

### Split and leakage controls

The core and boundary components are divided approximately into:

- 70% training;
- 15% validation;
- 15% test.

The split is deterministic and stratified by the exercise label. OOD components are never eligible for training.

The project treats leakage control as a first-class requirement:

- feature scalers are fitted only on training observations;
- validation data select checkpoints, thresholds, and loss configurations;
- the test set is not used for model selection;
- OOD sets are reserved exclusively for robustness analysis;
- Notebook 09 verifies identical `sample_id` membership and identical targets before comparing static models.

---

## Model families

### 1. Direct MLP — Notebook 04

A conventional four-layer MLP predicts the complete normalized American price.

```text
Input(5)
→ 128 → BatchNorm → SiLU
→ 128 → BatchNorm → SiLU
→ 64  → SiLU
→ 32  → SiLU
→ 1   → Softplus
```

Purpose: establish whether basic deep-learning function approximation improves on the European proxy.

### 2. Premium and floor-residual MLPs — Notebook 05

The hidden backbone is held constant while the target formulation changes.

The selected model predicts

$$
R_F =
\frac{V_A-\max(V_E,I)}{K}
$$

with a non-negative output.

Purpose: test whether financial structure and residual learning matter more than additional network complexity.

### 3. Exercise-only and multi-task models — Notebook 06

The exercise specialist predicts

$$
Y_E =
\mathbb{1}[I\geq C].
$$

A second model shares one backbone between a constrained pricing head and an exercise head.

Purpose: test whether joint representation learning improves the exercise boundary without sacrificing pricing.

### 4. Classical and neural Longstaff–Schwartz — Notebook 07

The classical experiment estimates continuation values from simulated paths using regression basis functions. The neural version replaces the continuation regression with time-indexed MLPs.

Purpose: test whether deep learning improves the numerical optimal-stopping algorithm itself, rather than only approximating a static pricing surface.

### 5. Integrated four-head model — Notebook 08

```text
Input(5)
→ Shared backbone
   ├── Financial-floor residual head
   ├── Direct-price head
   ├── Continuation-value head
   └── Exercise-classification head
```

The authoritative price is the constrained residual reconstruction. The direct-price and continuation outputs are supporting heads, while the exercise head produces the deployment decision.

Purpose: test whether one larger model can combine several related outputs without losing too much specialist performance.

---

## Model architecture summary

Notebook 01 instantiates every neural architecture on CPU with one five-feature observation and reports the executed `torchinfo` summaries. The table below condenses those outputs.

| Model | Architecture | Outputs | Trainable parameters |
|---|---|---|---:|
| **Direct American put MLP** | $5 \rightarrow 128 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$, with batch normalization after the first two hidden layers and `SiLU` activations | Non-negative normalized American price | **28,161** |
| **Unconstrained premium MLP** | Same four-layer backbone as the direct MLP; linear final output | Normalized early-exercise premium | **28,161** |
| **Constrained floor-residual MLP** | Same four-layer backbone; `Softplus` residual output | Non-negative residual above $\max(V_E,I)$ | **28,161** |
| **Exercise-only classifier** | Same four-layer backbone; linear logit output | Exercise-versus-continuation logit | **28,161** |
| **Notebook 06 multi-task MLP** | Shared $5 \rightarrow 128 \rightarrow 128 \rightarrow 64$ backbone with separate $64 \rightarrow 32 \rightarrow 1$ pricing and classification heads | Floor residual and exercise logit | **30,274** |
| **Neural LSM continuation network** | $5 \rightarrow 64 \rightarrow 64 \rightarrow 32 \rightarrow 1$ with `SiLU` and `Softplus` output | Non-negative continuation value at one exercise date | **6,657** per time-indexed network |
| **Notebook 08 integrated multi-head MLP** | Shared $5 \rightarrow 192 \rightarrow 192 \rightarrow 96$ backbone with four $96 \rightarrow 48 \rightarrow 1$ heads | Floor residual, direct price, continuation value, and exercise logit | **76,324** |

The architecture comparison supports the project’s main modelling conclusion. Notebook 05 holds model capacity constant and changes only the target construction and output constraint, so its improvement is attributable to financial structure rather than a larger network. The integrated model has the greatest capacity, but it does not produce the best specialist price. Recurrent architectures are not used because the static pricing input $[\log(S/K),T,r,q,\sigma]$ is not a sequence.

---

## Notebook workflow

The nine notebooks form one sequential research argument.

| Notebook | Role | Final conclusion |
|---|---|---|
| **01 — Option pricing foundations** | Defines American optimal stopping, literature context, research questions, and hypotheses | Deep learning must be justified by the American problem and repeated computation |
| **02 — American option data generation** | Validates pricing engines, studies CRR convergence, and designs synthetic generation | A 250-step CRR engine provides a controlled production reference |
| **03 — Dataset analysis and validation** | Audits the 1.45 million rows, freezes splits, and defines OOD regimes | The dataset is suitable for controlled interpolation and stress testing |
| **04 — Direct MLP pricer** | Trains the direct neural baseline | H1 supported |
| **05 — Early-exercise premium models** | Compares direct, premium, and financially constrained targets | H2 and H3 supported; constrained residual is best static price |
| **06 — Exercise-boundary analysis** | Trains specialist and multi-task exercise models | Joint learning is feasible but H4 is not supported |
| **07 — Neural Longstaff–Schwartz** | Compares classical and neural continuation regression | Classical LSM is preferred |
| **08 — Final integrated multi-head model** | Combines price, continuation, and exercise outputs | Useful combined deployment, but not the overall winner |
| **09 — Final evaluation** | Aligns all evidence, decides H1–H6, audits artifacts, and evaluates the business case | Deep learning is justified conditionally for repeated high-volume in-domain pricing |

Notebook 09 is the authoritative project conclusion.

---

## Technical documentation

The `docs/` directory is an engineering reference, not a second copy of the notebooks.

Start with [docs/README.md](docs/README.md).

| Document | Scope |
|---|---|
| [System architecture](docs/01_system_architecture.md) | End-to-end data, training, artifact, and evaluation flow |
| [Pricing engines](docs/02_pricing_engines.md) | Black–Scholes, CRR, QuantLib, simulation, and LSM |
| [Synthetic data pipeline](docs/03_synthetic_data_pipeline.md) | Sampling, Numba pricing, components, chunking, and manifests |
| [Dataset schema and splits](docs/04_dataset_schema_and_splits.md) | Formal column contract, normalization, and leakage controls |
| [Model architectures](docs/05_model_architectures.md) | All static and path-based neural architectures |
| [Training and artifact management](docs/06_training_and_artifact_management.md) | Training loops, checkpoints, completion manifests, and fingerprints |
| [Evaluation framework](docs/07_evaluation_framework.md) | Metrics, financial checks, OOD tests, and aligned comparisons |
| [Runtime and business case](docs/08_runtime_and_business_case.md) | Scaling, crossover, annual workloads, and lifecycle break-even |
| [Reproducibility and execution](docs/09_reproducibility_and_execution.md) | Environment, commands, profiles, and troubleshooting |
| [Results reference](docs/10_results_reference.md) | Frozen headline results and artifact sources |

---

## Repository architecture

```text
deep_learning_american_option_pricing/
├── notebooks/                  # Nine executable research notebooks
├── docs/                       # Technical engineering documentation
├── src/
│   ├── pricing/                # Black–Scholes, CRR, simulation, LSM
│   ├── data/                   # Generation, validation, splits, datasets
│   ├── models/                 # Direct, residual, multi-task, LSM, multi-head
│   ├── training/               # Losses, loops, checkpoints, lineage metadata
│   └── evaluation/             # Metrics, audits, comparisons, business case
├── scripts/                    # Production generation and validation entry points
├── tests/                      # Unit and integration tests
├── data/
│   ├── generated/              # DVC-managed production Parquet files
│   └── manifests/              # Tracked design and generation metadata
├── artifacts/                  # DVC-managed checkpoints and final outputs
├── references/                 # Papers, bibliography, and citation audit
├── requirements.txt
├── pytest.ini
├── LICENSE
└── README.md
```

---

## Quick start for reviewers and graders

> [!IMPORTANT]
> A normal Git clone contains the code and lightweight DVC pointers, but not the large production dataset or trained artifacts. Run `dvc pull` before opening the load-mode model notebooks.

### Windows PowerShell

```powershell
git clone https://github.com/Kamend1/deep_learning_american_option_pricing.git
cd deep_learning_american_option_pricing

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

dvc pull
```

### macOS or Linux

```bash
git clone https://github.com/Kamend1/deep_learning_american_option_pricing.git
cd deep_learning_american_option_pricing

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

dvc pull
```

The default DVC remote is publicly readable. Downloading the published dataset and model artifacts does not require Cloudflare credentials.

Saved checkpoints and the final evaluation can be reviewed on CPU. CUDA is useful for full retraining; install the appropriate CUDA-enabled PyTorch build for the local machine before installing the remaining dependencies.

After `dvc pull`, the principal restored paths are:

```text
data/generated/
artifacts/direct_mlp/
artifacts/premium_models/
artifacts/multitask_model/
artifacts/neural_lsm/
artifacts/final_multihead/
artifacts/final_evaluation/
```

Verify the local state:

```bash
dvc status
```

---

## Reproducing the project

### Load and review the submitted models

The submitted model notebooks are designed to reuse complete saved training packages.

```python
FORCE_TRAIN = False
```

Each notebook validates its required checkpoint, scaler, manifest, training profile, and dependency fingerprints before loading the package.

Recommended review order:

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

### Regenerate the full dataset

```bash
python scripts/generate_production_dataset.py
```

The script is component-restartable and writes the complete generation manifest only after all six components are present and verified.

### Validate the final production package

```bash
python scripts/validate_production_project.py --deep
```

### Rebuild consolidated final results

```bash
python scripts/build_final_results.py --strict
```

Notebook 09 performs additional checks before comparing models:

- artifact presence and schema validation;
- checkpoint and manifest coherence;
- training-profile validation;
- dependency fingerprint inventory;
- duplicate-ID detection;
- common static-test alignment;
- true-target equality;
- shared state-field equality;
- final export readiness.

---

## Running the tests

Run the default test suite:

```bash
python -m pytest -q
```

Run pricing-engine tests:

```bash
python -m pytest -q \
    tests/test_black_scholes.py \
    tests/test_binomial_tree.py
```

Run data and split tests:

```bash
python -m pytest -q \
    tests/test_data_generation.py \
    tests/test_data_splitting.py \
    tests/test_torch_datasets.py
```

Run model and training tests:

```bash
python -m pytest -q \
    tests/test_direct_pricer.py \
    tests/test_premium_pricer.py \
    tests/test_multitask_pricer.py \
    tests/test_integrated_multihead_pricer.py \
    tests/test_training_pipeline.py
```

Run simulation and Longstaff–Schwartz tests:

```bash
python -m pytest -q \
    tests/test_gbm_simulation.py \
    tests/test_longstaff_schwartz.py \
    tests/test_neural_longstaff_schwartz.py \
    tests/test_lsm_comparison.py
```

Run integration tests explicitly:

```bash
python -m pytest -q -m integration
```

The default `pytest` configuration excludes tests marked `integration` and `slow`. Full production generation and full model training are not unit-test workloads.

---

## Data and artifact access

Large generated data and fitted-model outputs are versioned with **DVC** and stored in a publicly readable Cloudflare R2 remote.

Git tracks:

- source code;
- notebooks;
- technical documentation;
- tests;
- DVC pointer files;
- data and experiment manifests;
- references and bibliography.

DVC manages:

- production Parquet datasets;
- model checkpoints;
- feature scalers;
- training histories;
- prediction files;
- runtime results;
- final consolidated evaluation artifacts.

This separation keeps the Git repository reviewable while preserving exact reproducibility of the large assets.

---

## Limitations

The conclusions are deliberately narrower than the full real-world option-pricing problem.

- **Synthetic reference prices:** the models reproduce a high-resolution CRR pricing rule; they are not validated against traded option prices.
- **Constant volatility:** the core experiment does not model stochastic or local volatility.
- **Continuous dividends:** discrete dividend events are excluded.
- **Fixed payoff family:** the trained surrogates apply to American puts under the documented assumptions, not arbitrary derivatives.
- **Static inputs:** the models do not forecast the underlying asset or learn from market time series.
- **Finite-tree labels:** exercise decisions and continuation values inherit CRR resolution and model assumptions.
- **Separate path experiment:** Longstaff–Schwartz results use simulated paths and cannot be ranked directly with static models.
- **OOD risk:** the broad in-domain range covers typical project use cases, but extrapolation into stress regimes remains unreliable.
- **Partial financial constraints:** lower bounds are guaranteed; every possible monotonicity and no-arbitrage relationship is not.
- **Runtime dependence:** throughput depends on hardware, process state, batch size, device, software versions, and serving design.
- **Lifecycle cost:** the business case depends on repeated volume and the cost of generating and maintaining reference labels.

These limitations are part of the result. They define where the surrogate may be used and where numerical pricing must remain authoritative.

---

## Selected references

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.
- Merton, R. C. (1973). *Theory of Rational Option Pricing*.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach*.
- Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach*.
- Ke, A., & Yang, A. (2019). *Option Pricing with Deep Learning*.
- Pu, V. R. H. (2021). *Pricing Options Using Deep Neural Networks from a Practical Perspective*.
- Zouaoui, H., & Naas, M.-N. (2023). *Option Pricing Using Deep Learning Based on LSTM-GRU Neural Networks*.
- Ding, L., Lu, E., & Cheung, K. (2025). *Deep Learning Option Pricing with Market Implied Volatility Surfaces*.
- Elbayed, Z., & Qadi El Idrissi, A. (2025). *Deep Learning in Financial Modeling: Predicting European Put Option Prices with Neural Networks*.
- Pimentel, R., et al. (2026). *Option Pricing with Deep Learning: A Long Short-Term Memory Approach*.

The complete bibliography and literature mapping are maintained in `references/` and discussed in Notebook 01 and Notebook 09.

---

## Connection to previous work

This project extends the Black–Scholes, Greeks, and hedging work developed in:

[Mathematical Foundations of Portfolio Optimization and Hedging](https://github.com/Kamend1/math_for_developers_final_project)

The earlier project focused on closed-form European option valuation and portfolio applications. This project moves to American optimal stopping, synthetic numerical labels, deep-learning surrogates, explicit exercise decisions, and computational deployment economics.

---

## Project status

**Complete.**

The final repository includes:

- nine completed research notebooks;
- validated Black–Scholes and CRR pricing engines;
- a 1.45 million-observation synthetic production dataset;
- direct, residual, constrained, exercise, multi-task, neural-LSM, and integrated models;
- common-test, segmented, financial, boundary, OOD, uncertainty, and runtime evaluation;
- formal H1–H6 decisions;
- final model-selection and deployment recommendations;
- artifact lineage and readiness audits;
- DVC-managed data and trained models;
- unit and integration tests;
- detailed technical documentation.

The final project conclusion is contained in **Notebook 09 — Final Evaluation**.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Kamen Dimitrov, CFA**

Finance professional and participant in the SoftUni AI and Machine Learning Upskill Program.

Project repository:

[github.com/Kamend1/deep_learning_american_option_pricing](https://github.com/Kamend1/deep_learning_american_option_pricing)
