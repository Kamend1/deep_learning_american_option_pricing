<!--
Generated from notebooks/07_neural_longstaff_schwartz.ipynb.
The notebook is the executable source of truth.
-->

# Classical and Neural Longstaff–Schwartz for American Put Options

## Learning continuation values from simulated risk-neutral paths

**SoftUni Deep Learning Final Project**  
**Author:** Kamen Dimitrov  
**Notebook:** `07_neural_longstaff_schwartz.ipynb`

---

The earlier notebooks learn a static American-option pricing surface from CRR-generated labels. This notebook studies a different deep-learning application: the neural network participates directly in the optimal-stopping algorithm by estimating continuation values from simulated paths.

The experiment compares:

1. a high-resolution Cox–Ross–Rubinstein tree;
2. classical Least-Squares Monte Carlo with polynomial or Laguerre basis functions;
3. an amortized neural Least-Squares Monte Carlo policy.

The stopping policy is always fitted on one set of paths and valued on an independent set. This separation is necessary because evaluating a policy on the paths used to estimate it produces an optimistic result.

## 1. Research objective

Longstaff and Schwartz (2001) introduced a practical regression-based Monte Carlo method for American options. At each exercise date, discounted future cash flows are regressed on functions of the current state. The estimated continuation value is then compared with intrinsic value.

The central question is:

> At a matched simulation budget, can neural continuation-value regression estimate American put prices and exercise policies more accurately than classical polynomial Longstaff–Schwartz?

This experiment complements the static neural surrogates developed in Notebooks 04–06. It does not replace them. The static models are expected to remain substantially faster for repeated inference after training, while neural LSM provides a more direct solution to the stopping problem.

## 2. Risk-neutral simulation

Under risk-neutral geometric Brownian motion,

\[
S_{t+\Delta t}
=
S_t\exp\left[
\left(r-q-\frac{1}{2}\sigma^2\right)\Delta t
+
\sigma\sqrt{\Delta t}Z
\right].
\]

The simulator uses fixed seeds and antithetic variates. Antithetic sampling reduces variance by pairing every normal shock sequence with its negative. The theoretical model is unchanged.

```python
from pathlib import Path
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.lsm_comparison import (
    aggregate_seed_results,
    compare_lsm_methods,
    confidence_interval_coverage,
    exercise_policy_metrics,
    runtime_summary,
    stopping_time_total_variation,
)
from src.models.neural_longstaff_schwartz import (
    ContinuationNetworkConfig,
    ContractPathBatch,
)
from src.pricing.binomial_tree import crr_option_price
from src.pricing.longstaff_schwartz import longstaff_schwartz_put_price
from src.pricing.simulation import (
    GBMContract,
    sample_contracts_latin_hypercube,
    simulate_contract_paths,
    validate_simulated_moments,
)
from src.training.lsm_training import (
    NeuralLSMTrainingConfig,
    evaluate_neural_lsm_policy,
    fit_neural_lsm_policy,
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "neural_lsm"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "data" / "manifests" / "lsm_experiment_config.json"

print(f"Project root: {PROJECT_ROOT}")
print(f"Torch device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

## 3. Execution profiles

The full experiment is computationally material. Three explicit profiles are supplied:

- **smoke** — verifies the full workflow locally;
- **standard** — produces intermediate research results;
- **final** — uses the approved contract and simulation budgets.

The final profile uses 200 training contracts, 50 validation contracts, 100 held-out contracts, 40 out-of-domain contracts, 25 exercise dates, 25,000 policy-training paths per contract, and 50,000 independent valuation paths per contract.

```python
with CONFIG_PATH.open("r", encoding="utf-8") as file:
    experiment_design = json.load(file)

PROFILE = os.getenv("LSM_PROFILE", experiment_design["default_profile"])
if PROFILE not in experiment_design["profiles"]:
    raise ValueError(f"Unknown LSM profile: {PROFILE}")

profile = experiment_design["profiles"][PROFILE]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(experiment_design["seeds"][0])

pd.Series(profile, name=PROFILE).to_frame()
```

## 4. Representative contract and simulation diagnostics

The first calculation validates the simulator independently of the stopping algorithm. The empirical terminal mean and variance should be close to their theoretical risk-neutral values. This check helps distinguish simulation errors from continuation-regression errors later in the notebook.

```python
representative_contract = GBMContract(
    contract_id="representative",
    spot=90.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    dividend_yield=0.01,
    volatility=0.20,
)

representative_paths = simulate_contract_paths(
    representative_contract,
    n_paths=max(20_000, profile["valuation_paths_per_contract"]),
    n_steps=profile["exercise_steps"],
    seed=SEED,
)

moment_validation = validate_simulated_moments(
    representative_paths,
    representative_contract,
)
pd.Series(moment_validation.to_dict(), name="Value").to_frame()
```

```python
plt.figure(figsize=(11, 6))
time_grid = np.linspace(
    0.0,
    representative_contract.time_to_maturity,
    profile["exercise_steps"] + 1,
)
for path in representative_paths[:40]:
    plt.plot(time_grid, path, alpha=0.35)
plt.axhline(representative_contract.strike, linestyle="--", label="Strike")
plt.xlabel("Time")
plt.ylabel("Underlying price")
plt.title("Representative Risk-Neutral GBM Paths")
plt.legend()
plt.grid(alpha=0.20)
plt.show()
```

The path chart is descriptive rather than evidential. The terminal-moment diagnostics provide the actual numerical validation. The stopping policy is examined only after the simulated process behaves consistently with its theoretical specification.

## 5. Classical Longstaff–Schwartz benchmark

Classical LSM regresses discounted future cash flows on basis functions of normalized spot. Three predefined specifications are compared:

- degree-2 polynomial;
- degree-3 polynomial;
- degree-3 weighted Laguerre basis.

The specification is selected on validation evidence rather than the final held-out contracts.

```python
training_paths = simulate_contract_paths(
    representative_contract,
    n_paths=profile["policy_training_paths_per_contract"],
    n_steps=profile["exercise_steps"],
    seed=SEED + 1,
)
valuation_paths = simulate_contract_paths(
    representative_contract,
    n_paths=profile["valuation_paths_per_contract"],
    n_steps=profile["exercise_steps"],
    seed=SEED + 2,
)

crr_representative = crr_option_price(
    spot=representative_contract.spot,
    strike=representative_contract.strike,
    time_to_maturity=representative_contract.time_to_maturity,
    risk_free_rate=representative_contract.risk_free_rate,
    dividend_yield=representative_contract.dividend_yield,
    volatility=representative_contract.volatility,
    steps=experiment_design["crr_benchmark_steps"],
    option_type="put",
    exercise_style="american",
)

basis_rows = []
basis_experiments = {}
for candidate in experiment_design["classical_basis_candidates"]:
    started = time.perf_counter()
    result = longstaff_schwartz_put_price(
        training_paths,
        valuation_paths,
        strike=representative_contract.strike,
        time_to_maturity=representative_contract.time_to_maturity,
        risk_free_rate=representative_contract.risk_free_rate,
        basis=candidate["basis"],
        degree=candidate["degree"],
    )
    runtime = time.perf_counter() - started
    name = f"{candidate['basis']}_degree_{candidate['degree']}"
    basis_experiments[name] = result
    basis_rows.append(
        {
            "specification": name,
            "price": result.valuation.price,
            "standard_error": result.valuation.standard_error,
            "ci_low": result.valuation.confidence_interval_low,
            "ci_high": result.valuation.confidence_interval_high,
            "absolute_error_vs_crr": abs(result.valuation.price - crr_representative),
            "early_exercise_rate": result.valuation.exercised_early_rate,
            "runtime_seconds": runtime,
        }
    )

basis_results = pd.DataFrame(basis_rows).sort_values("absolute_error_vs_crr")
basis_results
```

```python
BEST_CLASSICAL_SPECIFICATION = basis_results.iloc[0]["specification"]
best_classical = basis_experiments[BEST_CLASSICAL_SPECIFICATION]
BEST_CLASSICAL_BASIS = best_classical.policy.basis
BEST_CLASSICAL_DEGREE = best_classical.policy.degree

print(f"CRR benchmark: {crr_representative:.6f}")
print(f"Selected classical specification: {BEST_CLASSICAL_SPECIFICATION}")
```

The selected basis is not assumed to be universally optimal. Basis risk is an inherent limitation of classical LSM. A specification that performs well for one contract may perform differently across other parts of the pricing domain.

## 6. Path-count convergence

Monte Carlo estimates must be reported together with sampling uncertainty. The next experiment varies the path budget while keeping training and valuation samples independent.

```python
path_counts = [2_000, 5_000, 10_000]
if PROFILE != "smoke":
    path_counts.extend([25_000, 50_000])

path_convergence_rows = []
for count in path_counts:
    train_sample = simulate_contract_paths(
        representative_contract,
        n_paths=count,
        n_steps=profile["exercise_steps"],
        seed=SEED + 100 + count,
    )
    valuation_sample = simulate_contract_paths(
        representative_contract,
        n_paths=2 * count,
        n_steps=profile["exercise_steps"],
        seed=SEED + 200 + count,
    )
    started = time.perf_counter()
    experiment = longstaff_schwartz_put_price(
        train_sample,
        valuation_sample,
        strike=representative_contract.strike,
        time_to_maturity=representative_contract.time_to_maturity,
        risk_free_rate=representative_contract.risk_free_rate,
        basis=BEST_CLASSICAL_BASIS,
        degree=BEST_CLASSICAL_DEGREE,
    )
    path_convergence_rows.append(
        {
            "policy_paths": count,
            "valuation_paths": 2 * count,
            "price": experiment.valuation.price,
            "standard_error": experiment.valuation.standard_error,
            "ci_low": experiment.valuation.confidence_interval_low,
            "ci_high": experiment.valuation.confidence_interval_high,
            "absolute_error_vs_crr": abs(
                experiment.valuation.price - crr_representative
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
    )

path_convergence = pd.DataFrame(path_convergence_rows)
path_convergence
```

## 7. Exercise-date convergence

The American contract allows continuous exercise, while LSM evaluates a discrete set of exercise dates. Increasing the number of dates reduces discretization error but increases regression complexity and runtime.

```python
exercise_step_grid = [10, 25, 50]
if PROFILE == "smoke":
    exercise_step_grid = [5, 10, 20]

exercise_date_rows = []
for n_steps in exercise_step_grid:
    train_sample = simulate_contract_paths(
        representative_contract,
        n_paths=profile["policy_training_paths_per_contract"],
        n_steps=n_steps,
        seed=SEED + 300 + n_steps,
    )
    valuation_sample = simulate_contract_paths(
        representative_contract,
        n_paths=profile["valuation_paths_per_contract"],
        n_steps=n_steps,
        seed=SEED + 400 + n_steps,
    )
    started = time.perf_counter()
    experiment = longstaff_schwartz_put_price(
        train_sample,
        valuation_sample,
        strike=representative_contract.strike,
        time_to_maturity=representative_contract.time_to_maturity,
        risk_free_rate=representative_contract.risk_free_rate,
        basis=BEST_CLASSICAL_BASIS,
        degree=BEST_CLASSICAL_DEGREE,
    )
    exercise_date_rows.append(
        {
            "exercise_steps": n_steps,
            "price": experiment.valuation.price,
            "standard_error": experiment.valuation.standard_error,
            "absolute_error_vs_crr": abs(
                experiment.valuation.price - crr_representative
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
    )

exercise_date_convergence = pd.DataFrame(exercise_date_rows)
exercise_date_convergence
```

## 8. Contract-level research design

The neural policy is amortized across a domain of contracts. One continuation network is trained for each exercise-time index, while contract parameters enter the network as features. Test-contract identifiers are excluded from both network fitting and validation.

The neural input vector is

\[
\left[
\log(S/K),\;
T-t,\;
r,\;
q,\;
\sigma
\right].
\]

The target is the discounted future cash flow normalized by strike.

```python
in_domain_ranges = {
    key: tuple(value)
    for key, value in experiment_design["in_domain_ranges"].items()
}

training_contracts = sample_contracts_latin_hypercube(
    n_contracts=profile["training_contracts"],
    parameter_ranges=in_domain_ranges,
    seed=SEED + 500,
    prefix="train",
)
validation_contracts = sample_contracts_latin_hypercube(
    n_contracts=profile["validation_contracts"],
    parameter_ranges=in_domain_ranges,
    seed=SEED + 600,
    prefix="validation",
)
test_contracts = sample_contracts_latin_hypercube(
    n_contracts=profile["test_contracts"],
    parameter_ranges=in_domain_ranges,
    seed=SEED + 700,
    prefix="test",
)

contract_grid = pd.DataFrame(
    [
        {**contract.to_dict(), "sample": sample}
        for sample, contracts in [
            ("training", training_contracts),
            ("validation", validation_contracts),
            ("test", test_contracts),
        ]
        for contract in contracts
    ]
)
contract_grid.groupby("sample").size().to_frame("contracts")
```

```python
def build_path_batches(contracts, *, n_paths, n_steps, seed_offset):
    batches = []
    for index, contract in enumerate(contracts):
        paths = simulate_contract_paths(
            contract,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=SEED + seed_offset + index,
        )
        batches.append(ContractPathBatch(contract=contract, paths=paths))
    return batches

training_batches = build_path_batches(
    training_contracts,
    n_paths=profile["policy_training_paths_per_contract"],
    n_steps=profile["exercise_steps"],
    seed_offset=1_000,
)
validation_batches = build_path_batches(
    validation_contracts,
    n_paths=profile["policy_training_paths_per_contract"],
    n_steps=profile["exercise_steps"],
    seed_offset=2_000,
)
```

## 9. Neural continuation-policy training

The continuation network uses a non-negative `Softplus` output. This is economically appropriate because continuation value cannot be negative. Networks are trained backward from the latest exercise date, and the next earlier network may warm-start from the later network's weights.

The neural policy is evaluated only on independent paths from contracts excluded from policy training.

```python
network_config = ContinuationNetworkConfig(
    input_dim=5,
    hidden_dims=(64, 64, 32),
    dropout=0.0,
)
training_config = NeuralLSMTrainingConfig(
    network=network_config,
    epochs=profile["neural_epochs_per_step"],
    batch_size=1024,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=max(3, min(8, profile["neural_epochs_per_step"] // 3)),
    minimum_samples_per_step=64,
    maximum_samples_per_step=profile["max_training_samples_per_step"],
    warm_start_from_later_step=True,
    seed=SEED,
    device=DEVICE,
)

neural_training_started = time.perf_counter()
neural_policy, neural_training_history = fit_neural_lsm_policy(
    training_batches,
    validation_batches,
    config=training_config,
)
neural_training_runtime = time.perf_counter() - neural_training_started

print(f"Trained continuation networks: {len(neural_policy.steps)}")
print(f"Training runtime: {neural_training_runtime:.2f} seconds")
neural_training_history.tail()
```

```python
if not neural_training_history.empty:
    selected_steps = sorted(neural_training_history["step_index"].unique())
    selected_steps = selected_steps[:: max(len(selected_steps) // 4, 1)]
    plt.figure(figsize=(11, 6))
    for step_index in selected_steps:
        subset = neural_training_history[
            neural_training_history["step_index"] == step_index
        ]
        plt.plot(
            subset["epoch"],
            subset["validation_loss"],
            label=f"Step {step_index}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Smooth L1 loss")
    plt.title("Neural Continuation Validation Loss")
    plt.legend()
    plt.grid(alpha=0.20)
    plt.show()
```

## 10. Held-out contract evaluation

For every held-out contract:

1. a CRR benchmark is calculated;
2. a classical policy is fitted on contract-specific training paths;
3. classical and neural policies are evaluated on the same independent valuation paths.

Using common valuation paths reduces noise in the policy comparison.

```python
heldout_rows = []
heldout_policy_rows = []
runtime_rows = []
contract_valuation_cache = {}

for index, contract in enumerate(test_contracts):
    policy_paths = simulate_contract_paths(
        contract,
        n_paths=profile["policy_training_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=SEED + 3_000 + index,
    )
    valuation_paths = simulate_contract_paths(
        contract,
        n_paths=profile["valuation_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=SEED + 4_000 + index,
    )
    valuation_batch = ContractPathBatch(contract=contract, paths=valuation_paths)

    started = time.perf_counter()
    classical = longstaff_schwartz_put_price(
        policy_paths,
        valuation_paths,
        strike=contract.strike,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        basis=BEST_CLASSICAL_BASIS,
        degree=BEST_CLASSICAL_DEGREE,
    )
    classical_runtime = time.perf_counter() - started

    started = time.perf_counter()
    neural = evaluate_neural_lsm_policy(
        neural_policy,
        valuation_batch,
        device=DEVICE,
    )
    neural_runtime = time.perf_counter() - started

    started = time.perf_counter()
    crr_price = crr_option_price(
        spot=contract.spot,
        strike=contract.strike,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        dividend_yield=contract.dividend_yield,
        volatility=contract.volatility,
        steps=experiment_design["crr_benchmark_steps"],
        option_type="put",
        exercise_style="american",
    )
    crr_runtime = time.perf_counter() - started

    heldout_rows.append(
        {
            "contract_id": contract.contract_id,
            "crr_price": crr_price,
            "classical_lsm_price": classical.valuation.price,
            "classical_lsm_standard_error": classical.valuation.standard_error,
            "classical_lsm_ci_low": classical.valuation.confidence_interval_low,
            "classical_lsm_ci_high": classical.valuation.confidence_interval_high,
            "neural_lsm_price": neural.price,
            "neural_lsm_standard_error": neural.standard_error,
            "neural_lsm_ci_low": neural.confidence_interval_low,
            "neural_lsm_ci_high": neural.confidence_interval_high,
        }
    )
    policy_metrics = exercise_policy_metrics(
        classical.valuation.exercise_indices,
        neural.exercise_indices,
        maturity_index=profile["exercise_steps"],
    )
    policy_metrics["stopping_distribution_tv"] = stopping_time_total_variation(
        classical.valuation.exercise_indices,
        neural.exercise_indices,
        n_steps=profile["exercise_steps"],
    )
    heldout_policy_rows.append(
        {"contract_id": contract.contract_id, **policy_metrics}
    )
    runtime_rows.extend(
        [
            {"contract_id": contract.contract_id, "method": "CRR", "runtime_seconds": crr_runtime},
            {"contract_id": contract.contract_id, "method": "Classical LSM", "runtime_seconds": classical_runtime},
            {"contract_id": contract.contract_id, "method": "Neural LSM evaluation", "runtime_seconds": neural_runtime},
        ]
    )
    contract_valuation_cache[contract.contract_id] = {
        "classical": classical.valuation,
        "neural": neural,
    }

heldout_results = pd.DataFrame(heldout_rows)
heldout_results.head()
```

```python
heldout_comparison = compare_lsm_methods(
    heldout_results,
    benchmark_column="crr_price",
    method_columns=["classical_lsm_price", "neural_lsm_price"],
)
heldout_comparison
```

```python
coverage_summary = pd.Series(
    {
        "Classical LSM 95% CI coverage": confidence_interval_coverage(
            heldout_results["crr_price"],
            heldout_results["classical_lsm_ci_low"],
            heldout_results["classical_lsm_ci_high"],
        ),
        "Neural LSM 95% CI coverage": confidence_interval_coverage(
            heldout_results["crr_price"],
            heldout_results["neural_lsm_ci_low"],
            heldout_results["neural_lsm_ci_high"],
        ),
    }
)
coverage_summary.to_frame("Coverage")
```

```python
heldout_policy_metrics = pd.DataFrame(heldout_policy_rows)
heldout_policy_metrics.describe().T
```

Pricing agreement and policy agreement answer different questions. Two policies can generate similar prices while exercising different paths at different dates. The stopping-time comparison therefore complements the contract-level pricing errors.

## 11. Out-of-domain evaluation

Out-of-domain contracts deliberately exceed the training ranges. The objective is not to expect perfect extrapolation, but to measure how quickly the neural stopping policy deteriorates relative to classical LSM.

```python
ood_ranges = dict(in_domain_ranges)
ood_ranges["volatility"] = (0.70, 1.00)
ood_ranges["moneyness"] = (0.45, 1.55)
ood_contracts = sample_contracts_latin_hypercube(
    n_contracts=profile["ood_contracts"],
    parameter_ranges=ood_ranges,
    seed=SEED + 5_000,
    prefix="ood",
)

ood_rows = []
for index, contract in enumerate(ood_contracts):
    policy_paths = simulate_contract_paths(
        contract,
        n_paths=profile["policy_training_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=SEED + 6_000 + index,
    )
    valuation_paths = simulate_contract_paths(
        contract,
        n_paths=profile["valuation_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=SEED + 7_000 + index,
    )
    classical = longstaff_schwartz_put_price(
        policy_paths,
        valuation_paths,
        strike=contract.strike,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        basis=BEST_CLASSICAL_BASIS,
        degree=BEST_CLASSICAL_DEGREE,
    )
    neural = evaluate_neural_lsm_policy(
        neural_policy,
        ContractPathBatch(contract=contract, paths=valuation_paths),
        device=DEVICE,
    )
    crr_price = crr_option_price(
        spot=contract.spot,
        strike=contract.strike,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        dividend_yield=contract.dividend_yield,
        volatility=contract.volatility,
        steps=experiment_design["crr_benchmark_steps"],
        option_type="put",
        exercise_style="american",
    )
    ood_rows.append(
        {
            "contract_id": contract.contract_id,
            "crr_price": crr_price,
            "classical_lsm_price": classical.valuation.price,
            "neural_lsm_price": neural.price,
        }
    )

ood_results = pd.DataFrame(ood_rows)
ood_comparison = compare_lsm_methods(
    ood_results,
    benchmark_column="crr_price",
    method_columns=["classical_lsm_price", "neural_lsm_price"],
)
ood_comparison
```

## 12. Multi-seed robustness

Monte Carlo conclusions should not depend on one random seed. The selected representative contract is repriced with multiple independent valuation samples. The neural policy remains fixed, while both policy noise and valuation noise are visible in the classical results.

```python
robustness_rows = []
for seed in experiment_design["seeds"]:
    policy_paths = simulate_contract_paths(
        representative_contract,
        n_paths=profile["policy_training_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=seed + 8_000,
    )
    valuation_paths = simulate_contract_paths(
        representative_contract,
        n_paths=profile["valuation_paths_per_contract"],
        n_steps=profile["exercise_steps"],
        seed=seed + 9_000,
    )
    classical = longstaff_schwartz_put_price(
        policy_paths,
        valuation_paths,
        strike=representative_contract.strike,
        time_to_maturity=representative_contract.time_to_maturity,
        risk_free_rate=representative_contract.risk_free_rate,
        basis=BEST_CLASSICAL_BASIS,
        degree=BEST_CLASSICAL_DEGREE,
    )
    neural = evaluate_neural_lsm_policy(
        neural_policy,
        ContractPathBatch(
            contract=representative_contract,
            paths=valuation_paths,
        ),
        device=DEVICE,
    )
    robustness_rows.extend(
        [
            {
                "contract_id": representative_contract.contract_id,
                "seed": seed,
                "method": "classical_lsm",
                "price": classical.valuation.price,
            },
            {
                "contract_id": representative_contract.contract_id,
                "seed": seed,
                "method": "neural_lsm",
                "price": neural.price,
            },
        ]
    )

robustness_results = pd.DataFrame(robustness_rows)
aggregate_seed_results(robustness_results)
```

## 13. Runtime decomposition

The computational comparison separates:

- path simulation;
- classical policy fitting and valuation;
- neural policy training;
- marginal neural policy evaluation;
- CRR valuation.

This distinction matters because neural LSM has a high up-front policy-training cost but can reuse the policy across held-out contracts within the trained domain.

```python
runtime_records = pd.DataFrame(runtime_rows)
runtime_table = runtime_summary(runtime_records)
runtime_table
```

```python
runtime_context = pd.DataFrame(
    [
        {
            "method": "Neural LSM policy training",
            "runtime_seconds": neural_training_runtime,
            "frequency": "one-time per trained contract domain",
        },
        {
            "method": "Classical LSM",
            "runtime_seconds": runtime_records.loc[
                runtime_records["method"] == "Classical LSM",
                "runtime_seconds",
            ].mean(),
            "frequency": "repeated per contract",
        },
        {
            "method": "Neural LSM evaluation",
            "runtime_seconds": runtime_records.loc[
                runtime_records["method"] == "Neural LSM evaluation",
                "runtime_seconds",
            ].mean(),
            "frequency": "repeated per contract after policy training",
        },
    ]
)
runtime_context
```

## 14. Link to the static surrogate models

The direct, premium, and multi-task networks from Notebooks 04–06 solve a different operational problem. They map contract parameters directly to prices and are expected to remain the fastest approach for large-scale repeated valuation.

This notebook saves its held-out contract grid and CRR benchmark values so Notebook 08 can apply all trained static surrogates to exactly the same contracts. That consolidated comparison avoids rebuilding earlier model-loading logic inside the LSM analysis.

## 15. Artifacts and reproducibility

The policy, contract grid, training history, classical and neural results, stopping-policy metrics, runtime summaries, and robustness outputs are persisted. Generated model artifacts remain excluded from Git, while the experiment configuration under `data/manifests/` is tracked.

```python
def save_table(frame, path):
    """Save Parquet when available and fall back to CSV otherwise."""
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


neural_policy.save(ARTIFACT_DIR / "neural_lsm_policy.pt")
saved_paths = [
    save_table(contract_grid, ARTIFACT_DIR / "contract_test_grid.parquet"),
    save_table(neural_training_history, ARTIFACT_DIR / "training_history.parquet"),
    save_table(basis_results, ARTIFACT_DIR / "classical_basis_results.parquet"),
    save_table(path_convergence, ARTIFACT_DIR / "path_convergence.parquet"),
    save_table(
        exercise_date_convergence,
        ARTIFACT_DIR / "exercise_date_convergence.parquet",
    ),
    save_table(heldout_results, ARTIFACT_DIR / "heldout_pricing_results.parquet"),
    save_table(
        heldout_policy_metrics,
        ARTIFACT_DIR / "heldout_policy_metrics.parquet",
    ),
    save_table(ood_results, ARTIFACT_DIR / "ood_results.parquet"),
    save_table(robustness_results, ARTIFACT_DIR / "robustness_results.parquet"),
    save_table(runtime_records, ARTIFACT_DIR / "runtime_records.parquet"),
]

summary = {
    "profile": PROFILE,
    "selected_classical_basis": BEST_CLASSICAL_BASIS,
    "selected_classical_degree": int(BEST_CLASSICAL_DEGREE),
    "neural_policy_steps": len(neural_policy.steps),
    "neural_training_runtime_seconds": neural_training_runtime,
    "heldout_comparison": heldout_comparison.to_dict(orient="records"),
    "ood_comparison": ood_comparison.to_dict(orient="records"),
    "coverage": coverage_summary.to_dict(),
    "saved_tables": [str(path) for path in saved_paths],
}
with (ARTIFACT_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(f"Artifacts saved to: {ARTIFACT_DIR}")
```

## 16. Interpretation framework

The final interpretation should distinguish four cases:

1. **Neural LSM improves both pricing and policy accuracy.**  
   This supports the hypothesis that nonlinear continuation regression adds value beyond fixed basis functions.

2. **Neural LSM improves pricing but not policy agreement.**  
   Similar aggregate values may conceal different stopping decisions.

3. **Neural LSM improves policy agreement but not price accuracy.**  
   Monte Carlo noise or continuation-value bias may offset the policy improvement.

4. **Classical LSM remains stronger.**  
   The added neural complexity would not be justified for this controlled one-dimensional problem.

No conclusion should be chosen before the final-profile results are produced.

## 17. Limitations

- Both classical and neural LSM use discretized exercise opportunities rather than continuous exercise.
- The neural policy is trained under geometric Brownian motion with constant parameters.
- Monte Carlo standard errors capture valuation-path uncertainty but not the full uncertainty of policy estimation.
- The neural policy can extrapolate poorly outside its contract domain.
- One continuation network per exercise index increases training and storage requirements.
- CRR and LSM discretization errors are different, so neither comparison is perfectly model-free.
- A strong result in this one-dimensional setting does not imply superiority for higher-dimensional derivatives.

## 18. References used in this notebook

Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics, 7*(3), 229–263.

Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *Review of Financial Studies, 14*(1), 113–147.

Pu, V. R. H. (2021). *Pricing options using deep neural networks from a practical perspective: A comparative study of supervised and unsupervised learning* [Master's thesis, Imperial College London].

Zouaoui, H., & Naas, M.-N. (2023). Option pricing using deep learning approach based on LSTM-GRU neural networks. *Data Science in Finance and Economics, 3*(3), 267–284.

## 19. Development checkpoint

Step 7 is complete after the final profile has produced:

- validated GBM moments;
- classical basis, path-count, and exercise-date studies;
- a trained amortized neural stopping policy;
- held-out and out-of-domain pricing results;
- policy-agreement and stopping-time metrics;
- multi-seed robustness results;
- separated runtime costs.

Notebook 08 will consolidate all static and simulation-based models, decide the remaining hypotheses, and write the final project conclusions.
