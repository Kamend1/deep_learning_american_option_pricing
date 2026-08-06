# Model Architectures

## Purpose

The project compares simple and structured neural designs under a controlled five-feature input. The objective is to determine which formulation adds measurable value, not to maximize architectural complexity.

## Shared static input

Every static model receives:

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

The inputs are standardized with a scaler fitted on training observations.

## 1. Direct MLP

Files:

- `src/models/direct_pricer.py`
- Notebook 04

Configuration: `DirectMLPConfig`

Default architecture:

```text
Input(5)
→ Linear(128) → BatchNorm → SiLU
→ Linear(128) → BatchNorm → SiLU
→ Linear(64)  → SiLU
→ Linear(32)  → SiLU
→ Linear(1)   → Softplus
```

Target:

$$
y=\frac{V_A}{K}.
$$

The final `Softplus` guarantees non-negative output but does not guarantee that the price exceeds intrinsic or European value.

Role: strong conventional baseline and control for later target redesign.

## 2. Residual premium and financial-floor MLPs

Files:

- `src/models/premium_pricer.py`
- Notebook 05

Configuration: `PremiumMLPConfig`

The hidden backbone matches the direct MLP so comparisons isolate target and output construction rather than capacity.

### European residual

The early-exercise-premium target is:

$$
R_E=
\frac{V_A-V_E}{K}.
$$

Reconstruction:

$$
\widehat V_A/K
=
V_E/K+\widehat R_E.
$$

A linear output permits negative premiums. A `Softplus` output guarantees a non-negative premium.

### Financial-floor residual

The financial floor is:

$$
F=\max(V_E,I).
$$

The target is:

$$
R_F=
\frac{V_A-F}{K}.
$$

Reconstruction:

$$
\widehat V_A/K
=
F/K+\operatorname{Softplus}(\widehat R_F).
$$

This is the selected specialist pricing model.

### Candidate ablation

Notebook 05 evaluates:

1. unconstrained European residual;
2. non-negative European residual;
3. unweighted floor residual;
4. magnitude-weighted floor residual;
5. boundary-weighted floor residual.

Candidate selection uses validation evidence only.

## 3. Exercise-only classifier

Files:

- `src/models/multitask_pricer.py`
- Notebook 06

Configuration: `ExerciseClassifierConfig`

Architecture:

```text
Input(5)
→ Linear(128) → BatchNorm → SiLU
→ Linear(128) → BatchNorm → SiLU
→ Linear(64)  → SiLU
→ Linear(32)  → SiLU
→ Linear(1)   → logit
```

Target:

$$
Y_E=
\mathbb{1}[I\geq C].
$$

The output is a raw logit. Probability is obtained through the sigmoid function.

Class imbalance is addressed through a positive-class weight estimated on training observations. The operating threshold is selected on validation data by F1.

## 4. Notebook 06 multi-task model

Files:

- `src/models/multitask_pricer.py`
- `src/training/multitask_losses.py`
- `src/training/multitask_loops.py`

Configuration: `MultiTaskMLPConfig`

Architecture:

```text
Input(5)
→ Shared backbone: 128 → 128 → 64
   ├── Residual head: 32 → Softplus
   └── Exercise head: 32 → logit
```

Outputs:

- non-negative residual above the financial floor;
- exercise logit.

Loss:

$$
\mathcal L
=
\mathcal L_{\text{price}}
+
\lambda
\mathcal L_{\text{exercise}}.
$$

Notebook 06 tests predefined $\lambda$ values. Selection prioritizes boundary-band F1 and uses price error as a tie-breaker.

The experiment tests whether shared representation learning improves both the boundary decision and pricing. The final H4 result is not supported.

## 5. Neural Longstaff–Schwartz continuation networks

Files:

- `src/models/neural_longstaff_schwartz.py`
- `src/training/lsm_training.py`
- Notebook 07

Configuration: `ContinuationNetworkConfig`

Default architecture:

```text
Input(5)
→ Linear(64) → SiLU
→ Linear(64) → SiLU
→ Linear(32) → SiLU
→ Linear(1)  → Softplus
```

Input at exercise time $t$:

$$
\left[
\log(S_t/K),
T-t,
r,
q,
\sigma
\right].
$$

Output: normalized non-negative continuation value.

The policy stores a separate continuation network for each exercise index. Networks are trained backward in time and may warm-start from the next later exercise date.

A `NeuralLSMPolicy` serializes:

- number of exercise steps;
- feature names;
- one scaler and network state per time index;
- sample counts.

This model is path-based and does not share the static model leaderboard.

## 6. Integrated four-head static model

Files:

- `src/models/integrated_multihead_pricer.py`
- `src/training/multihead_losses.py`
- `src/training/multihead_loops.py`
- Notebook 08

Configuration: `IntegratedMultiHeadConfig`

Default scratch backbone:

```text
Input(5)
→ Shared backbone: 192 → 192 → 96
   ├── Floor-residual head: 48 → Softplus
   ├── Direct-price head: 48 → Softplus
   ├── Continuation head: 48 → Softplus
   └── Exercise head: 48 → logit
```

Dropout is 0.10 in the default scratch configuration.

A Step-6-compatible configuration uses:

- backbone: 128 → 128 → 64;
- no dropout;
- 32-unit heads.

This configuration allows compatible backbone weights from Notebook 06 to be copied for the warm-start experiment.

### Authoritative price

The residual head is authoritative:

$$
\widehat V_A^{constrained}
=
\max(V_E,I)
+
K\operatorname{Softplus}(\widehat R_F).
$$

The direct head is diagnostic. It reveals whether an unconstrained full-price output agrees with the protected reconstruction.

### Two exercise paths

The direct exercise probability is:

$$
\widehat p_E=\sigma(\ell_E).
$$

The continuation-implied probability is:

$$
\widehat p_C
=
\sigma
\left[
\kappa
\left(
\widetilde I-\widehat C
\right)
\right],
$$

where $\kappa$ is `decision_sharpness`.

The model records:

- price-head gap;
- exercise-probability gap;
- contradictory outputs;
- boundary-specific disagreement.

### Warm start

`copy_compatible_backbone_weights` copies only parameters whose names and shapes match. It returns copied and skipped keys for audit.

The warm-start model is a convergence and deployment experiment, not a clean capacity comparison with the larger scratch model.

## Initialization and input validation

Static MLPs use Kaiming-uniform initialization for linear weights and zero biases.

Model forward methods validate:

- two-dimensional input;
- expected feature count;
- finite values where applicable.

## Comparative summary

| Model | Inputs | Outputs | Hard guarantees | Primary role | Final status |
|---|---|---|---|---|---|
| Black–Scholes proxy | Contract parameters | European price | Analytical European value | Non-neural baseline | Not sufficient for American value |
| Direct MLP | Static state | Full normalized price | Non-negative | Control model | Supports H1 |
| Premium MLP | Static state | Residual above European | Optional non-negative premium | Target ablation | Improved over direct |
| Floor residual MLP | Static state | Residual above financial floor | Price above European and intrinsic | Best specialist price | Preferred static pricer |
| Exercise classifier | Static state | Exercise probability | None on price | Exercise-only use | Preferred narrow deployment |
| Notebook 06 multi-task | Static state | Price residual and exercise probability | Protected price | H4 experiment | No material joint-learning advantage |
| Neural LSM | Simulated paths and time | Continuation policy | Non-negative continuation | Path-based neural test | Not preferred |
| Integrated model | Static state | Protected price, direct price, continuation, exercise | Protected authoritative price | Combined deployment | Useful compromise |

## Why no LSTM or GRU for the static models

The static input is one contract-state vector, not a sequence. A recurrent architecture would impose sequence machinery without genuine temporal observations.

The project uses recurrence only in the economic algorithmic sense of backward stopping logic, not by forcing static variables into an artificial sequence.

## Related notebooks

- Notebook 04: direct baseline
- Notebook 05: residual and constrained models
- Notebook 06: classifier and multi-task model
- Notebook 07: neural continuation policy
- Notebook 08: integrated model
