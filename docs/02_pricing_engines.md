# Pricing Engines

## Purpose

The neural models reproduce a specified numerical pricing function. The credibility of the project therefore depends on the pricing layer being correct, explicit, and independently testable.

The implementation separates:

- readable scalar reference functions;
- accelerated batch label generation;
- independent QuantLib validation;
- path-based Longstaff–Schwartz experiments.

## Black–Scholes–Merton implementation

File: `src/pricing/black_scholes.py`

Public functions:

- `black_scholes_call_price`
- `black_scholes_put_price`

For a European put with continuous dividend yield,

$$
P_E = K e^{-rT}N(-d_2) -
S e^{-qT}N(-d_1),
$$

where

$$
d_1 =
\frac{
\ln(S/K)
+
\left(
r-q+\frac{1}{2}\sigma^2
\right)T
}{
\sigma\sqrt{T}
},
$$

and

$$
d_2=d_1-\sigma\sqrt{T}.
$$

### Input validation

The scalar API rejects:

- non-finite inputs;
- non-positive spot;
- non-positive strike;
- negative maturity;
- negative volatility.

Rates and dividend yields may be negative if finite, although the production domain uses non-negative values.

### Edge cases

At maturity, the function returns terminal intrinsic value.

For $T=0$:

$$
P_E = \max(K-S,0).
$$

For $\sigma=0$, the function uses the deterministic discounted payoff rather than dividing by zero in $d_1$.

### Role in the project

Black–Scholes is used as:

1. an analytical European benchmark;
2. a convergence target for European CRR;
3. a lower bound for the corresponding American option;
4. the base component in the early-exercise-premium model;
5. one input to the financial floor.

## Scalar Cox–Ross–Rubinstein engine

File: `src/pricing/binomial_tree.py`

Public interface:

- `CRRPriceResult`
- `crr_option_diagnostics`
- `crr_option_price`

The engine supports calls and puts, European and American exercise, and returns root-node diagnostics.

### Tree construction

For $N$ time steps,

$$
\Delta t = \frac{T}{N},
$$

$$
u=e^{\sigma\sqrt{\Delta t}},
$$

$$
d=\frac{1}{u},
$$

and the risk-neutral probability is

$$
p=
\frac{
e^{(r-q)\Delta t}-d
}{
u-d
}.
$$

The one-step discount factor is

$$
e^{-r\Delta t}.
$$

The implementation validates $p$ and raises an error when the parameter combination and step count produce a materially invalid risk-neutral probability.

### Backward induction

Terminal values are intrinsic payoffs. At each earlier node, continuation value is

$$
C_{i,j} =
e^{-r\Delta t}
\left[
pV_{i+1,j+1}
+
(1-p)V_{i+1,j}
\right].
$$

For a European option:

$$
V_{i,j}=C_{i,j}.
$$

For an American option:

$$
V_{i,j}=
\max(I_{i,j}, C_{i,j}).
$$

### Root diagnostics

`CRRPriceResult` stores:

- final root price;
- root intrinsic value;
- root continuation value before the maximum operator;
- `exercise_now`.

The exercise label is based on:

$$
I_0 \geq C_0 - 10^{-12}.
$$

This diagnostic is required for supervised exercise classification and boundary-distance construction.

### Memory design

The scalar tree uses one-dimensional NumPy arrays during backward induction. Memory complexity is $O(N)$ rather than storing the full triangular tree.

### Deterministic case

For $\sigma=0$, the engine performs deterministic backward induction along the risk-neutral path. This avoids unstable tree factors and preserves the American exercise comparison.

## Numba production CRR engine

File: `src/data/production_generation.py`

The production generator implements Numba-compatible versions of Black–Scholes and American put CRR. The core single-contract function is `_crr_american_put_one`; `price_american_put_batch` applies the calculation in parallel across arrays.

The production engine returns:

- European price;
- raw American price;
- intrinsic value;
- root continuation value;
- exercise decision.

### No-arbitrage floor repair

A finite tree may oscillate slightly below a theoretical lower bound. The generator retains the raw value and applies:

$$
V_A =
\max(
V_A^{raw},
V_E,
I
).
$$

The `pricing_floor_adjustment` column is calculated as:

$$
A_{\mathrm{floor}} =
V_{A} - V_{A}^{\mathrm{raw}}.
$$

where $A_{\mathrm{floor}}$ denotes the adjustment applied to the raw CRR price.

This repair is transparent. A high adjustment rate or large maximum adjustment would signal that the selected tree resolution is unsuitable.

## Production tree resolution

The project uses 250 CRR steps for production labels.

The resolution is selected through a convergence-runtime study rather than by convention. Notebook 02 compares:

- European CRR with Black–Scholes;
- American CRR with a finer reference tree;
- runtime per option.

The selected production rule requires the tested error tolerances and a conservative minimum resolution.

## QuantLib validation

File: `src/pricing/validation.py` and the final business-case benchmark modules.

QuantLib provides an independently implemented comparison using:

- binomial CRR;
- finite-difference Black–Scholes engines.

QuantLib is not the label generator. Its purpose is to identify major implementation discrepancies and to provide a standard-library runtime benchmark.

Because QuantLib is optional in some environments, its checks are explicit and cannot silently replace missing project evidence.

## Risk-neutral path simulation

File: `src/pricing/simulation.py`

Under geometric Brownian motion,

$$
S_{t+\Delta t} =
S_t
\exp
\left[
\left(
r-q-\frac{1}{2}\sigma^2
\right)\Delta t
+
\sigma\sqrt{\Delta t}Z
\right].
$$

The simulator uses fixed seeds and supports antithetic variates. Simulation validation compares empirical terminal moments with their theoretical risk-neutral counterparts before any stopping policy is fitted.

## Classical Longstaff–Schwartz

File: `src/pricing/longstaff_schwartz.py`

At each exercise date, the classical method:

1. identifies in-the-money paths;
2. regresses discounted realized future cash flows on basis functions of the current state;
3. estimates continuation value;
4. exercises where intrinsic value is at least continuation value;
5. moves backward to the preceding exercise date.

The experiment tests polynomial and weighted Laguerre specifications.

### Separation of fitting and valuation paths

The stopping policy is fitted on one path set and valued on an independent path set. Reusing training paths for valuation would bias the estimated option value upward.

### Monte Carlo uncertainty

Path-based values are reported with:

- standard errors;
- confidence intervals;
- interval coverage against CRR;
- multi-seed robustness.

## Neural Longstaff–Schwartz

Files:

- `src/models/neural_longstaff_schwartz.py`
- `src/training/lsm_training.py`

The neural variant replaces the classical continuation regression with time-indexed MLPs.

At each exercise index, the input is:

$$
\left[
\log(S_t/K),
T-t,
r,
q,
\sigma
\right].
$$

The target is normalized discounted future cash flow. Each network uses a non-negative `Softplus` output.

A `NeuralLSMPolicy` stores one `NeuralContinuationStep` per exercise index, including:

- feature standardizer;
- network configuration;
- state dictionary;
- training and validation sample counts.

The policy remains separate from static models because it consumes simulated paths and applies backward stopping logic.

## Validation and failure modes

The pricing layer explicitly checks:

- non-finite or invalid inputs;
- invalid CRR probabilities;
- European CRR convergence;
- American value above intrinsic;
- American value above European value;
- deterministic edge cases;
- simulated terminal moments;
- path-set independence;
- finite and non-negative outputs.

If the pricing engine is wrong, the neural network will learn the error accurately. Pricing validation is therefore an upstream gate, not a final diagnostic.

## Related notebooks

- Notebook 01: financial equations and method selection
- Notebook 02: convergence, validation, and pilot generation
- Notebook 07: classical and neural Longstaff–Schwartz
- Notebook 09: runtime and final comparison
