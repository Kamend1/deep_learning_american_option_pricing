# Literature Matrix

## Purpose

This file records exactly what the project takes from each core source, where the
idea appears in the repository, and what evidence must eventually be compared with
the source.

The current canonical set contains six supplied deep-learning studies and four
foundational option-pricing papers.

| Citation key | Source | Main contribution used here | Project decision influenced | Main notebook locations | Final comparison required |
|---|---|---|---|---|---|
| `black_scholes_1973` | Black and Scholes (1973) | Closed-form European option valuation under no-arbitrage assumptions. | Use the European put price as analytical benchmark and as the known component of the early-exercise-premium model. | 01, 02, 04, 05, 09 | Compare the European Black–Scholes proxy with every American-pricing model. |
| `merton_1973` | Merton (1973) | Extends rational option-pricing theory and formalizes dividend-sensitive valuation relationships. | Include continuous dividend yield and state the theoretical lower-bound relationships carefully. | 01, 02, 09 | Discuss limitations of constant rates, constant volatility, and continuous dividends. |
| `cox_ross_rubinstein_1979` | Cox, Ross, and Rubinstein (1979) | Transparent discrete-time tree with backward induction and support for early exercise. | Select the Cox–Ross–Rubinstein tree as the main American put label generator and benchmark. | 01, 02, 03, 04–09 | Report convergence, runtime, label quality, and numerical approximation limitations. |
| `longstaff_schwartz_2001` | Longstaff and Schwartz (2001) | Estimates continuation values by least-squares regression on simulated paths. | Implement classical Least-Squares Monte Carlo and replace polynomial continuation regression with neural networks in a separate experiment. | 01, 07, 09 | Compare classical and neural continuation policies, uncertainty, accuracy, and runtime. |
| `ke_yang_2019` | Ke and Yang (2019) | Compares multilayer perceptron and long short-term memory models; multi-task bid/ask prediction performs strongly. | Use a feed-forward multilayer perceptron as the static baseline and treat architecture choice and target design separately. | 01, 04, 06, 08, 09 | Compare whether multi-task learning improves exercise-boundary performance in this project. |
| `elbayed_qadi_el_idrissi_2025` | Elbayed and Qadi El Idrissi (2025) | Dense neural network approximates synthetic European put prices generated from Black–Scholes. | Treat direct neural function approximation as a baseline, not as the final contribution. | 01, 04, 09 | Determine whether American-option residual learning provides value beyond reproducing a known smooth function. |
| `pimentel_et_al_2026` | Pimentel et al. (2026) | Long short-term memory models use genuinely sequential option data and outperform several benchmarks; segmented errors and explainability matter. | Do not introduce recurrent models for static inputs; require segmented analysis by moneyness and maturity. | 01, 03, 04, 09 | Explain why static multilayer perceptrons are appropriate here and compare segmented error patterns. |
| `zouaoui_naas_2023` | Zouaoui and Naas (2023) | Reviews and applies long short-term memory and gated recurrent unit models while emphasizing data, overfitting, validation, interpretability, and compute constraints. | Define abbreviations clearly, justify model choice, and document compute and validation requirements. | 01, 04–09 | Compare practical computational requirements and discuss overfitting and interpretability. |
| `pu_2021` | Pu (2021) | Compares supervised neural pricing with neural partial-differential-equation approaches and highlights weaker extrapolation outside the training domain. | Include mandatory out-of-domain test regimes and separate supervised surrogate learning from equation-based methods. | 01, 03, 04–09 | Quantify deterioration outside the training domain and avoid broad extrapolation claims. |
| `ding_lu_cheung_2025` | Ding, Lu, and Cheung (2025) | Compresses market-implied volatility surfaces with a variational autoencoder and prices QuantLib-generated American and Asian options through a multilayer perceptron. | Keep constant volatility in the core project while identifying volatility-surface representations as a credible future extension. | 01, 02, 08, 09 | Compare inference-speed motivation and discuss richer market-state inputs as future research. |

## Method-to-source traceability

### Analytical and numerical foundation

- European analytical benchmark: `black_scholes_1973`, `merton_1973`
- American binomial-tree benchmark: `cox_ross_rubinstein_1979`
- Simulation-based continuation regression: `longstaff_schwartz_2001`

### Neural architecture

- Direct multilayer perceptron baseline: `ke_yang_2019`, `elbayed_qadi_el_idrissi_2025`
- Sequential models used only for genuine sequences: `ke_yang_2019`, `pimentel_et_al_2026`, `zouaoui_naas_2023`
- Multi-task learning motivation: `ke_yang_2019`
- Neural continuation-value learning: `longstaff_schwartz_2001`, `pu_2021`
- Richer volatility-state representation: `ding_lu_cheung_2025`

### Evaluation design

- In-domain pricing accuracy: all neural-pricing studies
- Moneyness and maturity segmentation: `pimentel_et_al_2026`
- Out-of-domain deterioration: `pu_2021`
- Computational burden and validation: `zouaoui_naas_2023`
- Static surrogate speed motivation: `ding_lu_cheung_2025`
- Financial consistency and early exercise: `cox_ross_rubinstein_1979`, `longstaff_schwartz_2001`

## Update protocol

After each notebook is executed:

1. Add the actual finding in the relevant row.
2. State whether the result agrees, partly agrees, or disagrees with the source.
3. Explain whether differences arise from data, option type, target design, architecture, or evaluation domain.
4. Add the exact notebook section and figure or table number.
5. Do not claim support from a paper unless the cited paper actually addresses that point.
