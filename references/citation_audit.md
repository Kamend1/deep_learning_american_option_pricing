# Citation Audit

## Status legend

- **Verified** — bibliographic details checked and citation is ready for use.
- **Used** — substantively discussed in at least one notebook.
- **Pending result comparison** — methodology is cited, but the final empirical comparison requires completed model outputs.
- **Pending placement review** — the final communication pass must confirm the exact section citation.

## Core-source audit

| Citation key | Bibliography verified | Used substantively | Current role | Required final action |
|---|---:|---:|---|---|
| `black_scholes_1973` | Yes | Yes | European analytical benchmark and theoretical foundation. | Cite every first presentation of the formula and discuss benchmark limitations. |
| `merton_1973` | Yes | Yes | Rational pricing restrictions and dividend-sensitive theory. | Confirm citations where dividend yield and theoretical bounds are introduced. |
| `cox_ross_rubinstein_1979` | Yes | Yes | Primary American put numerical benchmark and label generator. | Cite the algorithm, convergence discussion, and early-exercise logic. |
| `longstaff_schwartz_2001` | Yes | Yes | Classical simulation-based continuation-value benchmark. | Cite the backward-regression algorithm and compare with neural continuation learning. |
| `ke_yang_2019` | Yes | Yes | Multilayer perceptron, long short-term memory, and multi-task target-design motivation. | Define all abbreviations on first use and avoid implying the paper studies the same American-put setup. |
| `elbayed_qadi_el_idrissi_2025` | Yes | Yes | Synthetic direct-pricing neural baseline. | Clarify that approximating Black–Scholes is a baseline with limited computational justification. |
| `pimentel_et_al_2026` | Yes | Yes | Sequential option-pricing evidence and segmented evaluation. | Compare actual moneyness and maturity error patterns after production runs. |
| `zouaoui_naas_2023` | Yes | Yes | Recurrent architectures and practical validation risks. | Use in compute, overfitting, interpretability, and validation discussion. |
| `pu_2021` | Yes | Yes | Supervised versus equation-based neural pricing and extrapolation risk. | Compare out-of-domain deterioration after all model runs. |
| `ding_lu_cheung_2025` | Yes | Yes | Volatility-surface compression and fast American-option surrogate pricing. | Discuss as richer-state future extension and compare speed motivation. |

## Foundational citation requirements

The following claims require a direct citation:

- Black–Scholes–Merton formulas and assumptions.
- Cox–Ross–Rubinstein tree construction and backward induction.
- Longstaff–Schwartz least-squares continuation regression.
- Published neural architectures or empirical findings.
- Claims about one model outperforming another.
- Claims about extrapolation, overfitting, interpretability, or computational advantage.

## Claims that normally do not require external citation

- Project-specific file paths.
- Project-specific sample sizes and random seeds.
- Results calculated directly in the notebooks.
- Implementation decisions explicitly presented as choices made in this project.
- Conclusions derived transparently from the project outputs.

## Abbreviation audit

Every notebook must define abbreviations at first use.

| Abbreviation | Required first-use wording |
|---|---|
| MLP | multilayer perceptron (MLP) |
| LSTM | long short-term memory network (LSTM) |
| GRU | gated recurrent unit (GRU) |
| PDE | partial differential equation (PDE) |
| CRR | Cox–Ross–Rubinstein (CRR) |
| LSM | Least-Squares Monte Carlo (LSM) |
| DNN | deep neural network (DNN) |
| OOD | out-of-domain (OOD) |
| MAE | mean absolute error (MAE) |
| RMSE | root mean squared error (RMSE) |
| VAE | variational autoencoder (VAE) |
| SHAP | SHapley Additive exPlanations (SHAP) |

## Notebook audit checklist

For every notebook:

- [ ] Every external factual claim has a citation.
- [ ] Every abbreviation is defined at first use.
- [ ] Every cited paper is represented accurately.
- [ ] Methodological influence is distinguished from empirical agreement.
- [ ] Section conclusions cite relevant literature where comparison is made.
- [ ] The notebook conclusion states agreement or disagreement with prior work.
- [ ] The notebook-to-notebook handoff identifies the evidence passed forward.
- [ ] The Markdown twin matches the executed notebook.
- [ ] The bibliography keys match `references/references.bib`.

## Final-project audit

Before submission:

- [ ] All ten core sources are cited substantively.
- [ ] No source appears only in the bibliography without discussion.
- [ ] No paper is credited with an idea it does not contain.
- [ ] H1–H6 conclusions refer to actual tables and metrics.
- [ ] Static neural surrogates and path-based neural Least-Squares Monte Carlo are discussed separately.
- [ ] In-domain performance and out-of-domain robustness are not conflated.
- [ ] Computational speed separates up-front generation and training from marginal inference.
- [ ] The final conclusion distinguishes what the evidence demonstrates, suggests, and does not establish.
