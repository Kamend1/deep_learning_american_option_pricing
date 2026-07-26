"""Classical pricing engines and validation utilities."""

from .black_scholes import black_scholes_call_price, black_scholes_put_price
from .binomial_tree import CRRPriceResult, crr_option_diagnostics, crr_option_price

__all__ = [
    "black_scholes_call_price",
    "black_scholes_put_price",
    "CRRPriceResult",
    "crr_option_diagnostics",
    "crr_option_price",
]

from .longstaff_schwartz import (
    BasisName,
    ContinuationRegression,
    LSMExperimentResult,
    LSMPriceResult,
    LongstaffSchwartzPolicy,
    evaluate_longstaff_schwartz_policy,
    fit_longstaff_schwartz_policy,
    longstaff_schwartz_put_price,
)
from .simulation import (
    GBMContract,
    GBMMomentValidation,
    generate_antithetic_normals,
    sample_contracts_latin_hypercube,
    simulate_contract_paths,
    simulate_gbm_paths,
    theoretical_terminal_moments,
    validate_simulated_moments,
)

__all__ += [
    "BasisName",
    "ContinuationRegression",
    "GBMContract",
    "GBMMomentValidation",
    "LSMExperimentResult",
    "LSMPriceResult",
    "LongstaffSchwartzPolicy",
    "evaluate_longstaff_schwartz_policy",
    "fit_longstaff_schwartz_policy",
    "generate_antithetic_normals",
    "longstaff_schwartz_put_price",
    "sample_contracts_latin_hypercube",
    "simulate_contract_paths",
    "simulate_gbm_paths",
    "theoretical_terminal_moments",
    "validate_simulated_moments",
]

