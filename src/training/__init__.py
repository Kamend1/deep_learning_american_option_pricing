"""Training, checkpointing, and loss utilities."""

from src.training.losses import (
    FittedPremiumWeighting,
    PremiumWeightConfig,
    WeightedSmoothL1Loss,
    apply_premium_weighting,
    fit_premium_weighting,
)
from src.training.loops import (
    TrainingConfig,
    fit_regression_model,
    predict_regression_model,
    set_global_seed,
)
from src.training.multitask_losses import (
    MultiTaskLossConfig,
    MultiTaskPricingLoss,
    calculate_positive_class_weight,
)
from src.training.multitask_loops import (
    MultiTaskTrainingConfig,
    fit_exercise_classifier,
    fit_multitask_model,
    predict_exercise_classifier,
    predict_multitask_model,
    run_classifier_epoch,
    run_multitask_epoch,
    set_multitask_seed,
)

__all__ = [
    "FittedPremiumWeighting",
    "MultiTaskLossConfig",
    "MultiTaskPricingLoss",
    "MultiTaskTrainingConfig",
    "PremiumWeightConfig",
    "TrainingConfig",
    "WeightedSmoothL1Loss",
    "apply_premium_weighting",
    "calculate_positive_class_weight",
    "fit_exercise_classifier",
    "fit_multitask_model",
    "fit_premium_weighting",
    "fit_regression_model",
    "predict_exercise_classifier",
    "predict_multitask_model",
    "predict_regression_model",
    "run_classifier_epoch",
    "run_multitask_epoch",
    "set_global_seed",
    "set_multitask_seed",
]

from src.training.lsm_training import (
    NeuralLSMTrainingConfig,
    evaluate_neural_lsm_policy,
    fit_neural_lsm_policy,
    load_neural_lsm_policy,
    save_neural_lsm_policy,
    set_lsm_seed,
    validate_contract_separation,
)

__all__ += [
    "NeuralLSMTrainingConfig",
    "evaluate_neural_lsm_policy",
    "fit_neural_lsm_policy",
    "load_neural_lsm_policy",
    "save_neural_lsm_policy",
    "set_lsm_seed",
    "validate_contract_separation",
]

