"""Deep-learning model architectures."""

from src.models.direct_pricer import DirectAmericanPutMLP, DirectMLPConfig
from src.models.multitask_pricer import (
    ExerciseClassifierConfig,
    ExerciseClassifierMLP,
    MultiTaskAmericanPutMLP,
    MultiTaskMLPConfig,
)
from src.models.premium_pricer import (
    PremiumAmericanPutMLP,
    PremiumMLPConfig,
    calculate_normalized_residual_target,
    reconstruct_normalized_american_price,
)

__all__ = [
    "DirectAmericanPutMLP",
    "DirectMLPConfig",
    "ExerciseClassifierConfig",
    "ExerciseClassifierMLP",
    "MultiTaskAmericanPutMLP",
    "MultiTaskMLPConfig",
    "PremiumAmericanPutMLP",
    "PremiumMLPConfig",
    "calculate_normalized_residual_target",
    "reconstruct_normalized_american_price",
]

from src.models.neural_longstaff_schwartz import (
    ContinuationNetworkConfig,
    ContinuationValueNetwork,
    ContractPathBatch,
    FeatureStandardizer,
    NeuralContinuationStep,
    NeuralLSMPolicy,
    build_continuation_features,
)

__all__ += [
    "ContinuationNetworkConfig",
    "ContinuationValueNetwork",
    "ContractPathBatch",
    "FeatureStandardizer",
    "NeuralContinuationStep",
    "NeuralLSMPolicy",
    "build_continuation_features",
]


from src.models.integrated_multihead_pricer import (
    IntegratedAmericanPutMultiHeadMLP,
    IntegratedMultiHeadConfig,
    copy_compatible_backbone_weights,
    reconstruct_integrated_outputs,
)

__all__ += [
    "IntegratedAmericanPutMultiHeadMLP",
    "IntegratedMultiHeadConfig",
    "copy_compatible_backbone_weights",
    "reconstruct_integrated_outputs",
]
