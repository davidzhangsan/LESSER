"""LESSER: output-layer gradient features for post-training data selection."""
from .output_grad import output_layer_gradient, token_factors, token_factors_from_logits
from .projection import RL_PAPER, SFT_PAPER, TwoSidedConfig, TwoSidedProjection, l2_normalize
from .select import cosine_scores, round_robin, select

__all__ = [
    "output_layer_gradient", "token_factors", "token_factors_from_logits",
    "TwoSidedConfig", "TwoSidedProjection", "SFT_PAPER", "RL_PAPER", "l2_normalize",
    "cosine_scores", "round_robin", "select",
]
