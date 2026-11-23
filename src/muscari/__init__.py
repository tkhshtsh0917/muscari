"""Sparse autoencoders for text encoders."""

from .activations import EncoderActivationSource, shuffled_batches
from .data import TextColumn, is_holdout, iter_texts, repeat_texts
from .devices import resolve_device
from .export import ExportedSAE, evaluate_sae, load_sae, save_saelens_sae, save_topk_sae
from .sae import TopKSAE, geometric_median, normalized_mse
from .trainers import (
    SAELensTrainerConfig,
    TopKTrainer,
    TopKTrainerConfig,
    train_with_saelens,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "EncoderActivationSource",
    "ExportedSAE",
    "SAELensTrainerConfig",
    "TextColumn",
    "TopKSAE",
    "TopKTrainer",
    "TopKTrainerConfig",
    "__version__",
    "evaluate_sae",
    "geometric_median",
    "is_holdout",
    "iter_texts",
    "load_sae",
    "normalized_mse",
    "repeat_texts",
    "resolve_device",
    "save_saelens_sae",
    "save_topk_sae",
    "shuffled_batches",
    "train_with_saelens",
]
