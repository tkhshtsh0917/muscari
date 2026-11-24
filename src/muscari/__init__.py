"""Sparse autoencoders for text encoders."""

from .activations import EncodedBatch, EncodedText, EncoderActivationSource, shuffled_batches
from .analysis import evaluate_texts, latent_statistics
from .data import TextColumn, is_holdout, iter_texts, repeat_texts
from .devices import resolve_device
from .export import ExportedSAE, evaluate_sae, load_sae, save_saelens_sae, save_topk_sae
from .mlm import mlm_loss_recovered
from .probe import RerankingExample, probe_reranking
from .sae import TopKSAE, geometric_median, normalized_mse
from .trainers import (
    SAELensTrainerConfig,
    TopKTrainer,
    TopKTrainerConfig,
    train_with_saelens,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "EncodedBatch",
    "EncodedText",
    "EncoderActivationSource",
    "ExportedSAE",
    "RerankingExample",
    "SAELensTrainerConfig",
    "TextColumn",
    "TopKSAE",
    "TopKTrainer",
    "TopKTrainerConfig",
    "__version__",
    "evaluate_sae",
    "evaluate_texts",
    "geometric_median",
    "is_holdout",
    "iter_texts",
    "latent_statistics",
    "load_sae",
    "mlm_loss_recovered",
    "normalized_mse",
    "probe_reranking",
    "repeat_texts",
    "resolve_device",
    "save_saelens_sae",
    "save_topk_sae",
    "shuffled_batches",
    "train_with_saelens",
]
