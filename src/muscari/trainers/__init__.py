"""Two ways to train an SAE on the same activation stream."""

from .saelens import SAELensTrainerConfig, train_with_saelens
from .shizuku import ShizukuTrainer, ShizukuTrainerConfig, warmup_cosine

__all__ = [
    "SAELensTrainerConfig",
    "ShizukuTrainer",
    "ShizukuTrainerConfig",
    "train_with_saelens",
    "warmup_cosine",
]
