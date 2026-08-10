"""StyleGAN2-ADA for synthetic defect image generation.

A modular PyTorch implementation of StyleGAN2 with Adaptive Discriminator
Augmentation (ADA), designed for training on small datasets.
"""

from .config import InferenceConfig, TrainConfig
from .inference import generate_samples
from .networks import (
    DiscriminatorStyleGAN2,
    GeneratorStyleGAN2,
    MappingNetworkADA,
)
from .training import train_stylegan2_ada

__all__ = [
    "TrainConfig",
    "InferenceConfig",
    "GeneratorStyleGAN2",
    "DiscriminatorStyleGAN2",
    "MappingNetworkADA",
    "train_stylegan2_ada",
    "generate_samples",
]

__version__ = "1.0.0"
