"""Configuration dataclasses for training and inference.

Defaults mirror the values used in the original notebook, tuned for small
datasets on an 8 GB GPU. Override any field from the CLI scripts.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for training StyleGAN2-ADA."""

    # Paths
    data_dir: str = ""
    output_dir: str = "stylegan2_checkpoints"

    # Architecture
    img_size: int = 256
    z_dim: int = 512
    w_dim: int = 512

    # Training
    batch_size: int = 4          # 256x256 within an 8 GB VRAM budget
    epochs: int = 1000
    lr_g: float = 0.0002         # lower than the paper; better on small data
    lr_d: float = 0.0002

    # Regularization
    r1_gamma: float = 2.0        # D gradient penalty; raised to slow D overfitting on tiny data
    r1_interval: int = 16        # lazy regularization frequency
    pl_weight: float = 0.0       # disabled until G produces decent outputs
    pl_interval: int = 4

    # ADA
    ada_target: float = 0.6      # paper value: a lower target forces MORE augmentation (p saturates at 1.0 and leaks into G)
    ada_speed: int = 100         # faster adaptation for small datasets

    # Misc
    style_mixing_prob: float = 0.9   # mix two latents across layers for more variety
    ema_decay: float = 0.999         # small batches need faster EMA updates
    sample_interval: int = 10
    save_interval: int = 200
    resume: bool = True


@dataclass
class InferenceConfig:
    """Parameters for generating synthetic images from a checkpoint."""

    checkpoint_path: str = "stylegan2_checkpoints/latest.pt"
    output_dir: str = "generated_samples"
    num_samples: int = 50
    class_idx: int = 0
    truncation_psi: float = 1.0  # 1.0 = max diversity; lower (0.5-0.7) trades variety for quality
    batch_size: int = 8
    seed: int = 42
