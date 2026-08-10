"""Command-line entry points for training and generation.

These back both the ``stylegan2-train`` / ``stylegan2-generate`` console scripts
(see ``pyproject.toml``) and the thin wrappers under ``scripts/``.
"""

import argparse

from .config import InferenceConfig, TrainConfig
from .device import gpu_report
from .inference import generate_samples
from .training import train_stylegan2_ada


def train() -> None:
    """Parse training arguments and run the training loop."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train StyleGAN2-ADA for synthetic image generation."
    )

    # Paths
    parser.add_argument("--data-dir", default=defaults.data_dir, required=not defaults.data_dir,
                        help="ImageFolder dataset root (data_dir/class_x/*.png).")
    parser.add_argument("--output-dir", default=defaults.output_dir,
                        help="Directory for checkpoints and sample grids.")

    # Architecture
    parser.add_argument("--img-size", type=int, default=defaults.img_size)
    parser.add_argument("--z-dim", type=int, default=defaults.z_dim)
    parser.add_argument("--w-dim", type=int, default=defaults.w_dim)

    # Training
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr-g", type=float, default=defaults.lr_g)
    parser.add_argument("--lr-d", type=float, default=defaults.lr_d)

    # Regularization
    parser.add_argument("--r1-gamma", type=float, default=defaults.r1_gamma)
    parser.add_argument("--r1-interval", type=int, default=defaults.r1_interval)
    parser.add_argument("--pl-weight", type=float, default=defaults.pl_weight)
    parser.add_argument("--pl-interval", type=int, default=defaults.pl_interval)

    # ADA
    parser.add_argument("--ada-target", type=float, default=defaults.ada_target)
    parser.add_argument("--ada-speed", type=int, default=defaults.ada_speed)

    # Misc
    parser.add_argument("--style-mixing-prob", type=float, default=defaults.style_mixing_prob)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--sample-interval", type=int, default=defaults.sample_interval)
    parser.add_argument("--save-interval", type=int, default=defaults.save_interval)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        default=defaults.resume,
                        help="Ignore any existing checkpoint and train from scratch.")

    cfg = TrainConfig(**vars(parser.parse_args()))
    gpu_report()

    train_stylegan2_ada(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        img_size=cfg.img_size,
        z_dim=cfg.z_dim,
        w_dim=cfg.w_dim,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        r1_gamma=cfg.r1_gamma,
        r1_interval=cfg.r1_interval,
        pl_weight=cfg.pl_weight,
        pl_interval=cfg.pl_interval,
        ada_target=cfg.ada_target,
        ada_speed=cfg.ada_speed,
        style_mixing_prob=cfg.style_mixing_prob,
        ema_decay=cfg.ema_decay,
        sample_interval=cfg.sample_interval,
        save_interval=cfg.save_interval,
        resume=cfg.resume,
    )


def generate() -> None:
    """Parse generation arguments and write synthetic images to disk."""
    defaults = InferenceConfig()
    parser = argparse.ArgumentParser(
        description="Generate synthetic images from a StyleGAN2-ADA checkpoint."
    )
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path,
                        help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--output-dir", default=defaults.output_dir,
                        help="Directory to write generated PNGs.")
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--class-idx", type=int, default=defaults.class_idx)
    parser.add_argument("--truncation-psi", type=float, default=defaults.truncation_psi,
                        help="0.5-0.7 for quality, 1.0 for maximum diversity.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--no-preview", dest="show_preview", action="store_false",
                        default=True, help="Skip the matplotlib preview grid.")

    args = parser.parse_args()
    gpu_report()

    generate_samples(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        class_idx=args.class_idx,
        truncation_psi=args.truncation_psi,
        batch_size=args.batch_size,
        seed=args.seed,
        show_preview=args.show_preview,
    )
