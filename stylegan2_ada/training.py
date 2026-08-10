"""StyleGAN2-ADA training loop.

Each step: train D (reals + fakes, softplus loss), apply the lazy R1 penalty,
let ADA adjust the augmentation probability, train G to fool D, apply lazy path
length regularization, then update the EMA copy of G.
"""

import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .augment import AdaptiveAugment, AugmentPipe
from .data import build_dataloader
from .device import DEVICE
from .networks import DiscriminatorStyleGAN2, GeneratorStyleGAN2
from .regularization import path_length_regularization


def train_stylegan2_ada(
    data_dir: str,
    output_dir: str = "stylegan2_checkpoints",
    # Architecture
    img_size: int = 256, z_dim: int = 512, w_dim: int = 512,
    # Training
    batch_size: int = 8, epochs: int = 1200,
    lr_g: float = 0.0025, lr_d: float = 0.0025,
    # Regularization
    r1_gamma: float = 10.0, r1_interval: int = 16,
    pl_weight: float = 2.0, pl_interval: int = 4,
    # ADA
    ada_target: float = 0.6, ada_speed: int = 500,
    # Misc
    style_mixing_prob: float = 0.9, ema_decay: float = 0.9999,
    sample_interval: int = 10, save_interval: int = 50,
    resume: bool = True,
):
    """Train a StyleGAN2-ADA model.

    Returns ``(generator_ema, discriminator, class_names, history)``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_dir = output_path / "samples"
    samples_dir.mkdir(exist_ok=True)

    # ── Data ──
    dataloader, class_names, num_classes = build_dataloader(
        data_dir, img_size, batch_size
    )

    # ── Models ──
    generator = GeneratorStyleGAN2(z_dim, w_dim, num_classes, img_size).to(DEVICE)
    discriminator = DiscriminatorStyleGAN2(img_size, 3, num_classes).to(DEVICE)

    # EMA copy of G — used for sampling (smoother outputs)
    generator_ema = deepcopy(generator).eval()
    for p in generator_ema.parameters():
        p.requires_grad = False

    # betas=(0, 0.99): no first-order momentum, standard for StyleGAN2
    opt_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99))

    # ── ADA ──
    ada = AdaptiveAugment(target_rt=ada_target, speed=ada_speed)
    augment = AugmentPipe()

    # ── State ──
    pl_mean = torch.zeros(1, device=DEVICE)
    start_epoch = 0
    global_step = 0

    history = {
        "G_loss": [], "D_loss": [], "D_real": [], "D_fake": [],
        "R1": [], "PL": [], "ADA_p": [],
    }

    # ── Resume from checkpoint ──
    latest_ckpt = output_path / "latest.pt"
    checkpoint_loaded = False

    if latest_ckpt.exists():
        print(f"\nCheckpoint found: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        saved_epoch = ckpt.get("epoch", 0)
        saved_config = ckpt.get("config", {})

        config_ok = (
            saved_config.get("z_dim", z_dim) == z_dim and
            saved_config.get("w_dim", w_dim) == w_dim and
            saved_config.get("img_size", img_size) == img_size
        )

        if not config_ok:
            print("  Incompatible config, starting fresh...")

        elif saved_epoch >= epochs:
            print(f"  Already at {saved_epoch} epochs (>= {epochs}). Nothing to train.")
            if resume:
                generator.load_state_dict(ckpt["generator"])
                generator_ema.load_state_dict(ckpt["generator_ema"])
                discriminator.load_state_dict(ckpt["discriminator"])
                return generator_ema, discriminator, \
                    ckpt.get("class_names", class_names), \
                    ckpt.get("history", history)

        elif resume:
            print(f"  Resuming: {saved_epoch} -> {epochs} epochs")
            try:
                generator.load_state_dict(ckpt["generator"])
                discriminator.load_state_dict(ckpt["discriminator"])
                generator_ema.load_state_dict(ckpt["generator_ema"])
                opt_G.load_state_dict(ckpt["opt_G"])
                opt_D.load_state_dict(ckpt["opt_D"])
                start_epoch = saved_epoch
                global_step = ckpt.get("global_step", 0)
                pl_mean = ckpt.get("pl_mean", pl_mean)
                # Do NOT restore the saved ada.p: the previous run saturated it
                # at 1.0 (max augmentation, which leaks into G). Reset to 0 so
                # the milder augmentation pipeline re-adapts p from scratch.
                ada.p = 0.0
                history = ckpt.get("history", history)
                checkpoint_loaded = True
                print("  Checkpoint loaded OK")
            except Exception as e:
                print(f"  Error loading checkpoint: {e}")
                print("  Starting fresh...")
                start_epoch = 0
                global_step = 0
    else:
        print(f"\nNo checkpoint in {output_path}, training from scratch")

    if start_epoch >= epochs:
        return generator_ema, discriminator, class_names, history

    # ── Fixed noise for visual monitoring ──
    n_preview = min(num_classes * 4, 16)
    fixed_z = torch.randn(n_preview, z_dim, device=DEVICE)
    fixed_labels = torch.tensor(
        [i % num_classes for i in range(n_preview)], device=DEVICE
    )

    print(f"\n{'=' * 60}")
    print(f"STYLEGAN2-ADA — {img_size}x{img_size}")
    print(f"z/w dim: {z_dim}/{w_dim} | Classes: {num_classes}")
    print(f"G: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"D: {sum(p.numel() for p in discriminator.parameters()):,} params")
    print(f"Epochs: {start_epoch} -> {epochs} ({epochs - start_epoch} remaining)")
    if checkpoint_loaded:
        print(f"ADA p = {ada.p:.3f}")
    print(f"{'=' * 60}\n")

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            curr_batch = real_imgs.size(0)
            global_step += 1

            # ─── DISCRIMINATOR ───
            opt_D.zero_grad()

            real_aug = augment(real_imgs, ada.p)
            real_aug.requires_grad_(True)  # needed for R1 grad

            real_pred = discriminator(real_aug, labels)
            d_loss_real = F.softplus(-real_pred).mean()  # push D(real) up

            # R1: penalize large gradients on real images (lazy reg)
            r1_loss = torch.tensor(0.0, device=DEVICE)
            if global_step % r1_interval == 0:
                grad_real = torch.autograd.grad(
                    outputs=real_pred.sum(),
                    inputs=real_aug,
                    create_graph=True,
                )[0]
                r1_loss = grad_real.pow(2).view(curr_batch, -1).sum(1).mean()
                r1_loss = r1_loss * r1_gamma / 2

            # Fakes (no G gradients here)
            z = torch.randn(curr_batch, z_dim, device=DEVICE)
            with torch.no_grad():
                fake_imgs = generator(z, labels,
                                      style_mixing_prob=style_mixing_prob)

            fake_aug = augment(fake_imgs, ada.p)
            fake_pred = discriminator(fake_aug, labels)
            d_loss_fake = F.softplus(fake_pred).mean()  # push D(fake) down

            d_loss = d_loss_real + d_loss_fake + r1_loss
            d_loss.backward()
            opt_D.step()

            ada.update(real_pred.detach())

            # ─── GENERATOR ───
            opt_G.zero_grad()

            z = torch.randn(curr_batch, z_dim, device=DEVICE)
            fake_imgs = generator(z, labels,
                                  style_mixing_prob=style_mixing_prob)
            fake_aug = augment(fake_imgs, ada.p)
            fake_pred = discriminator(fake_aug, labels)

            g_loss = F.softplus(-fake_pred).mean()  # push D(fake) up

            # PLR (only if enabled and on schedule)
            pl_loss = torch.tensor(0.0, device=DEVICE)
            if pl_weight > 0 and global_step % pl_interval == 0:
                pl_loss, pl_mean = path_length_regularization(
                    generator, z, labels, pl_mean
                )
                pl_loss = pl_loss * pl_weight

            total_g_loss = g_loss + pl_loss
            total_g_loss.backward()
            opt_G.step()

            # EMA update: parameters + buffers (includes w_avg)
            with torch.no_grad():
                for p_ema, p in zip(generator_ema.parameters(),
                                    generator.parameters()):
                    p_ema.data.lerp_(p.data, 1 - ema_decay)
                for b_ema, b in zip(generator_ema.buffers(),
                                    generator.buffers()):
                    b_ema.copy_(b)

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

            pbar.set_postfix({
                "D": f"{d_loss.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "ADA": f"{ada.p:.2f}",
                "R1": f"{r1_loss.item():.3f}",
            })

        # ── History ──
        n_batches = len(dataloader)
        history["G_loss"].append(g_loss_epoch / n_batches)
        history["D_loss"].append(d_loss_epoch / n_batches)
        history["ADA_p"].append(ada.p)

        # ── Periodic visualization ──
        if (epoch + 1) % sample_interval == 0:
            generator_ema.eval()
            with torch.no_grad():
                fake = generator_ema(fixed_z, fixed_labels, truncation_psi=0.7)
                fake = (fake + 1) / 2

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i, ax in enumerate(axes.flatten()):
                    if i < fake.size(0):
                        img = fake[i].cpu().permute(1, 2, 0).numpy()
                        ax.imshow(np.clip(img, 0, 1))
                    ax.axis("off")
                plt.suptitle(f"Epoch {epoch + 1} | ADA p={ada.p:.2f}")
                plt.tight_layout()
                plt.savefig(samples_dir / f"epoch_{epoch + 1:04d}.png", dpi=150)
                plt.close()

        # ── Checkpoint ──
        # `latest.pt` is rewritten every epoch so an interrupted run always
        # resumes from the last completed epoch. The numbered snapshots are
        # archival history and stay on `save_interval`.
        ckpt = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "pl_mean": pl_mean,
            "ada_p": ada.p,
            "history": history,
            "class_names": class_names,
            "config": {"z_dim": z_dim, "w_dim": w_dim, "img_size": img_size},
        }
        # Write to a temp file and rename: os.replace is atomic, so killing the
        # process mid-save can never leave a half-written latest.pt behind.
        tmp_ckpt = latest_ckpt.with_suffix(".tmp")
        torch.save(ckpt, tmp_ckpt)
        os.replace(tmp_ckpt, latest_ckpt)

        if (epoch + 1) % save_interval == 0:
            torch.save(ckpt, output_path / f"checkpoint_{epoch + 1:04d}.pt")
            print(f"\n  Checkpoint saved: epoch {epoch + 1}")

    # ── Save final model ──
    final_ckpt = {
        "generator": generator.state_dict(),
        "generator_ema": generator_ema.state_dict(),
        "discriminator": discriminator.state_dict(),
        "class_names": class_names,
        "config": {"z_dim": z_dim, "w_dim": w_dim, "img_size": img_size},
    }
    torch.save(final_ckpt, output_path / "stylegan2_ada_final.pt")

    return generator_ema, discriminator, class_names, history
