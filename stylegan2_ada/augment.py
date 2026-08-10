"""Adaptive Discriminator Augmentation (ADA).

Contains the ADA controller that adapts the augmentation probability from the
discriminator overfitting signal, and the stochastic augmentation pipeline.
"""

import torch
import torch.nn as nn


class AdaptiveAugment:
    """Adjusts augmentation probability ``p`` from the discriminator signal.

    ``rt`` is the fraction of real images the discriminator scores as real
    (``D > 0``). When ``rt`` exceeds the target the discriminator is
    overfitting, so ``p`` is increased; when it falls below, ``p`` decreases.
    """

    def __init__(self, target_rt: float = 0.6, speed: float = 500,
                 initial_p: float = 0.0):
        self.target_rt = target_rt
        self.speed = speed
        self.p = initial_p
        self.rt_stat = 0.0

    def update(self, real_logits: torch.Tensor) -> float:
        rt = (real_logits.sign().mean().item() + 1) / 2
        self.rt_stat = self.rt_stat * 0.99 + rt * 0.01
        adjust = (self.rt_stat - self.target_rt) / self.speed
        self.p = max(0.0, min(1.0, self.p + adjust))
        return self.p


class AugmentPipe(nn.Module):
    """Stochastic augmentation pipeline applied per image with probability p.

    Combines geometric (translate, flip), color (brightness, contrast)
    and small cutout transforms. The same pipeline is applied to both real
    and fake images so the discriminator cannot exploit augmentation
    artifacts. Kept deliberately mild for a near-grayscale, canonically
    oriented domain to avoid augmentation leaking into the generator.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0:
            return x

        batch = x.size(0)
        device = x.device
        apply_mask = torch.rand(batch, device=device) < p

        if not apply_mask.any():
            return x

        x_aug = x.detach().clone() if not x.requires_grad else x.clone()

        # Geometric — rotate90 removed: the parts have a canonical orientation,
        # so 90-degree rotations are off-distribution and leak into G.
        if torch.rand(1).item() < 0.5:
            x_aug = self._translate(x_aug, apply_mask)
        if torch.rand(1).item() < 0.5:
            x_aug = self._flip(x_aug, apply_mask)

        # Color — saturation removed: the images are near-grayscale, so
        # saturation shifts inject color casts that leak as tinted backgrounds.
        if torch.rand(1).item() < 0.5:
            x_aug = self._brightness(x_aug, apply_mask)
        if torch.rand(1).item() < 0.5:
            x_aug = self._contrast(x_aug, apply_mask)

        # Cutout — small patches only; a large ratio taught G to emit dark blocks.
        if torch.rand(1).item() < 0.3:
            x_aug = self._cutout(x_aug, apply_mask)

        return x_aug

    def _translate(self, x, mask):
        """Shift masked images by up to +/-12.5% along both axes."""
        batch, _, h, w = x.shape
        max_shift = int(h * 0.125)
        for i in range(batch):
            if mask[i]:
                tx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                ty = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                shifted = torch.roll(x[i], shifts=(ty, tx), dims=(1, 2))
                # Zero the wrapped-around edges (fill = -1, black in [-1, 1]) so
                # this is a real translation, not a circular scroll with seams.
                if ty > 0:
                    shifted[:, :ty, :] = -1
                elif ty < 0:
                    shifted[:, ty:, :] = -1
                if tx > 0:
                    shifted[:, :, :tx] = -1
                elif tx < 0:
                    shifted[:, :, tx:] = -1
                x[i] = shifted
        return x

    def _flip(self, x, mask):
        """Randomly flip masked images horizontally and/or vertically."""
        for i in range(x.size(0)):
            if mask[i]:
                if torch.rand(1).item() < 0.5:
                    x[i] = x[i].flip(-1)
                if torch.rand(1).item() < 0.5:
                    x[i] = x[i].flip(-2)
        return x

    def _brightness(self, x, mask):
        """Scale masked images by a random factor in [0.8, 1.2]."""
        for i in range(x.size(0)):
            if mask[i]:
                x[i] = x[i] * torch.empty(1).uniform_(0.8, 1.2).item()
        return x.clamp(-1, 1)

    def _contrast(self, x, mask):
        """Push masked images toward/away from their own mean (contrast)."""
        for i in range(x.size(0)):
            if mask[i]:
                factor = torch.empty(1).uniform_(0.8, 1.2).item()
                mean = x[i].mean()
                x[i] = (x[i] - mean) * factor + mean
        return x.clamp(-1, 1)

    def _cutout(self, x, mask, ratio=0.2):
        """Blank out a random square patch (side = ratio * height)."""
        _, _, h, w = x.shape
        size = int(h * ratio)
        for i in range(x.size(0)):
            if mask[i]:
                top = torch.randint(0, h - size + 1, (1,)).item()
                left = torch.randint(0, w - size + 1, (1,)).item()
                x[i, :, top:top + size, left:left + size] = 0
        return x
