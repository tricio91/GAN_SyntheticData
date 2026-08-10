"""StyleGAN2 generator: synthesis block and full generator."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ModulatedConv2dADA
from .mapping import MappingNetworkADA

# Channel counts per resolution, shared with the discriminator.
CHANNELS = {
    4: 512, 8: 512, 16: 512, 32: 512,
    64: 256, 128: 128, 256: 64, 512: 32,
}


class SynthesisBlock(nn.Module):
    """One synthesis stage: conv1 (upsample) + conv2 + toRGB.

    Each convolution and the toRGB branch consume their own style slot from
    ``w``, and the RGB output is accumulated via a progressive skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, w_dim: int,
                 resolution: int, upsample: bool = True):
        super().__init__()
        self.resolution = resolution

        self.conv1 = ModulatedConv2dADA(in_channels, out_channels, 3,
                                        w_dim, upsample=upsample)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = ModulatedConv2dADA(out_channels, out_channels, 3,
                                        w_dim, upsample=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # toRGB: 1x1, no demodulation (per the StyleGAN2 paper).
        self.to_rgb = ModulatedConv2dADA(out_channels, 3, 1, w_dim,
                                         demodulate=False)

    def forward(self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                w_rgb: torch.Tensor, rgb: Optional[torch.Tensor] = None):
        x = self.act1(self.conv1(x, w1))
        x = self.act2(self.conv2(x, w2))

        # Progressive RGB skip connection
        rgb_new = self.to_rgb(x, w_rgb)
        if rgb is not None:
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear",
                                align_corners=False)
            rgb = rgb + rgb_new
        else:
            rgb = rgb_new

        return x, rgb


class GeneratorStyleGAN2(nn.Module):
    """StyleGAN2 generator.

    Maps ``z`` to ``w`` and synthesizes an image from a learned 4x4 constant,
    upsampling once per block. The number of style vectors is
    ``2`` (initial block) ``+ 3`` per subsequent block — 20 for 256x256.
    A running mean of ``w`` (``w_avg``) is accumulated during training and used
    for the truncation trick at inference.
    """

    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_classes: int = 1,
                 img_size: int = 256, img_channels: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_size = img_size

        self.log2_size = int(math.log2(img_size))
        assert img_size == 2 ** self.log2_size, "img_size must be power of 2"

        self.mapping = MappingNetworkADA(z_dim, w_dim, num_classes, num_layers=8)

        # Learned 4x4 constant
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Initial block (4x4, no upsample)
        self.conv1 = ModulatedConv2dADA(512, 512, 3, w_dim, upsample=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb1 = ModulatedConv2dADA(512, 3, 1, w_dim, demodulate=False)

        self.blocks = nn.ModuleList()
        in_ch = 512
        for res in [8, 16, 32, 64, 128, 256, 512]:
            if res > img_size:
                break
            out_ch = CHANNELS.get(res, 32)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, res))
            in_ch = out_ch

        # 2 from initial block + 3 per subsequent block
        self.num_ws = 2 + len(self.blocks) * 3

        # Running mean of w for truncation
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def synthesize(self, ws: torch.Tensor) -> torch.Tensor:
        """Takes ws ``[B, num_ws, w_dim]`` and returns an image."""
        batch = ws.size(0)
        x = self.const.repeat(batch, 1, 1, 1)

        # Initial 4x4 block
        x = self.act1(self.conv1(x, ws[:, 0]))
        rgb = self.to_rgb1(x, ws[:, 1])

        # Progressive blocks (8x8+)
        w_idx = 2
        for block in self.blocks:
            x, rgb = block(x, ws[:, w_idx], ws[:, w_idx + 1],
                           ws[:, w_idx + 2], rgb)
            w_idx += 3

        if rgb.size(-1) != self.img_size:
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)

        return torch.tanh(rgb)

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None,
                truncation_psi: float = 1.0,
                style_mixing_prob: float = 0.0) -> torch.Tensor:
        batch = z.size(0)

        w = self.mapping(z, labels)

        # Accumulate w mean (training only)
        if self.training:
            self.w_avg.lerp_(w.detach().mean(dim=0), 0.005)

        # Truncation: interpolate w towards mean
        if truncation_psi < 1.0:
            w = self.w_avg.lerp(w, truncation_psi)

        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)

        # Style mixing: swap w from a random cutoff point
        if self.training and style_mixing_prob > 0:
            if torch.rand(1).item() < style_mixing_prob:
                z2 = torch.randn_like(z)
                w2 = self.mapping(z2, labels)
                cutoff = torch.randint(1, self.num_ws, (1,)).item()
                ws[:, cutoff:] = w2.unsqueeze(1).expand(
                    -1, self.num_ws - cutoff, -1
                )

        return self.synthesize(ws)
