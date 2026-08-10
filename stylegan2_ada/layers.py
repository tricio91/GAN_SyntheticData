"""Core building-block layers shared by the generator and discriminator.

Includes equalized-learning-rate linear/conv layers, the style-modulated
convolution, and the minibatch-standard-deviation layer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate.

    Weights are stored as ``N(0, 1)`` and scaled at runtime by
    ``1 / sqrt(fan_in)``. This keeps the effective learning rate uniform
    across layers regardless of their size.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, lr_mul: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (1 / math.sqrt(in_features)) * lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """2D convolution with equalized learning rate."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        self.stride, self.padding)


class ModulatedConv2dADA(nn.Module):
    """Style-modulated convolution with weight demodulation.

    The style vector ``w`` modulates the filter weights per input channel,
    then demodulation normalizes the output variance. A grouped convolution
    applies per-sample filters in a single call. Per-pixel stochastic noise
    is added *after* the convolution (adding it before would amplify it).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 w_dim: int, upsample: bool = False, demodulate: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.affine = EqualizedLinear(w_dim, in_channels, bias=True)
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, w: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, in_c, h, w_size = x.shape

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h, w_size = h * 2, w_size * 2

        # Modulation
        styles = self.affine(w)
        weight = self.scale * self.weight.unsqueeze(0)
        weight = weight * styles.view(batch, 1, -1, 1, 1)

        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        # Grouped conv — one filter set per sample
        weight = weight.view(
            batch * self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_c, h, w_size)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, h, w_size)

        # Per-pixel stochastic noise
        if noise is None:
            noise = torch.randn(batch, 1, h, w_size, device=x.device)
        out = out + self.noise_strength * noise

        return out


class MinibatchStdDev(nn.Module):
    """Appends a channel with per-group standard deviation.

    Gives the discriminator a signal for detecting mode collapse: a batch of
    near-identical fakes has near-zero within-group variance.
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        group_size = min(self.group_size, batch)

        if batch % group_size != 0:
            group_size = 1

        y = x.view(group_size, batch // group_size, channels, height, width)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width).view(batch, 1, height, width)

        return torch.cat([x, y], dim=1)
