"""StyleGAN2 discriminator: residual block and full discriminator."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import EqualizedConv2d, EqualizedLinear, MinibatchStdDev
from .generator import CHANNELS


class DiscriminatorBlock(nn.Module):
    """Residual block with 2x downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, 1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(F.avg_pool2d(x, 2))

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        # 1/sqrt(2) to keep variance after the residual add
        return (x + residual) / math.sqrt(2)


class DiscriminatorStyleGAN2(nn.Module):
    """StyleGAN2 discriminator.

    Residual downsampling blocks followed by a minibatch-stddev layer.
    For multi-class data it uses projection conditioning: the class embedding
    is projected onto the feature vector and added to the real/fake logit.
    """

    def __init__(self, img_size: int = 256, img_channels: int = 3,
                 num_classes: int = 1):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        in_ch = CHANNELS.get(img_size, 32)
        self.from_rgb = EqualizedConv2d(img_channels, in_ch, 1)

        self.blocks = nn.ModuleList()
        res = img_size
        while res > 4:
            out_ch = CHANNELS.get(res // 2, 512)
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch
            res //= 2

        self.mbstd = MinibatchStdDev()
        self.conv_final = EqualizedConv2d(in_ch + 1, 512, 3, padding=1)
        self.fc = EqualizedLinear(512 * 4 * 4, 512)
        self.out = EqualizedLinear(512, 1)

        if num_classes > 1:
            self.class_embed = nn.Embedding(num_classes, 512)
        else:
            self.class_embed = None

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.act(self.from_rgb(x))

        for block in self.blocks:
            x = block(x)

        x = self.mbstd(x)
        x = self.act(self.conv_final(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc(x))

        out = self.out(x)

        if self.class_embed is not None and labels is not None:
            class_emb = self.class_embed(labels)
            out = out + (x * class_emb).sum(dim=1, keepdim=True)

        return out
