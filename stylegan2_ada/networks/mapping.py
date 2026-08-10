"""Mapping network: transforms latent z into the intermediate style space w."""

from typing import Optional

import torch
import torch.nn as nn

from ..layers import EqualizedLinear


class MappingNetworkADA(nn.Module):
    """Maps a latent code ``z`` to the intermediate style space ``w``.

    Uses 8 fully connected layers with ``lr_mul=0.01`` to prevent the mapping
    network from dominating synthesis. For conditional generation, a learned
    class embedding is concatenated with ``z`` before mapping.
    """

    def __init__(self, z_dim: int, w_dim: int, num_classes: int = 0,
                 num_layers: int = 8, lr_mul: float = 0.01):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_classes = num_classes

        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, z_dim)
            input_dim = z_dim * 2
        else:
            self.class_embed = None
            input_dim = z_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # PixelNorm on z
        z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)

        if self.class_embed is not None and labels is not None:
            class_emb = self.class_embed(labels)
            z = torch.cat([z, class_emb], dim=1)

        return self.mapping(z)
