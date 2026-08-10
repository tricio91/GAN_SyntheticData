"""Path length regularization for the generator."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def path_length_regularization(
    generator: nn.Module,
    z: torch.Tensor,
    labels: Optional[torch.Tensor],
    pl_mean: torch.Tensor,
    pl_decay: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize deviation of the Jacobian norm ``dG/dw`` from its running mean.

    Encourages a smooth mapping in ``w`` space. Gradients are taken with
    respect to ``w`` (not ``z``), and a random image projection is used to
    keep the computation cheap.

    Returns the penalty and the updated running mean (detached).
    """
    # Map z->w, detach, then require grad only on w
    w = generator.mapping(z, labels)
    w = w.detach().requires_grad_(True)

    ws = w.unsqueeze(1).repeat(1, generator.num_ws, 1)
    fake = generator.synthesize(ws)

    # Random projection trick to reduce cost
    noise = torch.randn_like(fake) / math.sqrt(fake.shape[2] * fake.shape[3])

    grad = torch.autograd.grad(
        outputs=(fake * noise).sum(),
        inputs=w,
        create_graph=True,
        retain_graph=True,
    )[0]

    pl_length = grad.pow(2).sum(dim=1).sqrt()

    pl_mean_new = pl_mean + pl_decay * (pl_length.mean() - pl_mean)
    pl_penalty = (pl_length - pl_mean_new).pow(2).mean()

    return pl_penalty, pl_mean_new.detach()
