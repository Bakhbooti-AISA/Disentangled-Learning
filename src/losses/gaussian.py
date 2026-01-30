# src/losses/gaussian.py
from __future__ import annotations

import math
import torch

LOG_2PI = math.log(2.0 * math.pi)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    z = mu + sigma * eps, where sigma = exp(0.5 * logvar)
    mu, logvar: [B, D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def log_normal_diag(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    log N(x; mean, exp(logvar)) summed over last dim.
    x, mean, logvar: [*, D]
    returns: [*]
    """
    return -0.5 * (LOG_2PI + logvar + (x - mean).pow(2) / torch.exp(logvar)).sum(dim=-1)


def kl_normal_diag(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z)||p(z)) where q(z) ~ N(mu, exp(logvar)) and p(z) ~ N(0, I).
    Summed over last dim.
    mu, logvar: [*, D]
    returns: [*]
    """
    return -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar)).sum(dim=-1)
