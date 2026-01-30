# src/models/heads.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Multiply gradient by -lambda (reverse)
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer (GRL):
      forward: identity
      backward: multiplies gradient by -lambda
    """
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFn.apply(x, self.lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)


@dataclass
class MLPHeadConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    dropout: float = 0.0


class MLPHead(nn.Module):
    """
    Simple 2-layer MLP head for classification.
    """
    def __init__(self, cfg: MLPHeadConfig):
        super().__init__()
        layers = [
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
        ]
        if cfg.dropout and cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_dim, cfg.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
