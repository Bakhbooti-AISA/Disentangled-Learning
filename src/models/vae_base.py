# src/models/vae_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from .outputs import VAEForwardOutput
from ..losses.gaussian import reparameterize


LatentType = Literal["mu", "z"]


@dataclass
class VAEBaseConfig:
    latent_dim: int


class BaseTimeSeriesVAE(nn.Module):
    """
    Base class: encode -> sample -> decode + get_latent().
    Backbones are injected (encoder returns features; decoder takes z + seq_len).
    """
    def __init__(self, cfg: VAEBaseConfig, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

        # These heads map encoder features -> (mu, logvar)
        enc_out_dim = getattr(encoder, "out_dim", None)
        if enc_out_dim is None:
            raise ValueError("Encoder must expose .out_dim (feature dimension).")

        self.to_mu = nn.Linear(enc_out_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(enc_out_dim, cfg.latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)           # [B,H]
        mu = self.to_mu(h)            # [B,D]
        logvar = self.to_logvar(h)    # [B,D]
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.decoder(z, seq_len=seq_len)

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor, latent: LatentType = "mu") -> torch.Tensor:
        """
        Extractable latent API:
          - "mu": deterministic representation
          - "z": stochastic sample
        """
        self.eval()
        mu, logvar = self.encode(x)
        if latent == "mu":
            return mu
        if latent == "z":
            return self.sample(mu, logvar)
        raise ValueError(f"Unknown latent='{latent}'")

    def forward(self, x: torch.Tensor) -> VAEForwardOutput:
        B, T, C = x.shape
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_recon = self.decode(z, seq_len=T)
        return VAEForwardOutput(x_recon=x_recon, mu=mu, logvar=logvar, z=z)
