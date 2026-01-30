# src/backbones/lstm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LSTMEncoderConfig:
    input_size: int          # C
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False


class LSTMEncoder(nn.Module):
    """
    Generic LSTM encoder for time series windows.
    Input:  x [B,T,C]
    Output: h [B,H] (last layer last timestep hidden)
    """
    def __init__(self, cfg: LSTMEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)          # h_n: [L*num_dir, B, H]
        h_last = h_n[-1]                    # [B, H] (last layer, last direction)
        return h_last


@dataclass
class LSTMDecoderConfig:
    output_size: int         # C
    latent_dim: int          # D
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class LSTMDecoder(nn.Module):
    """
    Simple decoder:
      z [B,D] -> repeat to [B,T,D] -> LSTM -> projection -> x_recon [B,T,C]
    """
    def __init__(self, cfg: LSTMDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        # z: [B,D]
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)  # [B,T,D]
        h, _ = self.lstm(z_seq)                         # [B,T,H]
        x_recon = self.proj(h)                          # [B,T,C]
        return x_recon
