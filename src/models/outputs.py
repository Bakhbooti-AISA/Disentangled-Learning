# src/models/outputs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VAEForwardOutput:
    x_recon: torch.Tensor        # [B,T,C]
    mu: torch.Tensor             # [B,D]
    logvar: torch.Tensor         # [B,D]
    z: torch.Tensor              # [B,D]

    # Activity classification (main task)
    logits: Optional[torch.Tensor] = None          # [B,num_classes]
    logits_adv: Optional[torch.Tensor] = None      # [B,num_classes] from GRL(mu_subj)

    # Subject classification (for disentanglement)
    subj_logits: Optional[torch.Tensor] = None       # [B,num_subjects]
    subj_logits_adv: Optional[torch.Tensor] = None   # [B,num_subjects] from GRL(mu_act)
