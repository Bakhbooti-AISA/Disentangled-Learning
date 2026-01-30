# src/losses/dts_eq7.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn.functional as F

from .tc_estimator import estimate_tc_terms


@dataclass
class DTSLossStats:
    loss: float
    recon: float
    tc: float
    dwkl: float
    kl_qz_pz: float
    # mi: float  # Removed as it's not part of the Eq 7 explicit calculation


def dts_eq7_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    dataset_size: Optional[int] = None,
    recon_loss: str = "mse",
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Implements the DTS Eq.(7) objective in (negative) loss form.

    Eq 7 Objective (Maximization):
      L = E[log p(x|z)] - beta*TC - beta*DWKL + (alpha - beta)*KL(q(z)||p(z))

    Loss to Minimize:
      Loss = -L = Recon + beta*TC + beta*DWKL - (alpha - beta)*KL(q(z)||p(z))
    
    Corrects the previous misinterpretation where alpha weighted TC.
    """
    B, D = z.shape
    if D % 2 != 0:
        raise ValueError(f"loss expects an even latent dim (two equal segments). Got latent_dim={D}.")

    if recon_loss == "mse":
        recon_per = F.mse_loss(x_recon, x, reduction="none").reshape(B, -1).mean(dim=1)
    elif recon_loss == "l1":
        recon_per = F.l1_loss(x_recon, x, reduction="none").reshape(B, -1).mean(dim=1)
    else:
        raise ValueError(f"Unknown recon_loss='{recon_loss}'")
    recon = recon_per.mean()

    tc_terms = estimate_tc_terms(z=z, mu=mu, logvar=logvar, dataset_size=dataset_size)

    # Note: tc_estimator returns scalar means for these terms
    
    # 1. Disentanglement Penalties (Weighted by Beta)
    # The paper applies beta to both Total Correlation and Dimension-wise KL
    disentanglement_loss = beta * (tc_terms.tc + tc_terms.dwkl)

    # 2. Information Maximization (Weighted by Alpha - Beta)
    # The paper adds (alpha - beta) * KL(q(z)||p(z)) to the ELBO.
    # Since we minimize loss, we SUBTRACT this term.
    # Increasing alpha increases the reward for Marginal KL, preventing KL vanishing.
    info_max_term = (alpha - beta) * tc_terms.kl_qz_pz

    kl_qzx_pz = (tc_terms.log_qzx - tc_terms.log_pz).mean()
    # 3. Final Loss
    loss = recon + disentanglement_loss - info_max_term + kl_qzx_pz


    stats = {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "tc": float(tc_terms.tc.detach().cpu()),
        "dwkl": float(tc_terms.dwkl.detach().cpu()),
        "kl_qz_pz": float(tc_terms.kl_qz_pz.detach().cpu()),
        "kl_qzx_pz": float(kl_qzx_pz.detach().cpu()),
    }
    return loss, stats