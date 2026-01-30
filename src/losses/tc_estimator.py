# src/losses/tc_estimator.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .gaussian import log_normal_diag, LOG_2PI


@dataclass
class TCTerms:
    log_qzx: torch.Tensor        # [B]
    log_qz: torch.Tensor         # [B]
    log_prod_qz: torch.Tensor    # [B]
    log_pz: torch.Tensor         # [B]
    tc: torch.Tensor             # scalar
    dwkl: torch.Tensor           # scalar
    kl_qz_pz: torch.Tensor       # scalar


@torch.no_grad()
def _log_importance_weight_matrix(batch_size: int, dataset_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Returns the log-weight matrix for the Minibatch Weighted Sampling estimator.
    """
    N = dataset_size
    M = batch_size
    
    # --- FIX 3A: Safety Check ---
    # If dataset is smaller than batch (rare) or equal, MWS isn't needed/stable.
    if N is None or N <= M:
        return None

    # Construct the weight matrix in log-space to avoid underflow
    # Weights: (1/(M-1)) for off-diagonal, (1/N) for diagonal
    # We construct this carefully to avoid division by zero
    
    # Standard approximation (Stratified Sampling):
    # w_ij = 1/(M(N-1)) if i != j, else 1/N ? 
    # The 'Minibatch Weighted Sampling' (MWS) usually uses:
    #   p(z) \approx \sum_j (r_j/M) q(z|x_j)
    #   where r_j = N/M (importance weight).
    # If simple Stratified:
    #   W[i,j] = 1/(M-1) * (1 - 1/N) for i!=j
    #   W[i,i] = 1/N
    
    # Simplified implementation that matches typical beta-TCVAE codebases:
    W = torch.full((M, M), 1.0 / (M - 1), device=device)
    W.view(-1)[:: M + 1] = 1.0 / N
    
    # Normalize so rows sum to 1
    W = W / W.sum(dim=1, keepdim=True)
    return W


def estimate_tc_terms(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: Optional[int] = None,
) -> TCTerms:
    """
    Estimates TC, DWKL, KL(q(z)||p(z)) with numerical safeguards.
    """
    B, D = z.shape
    device = z.device

    # 1. log q(z|x): exact
    log_qzx = log_normal_diag(z, mu, logvar)  # [B]

    # 2. log p(z): exact under standard normal
    zeros = torch.zeros_like(z)
    log_pz = log_normal_diag(z, zeros, zeros)  # [B]

    # 3. Compute log q(z_i | x_j) matrix
    z_i = z.unsqueeze(1)       # [B, 1, D]
    mu_j = mu.unsqueeze(0)     # [1, B, D]
    logvar_j = logvar.unsqueeze(0)

    # per-dim log prob: [B, B, D]
    # "What is prob of sample i given encoder output of sample j?"
    log_qz_condx_perdim = -0.5 * (LOG_2PI + logvar_j + (z_i - mu_j).pow(2) / torch.exp(logvar_j))

    # joint over dims: [B, B]
    log_qz_condx_joint = log_qz_condx_perdim.sum(dim=2)

    # 4. log prod_j q(z_j): Marginals
    # log q(z_ik) = log ( 1/B * sum_j q(z_ik | x_j) )
    # logsumexp over batch dim (dim=1) - log(B)
    log_qz_marginals = torch.logsumexp(log_qz_condx_perdim, dim=1) - math.log(B)
    log_prod_qz = log_qz_marginals.sum(dim=1)  # [B]

    # 5. log q(z): Aggregated Posterior
    # Uses importance weighting if dataset_size provided
    if dataset_size is not None and dataset_size > B:
        W = _log_importance_weight_matrix(B, dataset_size, device)
        if W is not None:
            log_w = torch.log(W + 1e-12)  # Add epsilon for log safety
            # Weighted logsumexp: log( sum( w_ij * exp(log_q) ) )
            # = logsumexp( log_w + log_q )
            log_qz = torch.logsumexp(log_w + log_qz_condx_joint, dim=1)
        else:
            log_qz = torch.logsumexp(log_qz_condx_joint, dim=1) - math.log(B)
    else:
        # Standard estimator (bias towards training batch)
        log_qz = torch.logsumexp(log_qz_condx_joint, dim=1) - math.log(B)

    # --- FIX 3B: Output Stability ---
    # Calculate differences. These are mathematically >= 0 (KL divergence),
    # but numerical noise can make them slightly negative.
    
    # TC = KL(q(z) || \prod q(z_j)) = log q(z) - \sum log q(z_j)
    tc = (log_qz - log_prod_qz).mean()
    
    # DWKL = \sum KL(q(z_j) || p(z_j)) = \sum log q(z_j) - log p(z)
    dwkl = (log_prod_qz - log_pz).mean()
    
    # Marginal KL = KL(q(z) || p(z)) = log q(z) - log p(z)
    kl_qz_pz = (log_qz - log_pz).mean()

    return TCTerms(
        log_qzx=log_qzx,
        log_qz=log_qz,
        log_prod_qz=log_prod_qz,
        log_pz=log_pz,
        tc=tc,
        dwkl=dwkl,
        kl_qz_pz=kl_qz_pz,
    )