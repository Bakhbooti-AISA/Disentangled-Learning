# # src/losses/dts_group.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Optional, Dict

# import torch
# import torch.nn.functional as F

# from .gaussian import kl_normal_diag
# from .tc_estimator import estimate_tc_terms

# @dataclass
# class DTSGroupLossStats:
#     loss: float
#     recon: float
#     kl: float
#     kl_qz_pz: float

# def dts_group_loss(
#     x: torch.Tensor,
#     x_recon: torch.Tensor,
#     z: torch.Tensor,
#     mu: torch.Tensor,
#     logvar: torch.Tensor,
#     *,
#     alpha: float,
#     beta: float = 1.0,
#     dataset_size: Optional[int] = None, 
#     recon_loss: str = "mse",
#     act_dim: int,
# ) -> tuple[torch.Tensor, Dict[str, float]]:
#     """
#     Implements the DTS Group Segment ELBO (Eq. 8) with stabilization.
#     """
#     if recon_loss == "mse":
#         recon = F.mse_loss(x_recon, x, reduction="mean")
#     elif recon_loss == "l1":
#         recon = F.l1_loss(x_recon, x, reduction="mean")
#     else:
#         raise ValueError(f"Unknown recon_loss='{recon_loss}'")

#     # 1. Standard KL Divergence (Penalty)
#     kl = kl_normal_diag(mu, logvar) 
#     kl_mean = kl.mean()

#     # 2. Marginal KL Divergence (Reward for MI)
#     tc_terms = estimate_tc_terms(z=z, mu=mu, logvar=logvar, dataset_size=dataset_size)
#     kl_qz_pz = tc_terms.kl_qz_pz.mean()

#     # --- FIX 1: STABILIZATION ---
#     # Prevents the optimizer from driving Marginal KL to infinity to minimize loss.
#     # We clamp the term so it contributes to gradients only when within a reasonable range.
#     # If kl_qz_pz > 100.0 (e.g., extremely distinct clusters), we cap the reward.
#     marginal_kl_reward = torch.clamp(kl_qz_pz, max=100.0)
    
#     # Loss = Recon + beta * KL(q|p) - alpha * KL(q||p)
#     # The clamp prevents 'loss' from exploding to -1e14
#     loss = recon + beta * kl_mean - alpha * marginal_kl_reward

#     stats = {
#         "loss": float(loss.detach().cpu()),
#         "recon": float(recon.detach().cpu()),
#         "kl": float(kl_mean.detach().cpu()),
#         "kl_qz_pz": float(kl_qz_pz.detach().cpu()),
#     }
#     return loss, stats


# src/losses/dts_group.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DTSGroupLossStats:
    loss: float
    recon: float
    kl_seg1: float
    kl_seg2: float
    kl_qz_pz: float
    batch_size: int
    latent_dim: int


def _kl_diag_gaussian_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) ) per sample.
    Returns shape: [B]
    """
    # stabilize logvar to avoid inf / nan from exp()
    logvar = logvar.clamp(min=-30.0, max=20.0)
    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=-1)


def _log_normal_diag(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Log density of N(mu, diag(exp(logvar))) at x, summed over last dim.
    Shapes broadcast as usual. Output summed over last dim.
    """
    logvar = logvar.clamp(min=-30.0, max=20.0)
    # log N(x; mu, var) = -0.5 * [ (x-mu)^2/var + logvar + log(2pi) ]
    return -0.5 * (((x - mu) ** 2) / logvar.exp() + logvar + math.log(2.0 * math.pi)).sum(dim=-1)


def _estimate_kl_qz_pz_minibatch(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Estimates KL(q(z) || p(z)) where q(z) is the aggregated posterior:
        q(z) = E_{p_D(x)}[ q(z|x) ]  (Eq. 8 uses KL(q(z)||p(z)) term)
    using a minibatch log-sum-exp estimator similar to TC-VAE style.

    Returns a scalar tensor.
    """
    # z, mu, logvar: [B, D]
    B, D = z.shape

    # Compute log q(z_b | x_i) for all pairs (b, i): [B, B]
    # z_b expanded to compare against each (mu_i, logvar_i)
    z_b = z.unsqueeze(1)         # [B, 1, D]
    mu_i = mu.unsqueeze(0)       # [1, B, D]
    lv_i = logvar.unsqueeze(0)   # [1, B, D]
    log_q_z_given_x = _log_normal_diag(z_b, mu_i, lv_i)  # [B, B]

    # q(z) = (1/N) sum_i q(z|x_i). Use N if provided else batch-size.
    # norm = float(dataset_size) if dataset_size is not None else float(B)
    # log_q_z = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(norm)  # [B]
    log_q_z = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(B)


    # log p(z) under standard normal: sum_d log N(0,1)
    log_p_z = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)  # [B]

    # KL(q(z)||p(z)) = E_{q(z)}[log q(z) - log p(z)]
    return (log_q_z - log_p_z).mean()


def dts_group_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    alpha: float,
    beta_aniller: float = 1.0,
    dataset_size: Optional[int] = None,
    recon_loss: str = "mse",
    act_dim: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Implements the DTS Group Segment ELBO (Eq. 8) as a minimization loss.

    Eq. (8) (paper) for two segments g_i, g_j:
        L_ELBO-G(x) =
            - KL(q(g_i|x) || p(g_i))
            - KL(q(g_j|x) || p(g_j))
            + E[ log p(x | g_i, g_j) ]
            + β * KL(q(z) || p(z))

    We minimize -L_ELBO-G, so:
        loss = recon + KL_seg_i + KL_seg_j - alpha * KL(q(z)||p(z))

    Here:
      - We assume z/mu/logvar are the concatenation of two equal-sized segments.
      - beta scales the segment KL terms (often left at 1.0).
      - alpha corresponds to the +β weight in Eq. (8) and therefore appears with a minus sign in the loss.
    """
    if x.shape != x_recon.shape:
        raise ValueError(f"x and x_recon must have same shape, got {tuple(x.shape)} vs {tuple(x_recon.shape)}")
    if z.ndim != 2 or mu.ndim != 2 or logvar.ndim != 2:
        raise ValueError("z, mu, logvar must be rank-2 tensors of shape [B, D]")
    if z.shape != mu.shape or z.shape != logvar.shape:
        raise ValueError(f"z, mu, logvar must have same shape, got {tuple(z.shape)}, {tuple(mu.shape)}, {tuple(logvar.shape)}")

    B, D = z.shape
    if D % 2 != 0:
        raise ValueError(f"Group loss expects an even latent dim (two equal segments). Got latent_dim={D}.")

    # Reconstruction as negative log-likelihood proxy:
    # Use sum over features per sample, then mean over batch (matches ELBO scaling better than a global mean).
    if recon_loss == "mse":
        # recon_per = F.mse_loss(x_recon, x, reduction="none").reshape(B, -1).mean(dim=1)
        recon_per = F.mse_loss(x_recon, x, reduction="none").reshape(B, -1).sum(dim=1)
    elif recon_loss == "l1":
        recon_per = F.l1_loss(x_recon, x, reduction="none").reshape(B, -1).mean(dim=1)
    else:
        raise ValueError(f"Unknown recon_loss='{recon_loss}'")
    recon = recon_per.mean()
    

    # if recon_loss == "mse":
    #     recon = F.mse_loss(x_recon, x, reduction="mean")
    # elif recon_loss == "l1":
    #     recon = F.l1_loss(x_recon, x, reduction="mean")
    # else:
    #     raise ValueError(f"Unknown recon_loss='{recon_loss}'")

    # Split into two group segments (g_i, g_j)
    mu1, mu2 = mu[:, :act_dim], mu[:, act_dim:]
    lv1, lv2 = logvar[:, :act_dim], logvar[:, act_dim:]

    kl_seg1 = _kl_diag_gaussian_to_standard_normal(mu1, lv1).mean()
    kl_seg2 = _kl_diag_gaussian_to_standard_normal(mu2, lv2).mean()

    # Aggregated posterior KL term: KL(q(z)||p(z))
    kl_qz_pz = _estimate_kl_qz_pz_minibatch(z=z, mu=mu, logvar=logvar, dataset_size=dataset_size)

    # Minimize -ELBO_G:
    loss = (1)*(recon) + beta_aniller*((kl_seg1 + kl_seg2) - alpha * kl_qz_pz)

    stats = {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl_seg1": float(kl_seg1.detach().cpu()),
        "kl_seg2": float(kl_seg2.detach().cpu()),
        "kl_qz_pz": float(kl_qz_pz.detach().cpu()),
        "alpha": float(alpha),
        "beta": float(beta_aniller),
        "act_dim": int(act_dim),
        "latent_dim": int(D),
    }
    return loss, stats
