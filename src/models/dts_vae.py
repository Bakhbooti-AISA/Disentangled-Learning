# src/models/dts_vae.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .vae_base import BaseTimeSeriesVAE, VAEBaseConfig
from .outputs import VAEForwardOutput
from .heads import GradientReversal, MLPHead, MLPHeadConfig
from ..losses.dts_eq7 import dts_eq7_loss
from ..losses.dts_group import dts_group_loss


@dataclass
class DTSVAEConfig(VAEBaseConfig):
    # DTS Eq(7)
    alpha: float = 10.0
    beta: float = 5.0
    recon_loss: str = "mse"
    loss_type: str = "group" # "individual" (Eq 7) or "group" (Eq 8)

    # Activity classification
    num_classes: Optional[int] = None
    cls_weight: float = 1.0

    # GRL / subject disentanglement
    use_grl: bool = False
    num_subjects: Optional[int] = None  # required if use_grl True
    act_dim: Optional[int] = None        # if None, defaults to latent_dim//2
    grl_lambda: float = 1.0

    subj_weight: float = 0.2       # CE(mu_subj -> subject)
    subj_adv_weight: float = 0.5   # CE(GRL(mu_act) -> subject)
    act_adv_weight: float = 0.5   # CE(GRL(mu_subj) -> activity) removes activity info from mu_subj

    head_hidden_dim: int = 128
    head_dropout: float = 0.0




class DTSVAE(BaseTimeSeriesVAE):
    """
    DTS-like VAE + optional:
      - activity classifier on mu_act (or mu if no split)
      - subject classifier on mu_subj
      - adversarial subject classifier on GRL(mu_act) to remove subject info from mu_act
    """
    def __init__(self, cfg: DTSVAEConfig, encoder: nn.Module, decoder: nn.Module):
        super().__init__(cfg, encoder, decoder)
        self.cfg: DTSVAEConfig = cfg

        # Heads
        self.activity_head: Optional[nn.Module] = None
        self.activity_head_adv: Optional[nn.Module] = None
        self.subject_head: Optional[nn.Module] = None
        self.subject_head_adv: Optional[nn.Module] = None
        self.grl: Optional[nn.Module] = None

        if cfg.num_classes is not None:
            # If using GRL we classify from mu_act; else from full mu
            act_in_dim = self._act_dim()
            self.activity_head = MLPHead(
                MLPHeadConfig(
                    in_dim=act_in_dim,
                    out_dim=cfg.num_classes,
                    hidden_dim=cfg.head_hidden_dim,
                    dropout=cfg.head_dropout,
                )
            )

        if cfg.use_grl:
            if cfg.num_subjects is None:
                raise ValueError("cfg.num_subjects must be set when use_grl=True")

            subj_dim = self._subj_dim()
            self.subject_head = MLPHead(
                MLPHeadConfig(
                    in_dim=subj_dim,
                    out_dim=cfg.num_subjects,
                    hidden_dim=cfg.head_hidden_dim,
                    dropout=cfg.head_dropout,
                )
            )
            self.grl = GradientReversal(cfg.grl_lambda)

            # subject from GRL(mu_act)
            self.subject_head_adv = MLPHead(
                MLPHeadConfig(
                    in_dim=self._act_dim(),
                    out_dim=cfg.num_subjects,
                    hidden_dim=cfg.head_hidden_dim,
                    dropout=cfg.head_dropout,
                )
            )

            # NEW: activity from GRL(mu_subj)  (cross-adversarial)
            if cfg.num_classes is None:
                raise ValueError("cfg.num_classes must be set when use_grl=True for cross-adversarial activity")
            self.activity_head_adv = MLPHead(
                MLPHeadConfig(
                    in_dim=subj_dim,
                    out_dim=cfg.num_classes,
                    hidden_dim=cfg.head_hidden_dim,
                    dropout=cfg.head_dropout,
                )
            )
        else:
            self.activity_head_adv = None


    def _act_dim(self) -> int:
        if self.cfg.use_grl:
            if self.cfg.act_dim is None:
                return self.cfg.latent_dim // 2
            return int(self.cfg.act_dim)
        # no split => act uses full mu
        return int(self.cfg.latent_dim)

    def _subj_dim(self) -> int:
        # only meaningful when use_grl
        return int(self.cfg.latent_dim - self._act_dim())

    def _split_mu(self, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split mu into (mu_act, mu_subj). Only used when cfg.use_grl.
        """
        act_dim = self._act_dim()
        mu_act = mu[:, :act_dim]
        mu_subj = mu[:, act_dim:]
        return mu_act, mu_subj

    def forward(self, x: torch.Tensor) -> VAEForwardOutput:
        out = super().forward(x)

        if self.activity_head is not None:
            if self.cfg.use_grl:
                mu_act, _ = self._split_mu(out.mu)
                out.logits = self.activity_head(mu_act)
            else:
                out.logits = self.activity_head(out.mu)

        if self.cfg.use_grl:
            mu_act, mu_subj = self._split_mu(out.mu)
            out.subj_logits = self.subject_head(mu_subj)

            # adversarial: try to predict subject from mu_act, but reverse gradients to encoder
            out.subj_logits_adv = self.subject_head_adv(self.grl(mu_act))
            
            # NEW: adversarial activity prediction from mu_subj (remove activity info from mu_subj)
            out.logits_adv = self.activity_head_adv(self.grl(mu_subj))

        return out

    def compute_loss(
        self,
        x: torch.Tensor,
        out: VAEForwardOutput,
        *,
        y: Optional[torch.Tensor] = None,
        subject: Optional[torch.Tensor] = None,
        dataset_size: Optional[int] = None,
        apply_subject_losses: bool = True,   # <--- NEW
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        # VAE loss (Eq. 7)
        if self.cfg.loss_type == "individual":
            loss_vae, stats = dts_eq7_loss(
                x=x,
                x_recon=out.x_recon,
                z=out.z,
                mu=out.mu,
                logvar=out.logvar,
                alpha=self.cfg.alpha,
                beta=self.cfg.beta,
                dataset_size=dataset_size,
                recon_loss=self.cfg.recon_loss,
            )
        elif self.cfg.loss_type == "group":
            # VAE Group ELBO (Eq. 8)
            loss_vae, stats = dts_group_loss(
                x=x,
                x_recon=out.x_recon,
                z=out.z,               # <--- Pass Z
                mu=out.mu,
                logvar=out.logvar,
                alpha=self.cfg.alpha,  # <--- Pass Alpha
                beta_aniller=self.cfg.beta,
                dataset_size=dataset_size,
                recon_loss=self.cfg.recon_loss,
                act_dim=self._act_dim(),
            )
        else:
            raise ValueError(f"Unknown loss_type='{self.cfg.loss_type}'")

        loss = loss_vae
        stats["loss_vae"] = stats["loss"]

        # Activity classification
        if out.logits is not None and y is not None:
            cls = nn.functional.cross_entropy(out.logits, y)
            loss = loss + self.cfg.cls_weight * cls * self.cfg.beta
            stats["cls"] = float(cls.detach().cpu())

        # Subject classification diagnostics (always computed if available),
        # but only added to loss if apply_subject_losses=True.
        if self.cfg.use_grl:
            if subject is None:
                raise ValueError("subject must be provided when use_grl=True")

            if out.subj_logits is not None:
                subj = nn.functional.cross_entropy(out.subj_logits, subject)
                stats["subj"] = float(subj.detach().cpu())
                if apply_subject_losses:
                    loss = loss + self.cfg.subj_weight * subj * self.cfg.beta

            if out.subj_logits_adv is not None:
                subj_adv = nn.functional.cross_entropy(out.subj_logits_adv, subject)
                stats["subj_adv"] = float(subj_adv.detach().cpu())
                if apply_subject_losses:
                    loss = loss + self.cfg.subj_adv_weight * subj_adv * self.cfg.beta

            # Cross-adversarial: activity from mu_subj via GRL
            if out.logits_adv is not None and y is not None:
                act_adv = nn.functional.cross_entropy(out.logits_adv, y)
                stats["act_adv"] = float(act_adv.detach().cpu())
                if apply_subject_losses:  # training only
                    loss = loss + self.cfg.act_adv_weight * act_adv * self.cfg.beta

        stats["loss_total"] = float(loss.detach().cpu())
        return loss, stats

