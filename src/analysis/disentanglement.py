# src/analysis/disentanglement.py
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

@torch.no_grad()
def extract_latents(
    model: nn.Module, 
    dl: DataLoader, 
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extracts mu, mu_act, mu_subj, y, and subject labels from the dataloader.
    Returns a dictionary of numpy arrays.
    """
    model.eval()
    
    mu_list = []
    y_list = []
    subj_list = []
    
    for batch in dl:
        # Unpack batch - handles (x, y, subj) or (x, y)
        if len(batch) == 3:
            x, y, s = batch
        else:
            x, y = batch
            s = torch.zeros_like(y) # Dummy if not present
            
        x = x.to(device)
        
        # Forward pass
        # distinct model signatures? assumes DTSVAE-like interface
        if hasattr(model, "encode"):
            mu, _ = model.encode(x)
        else:
            # Fallback
            out = model(x)
            mu = out.mu
            
        mu_list.append(mu.cpu().numpy())
        y_list.append(y.numpy())
        subj_list.append(s.numpy())
        
    mu = np.concatenate(mu_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subj = np.concatenate(subj_list, axis=0)
    
    # Split if model has splitting logic
    # Try DTSVAE specific logic
    if hasattr(model, "_split_mu"):
        # We need to convert back to tensor temporarily or replicate logic
        # Replicating logic is safer if simple
        act_dim = model._act_dim() if hasattr(model, "_act_dim") else mu.shape[1] // 2
        mu_act = mu[:, :act_dim]
        mu_subj = mu[:, act_dim:]
    else:
        # Fallback: assume 50/50 split
        d = mu.shape[1] // 2
        mu_act = mu[:, :d]
        mu_subj = mu[:, d:]
        
    return {
        "mu": mu,
        "mu_act": mu_act,
        "mu_subj": mu_subj,
        "y": y,
        "subject": subj
    }

def train_probe_score(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    max_iter: int = 1000
) -> float:
    """
    Trains a Logistic Regression probe and returns test accuracy.
    """
    # Simple pipeline: Scale -> LR
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=max_iter)
    )
    
    try:
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
    except Exception as e:
        print(f"Warning: Probe training failed with error {e}. Returning 0.0")
        score = 0.0
        
    return float(score)

def validate_disentanglement(
    model: nn.Module,
    dl_train: DataLoader,
    dl_test: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Runs the full disentanglement validation suite.
    """
    print("Extracting latents...")
    train_data = extract_latents(model, dl_train, device)
    test_data = extract_latents(model, dl_test, device)
    
    results = {}
    
    # 1. Activity Utility: mu_act -> y (Should be HIGH)
    print("Probing Activity Utility (mu_act -> y)...")
    results["acc_act_utility"] = train_probe_score(
        train_data["mu_act"], train_data["y"],
        test_data["mu_act"], test_data["y"]
    )
    
    # 2. Subject Utility: mu_subj -> subject (Should be HIGH)
    print("Probing Subject Utility (mu_subj -> subject)...")
    results["acc_subj_utility"] = train_probe_score(
        train_data["mu_subj"], train_data["subject"],
        test_data["mu_subj"], test_data["subject"]
    )
    
    # 3. Subject Leakage: mu_act -> subject (Should be LOW/Chance)
    print("Probing Subject Leakage (mu_act -> subject)...")
    results["acc_subj_leakage"] = train_probe_score(
        train_data["mu_act"], train_data["subject"],
        test_data["mu_act"], test_data["subject"]
    )
    
    # 4. Activity Leakage: mu_subj -> y (Should be LOW)
    print("Probing Activity Leakage (mu_subj -> y)...")
    results["acc_act_leakage"] = train_probe_score(
        train_data["mu_subj"], train_data["y"],
        test_data["mu_subj"], test_data["y"]
    )
    
    return results


from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def _fit_lr(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 2000,
    multi_class: str = "auto",
) -> Tuple[object, np.ndarray]:
    """
    Fit a scaler+logreg pipeline. Returns (fitted_pipe, classes_).
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            multi_class=multi_class,
        ),
    )
    pipe.fit(X, y)
    # classes_ lives on the LogisticRegression step
    classes_ = pipe.named_steps["logisticregression"].classes_
    return pipe, classes_


def _safe_macro_auc_multiclass(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """
    Macro AUC (OvR). Returns NaN if cannot be computed (e.g., only 1 class present).
    """
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return float("nan")

    # Binarize labels in the order of model classes
    Y = label_binarize(y_true, classes=classes)
    # roc_auc_score expects shape (n_samples, n_classes)
    try:
        return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


@torch.no_grad()
def compute_discriminability_stats(
    model: nn.Module,
    dl_source: DataLoader,
    dl_target: DataLoader,
    device: torch.device,
    feature: str,
    max_iter: int = 2000,
) -> Dict[str, float]:
    """
    Computes the 6 stats shown in the table for ONE feature type.

    feature:
      - "mu_act"  (usually class/task segment g_y)
      - "mu_subj" (usually domain/subject segment g_d)

    Returns:
      {
        "Acc/D-S": ...,
        "Acc/D-T": ...,
        "Acc/C-S": ...,
        "Acc/C-T": ...,
        "AUC/C-S": ...,
        "AUC/C-T": ...,
      }
    """
    # Extract latents for each domain
    src = extract_latents(model, dl_source, device)
    tgt = extract_latents(model, dl_target, device)

    Xs = src[feature]
    Xt = tgt[feature]
    ys = src["y"]
    yt = tgt["y"]

    # -------------------------
    # 1) Domain discriminability: predict domain (0=source, 1=target)
    # Train on BOTH domains (so both classes exist), then report accuracy on each domain.
    # -------------------------
    X_dom = np.concatenate([Xs, Xt], axis=0)
    y_dom = np.concatenate(
        [np.zeros(len(Xs), dtype=np.int64), np.ones(len(Xt), dtype=np.int64)],
        axis=0,
    )

    dom_clf, _ = _fit_lr(X_dom, y_dom, max_iter=max_iter, multi_class="auto")

    acc_d_s = float(dom_clf.score(Xs, np.zeros(len(Xs), dtype=np.int64)))
    acc_d_t = float(dom_clf.score(Xt, np.ones(len(Xt), dtype=np.int64)))

    # -------------------------
    # 2) Class/task discriminability: predict activity y
    # Typical DA protocol: train classifier on SOURCE labels, evaluate on source and target.
    # -------------------------
    # If source accidentally has only one class, we can’t train; return NaNs.
    if len(np.unique(ys)) < 2:
        acc_c_s = float("nan")
        acc_c_t = float("nan")
        auc_c_s = float("nan")
        auc_c_t = float("nan")
    else:
        cls_clf, cls_classes = _fit_lr(Xs, ys, max_iter=max_iter, multi_class="multinomial")

        acc_c_s = float(cls_clf.score(Xs, ys))
        acc_c_t = float(cls_clf.score(Xt, yt))

        # Probas for AUC
        proba_s = cls_clf.predict_proba(Xs)
        proba_t = cls_clf.predict_proba(Xt)

        auc_c_s = _safe_macro_auc_multiclass(ys, proba_s, cls_classes)
        auc_c_t = _safe_macro_auc_multiclass(yt, proba_t, cls_classes)

    return {
        "Acc/D-S": acc_d_s,
        "Acc/D-T": acc_d_t,
        "Acc/C-S": acc_c_s,
        "Acc/C-T": acc_c_t,
        "AUC/C-S": auc_c_s,
        "AUC/C-T": auc_c_t,
    }


@torch.no_grad()
def discriminability_table_for_model(
    model: nn.Module,
    dl_source: DataLoader,
    dl_target: DataLoader,
    device: torch.device,
    max_iter: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience wrapper to compute the table row for BOTH segments:
      - using mu_subj (domain/subject segment g_d)
      - using mu_act  (class/task segment g_y)

    Returns:
      {
        "mu_subj": { six stats... },
        "mu_act":  { six stats... }
      }
    """
    return {
        "mu_subj": compute_discriminability_stats(
            model, dl_source, dl_target, device, feature="mu_subj", max_iter=max_iter
        ),
        "mu_act": compute_discriminability_stats(
            model, dl_source, dl_target, device, feature="mu_act", max_iter=max_iter
        ),
    }
