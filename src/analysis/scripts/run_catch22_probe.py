import argparse
import inspect
import os
import re
from pathlib import Path

import numpy as np
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# Your repo helpers
from src.data.dataset_base import load_splits_npz, make_loaders_from_splits


# ----------------------------
# Utilities: infer arch
# ----------------------------

def _get_state_dict(ckpt_obj: dict) -> dict:
    # most common
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]

    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]

    # ✅ your format
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        return ckpt_obj["model_state"]

    # raw state_dict
    if isinstance(ckpt_obj, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj

    raise ValueError(
        f"Unrecognized checkpoint format keys="
        f"{list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else type(ckpt_obj)}"
    )


def infer_arch_from_state_dict(sd: dict) -> dict:
    # latent dim from to_mu
    latent_dim = sd["to_mu.weight"].shape[0]

    # encoder LSTM
    w_ih0 = sd["encoder.lstm.weight_ih_l0"]  # (4H, in)
    enc_hidden = w_ih0.shape[0] // 4
    enc_input = w_ih0.shape[1]
    enc_layers = 1 + max(
        int(m.group(1))
        for k in sd.keys()
        for m in [re.match(r"encoder\.lstm\.weight_ih_l(\d+)$", k)]
        if m
    )

    # decoder LSTM
    dw_ih0 = sd["decoder.lstm.weight_ih_l0"]
    dec_hidden = dw_ih0.shape[0] // 4
    dec_input = dw_ih0.shape[1]
    dec_layers = 1 + max(
        int(m.group(1))
        for k in sd.keys()
        for m in [re.match(r"decoder\.lstm\.weight_ih_l(\d+)$", k)]
        if m
    )

    # head input sizes (often reflects latent split)
    act_in = sd.get("activity_head.net.0.weight", None)
    subj_in = sd.get("subject_head.net.0.weight", None)
    act_in = act_in.shape[1] if act_in is not None else None
    subj_in = subj_in.shape[1] if subj_in is not None else None

    return {
        "latent_dim": int(latent_dim),
        "enc_input": int(enc_input),
        "enc_hidden": int(enc_hidden),
        "enc_layers": int(enc_layers),
        "dec_input": int(dec_input),
        "dec_hidden": int(dec_hidden),
        "dec_layers": int(dec_layers),
        "activity_head_in": None if act_in is None else int(act_in),
        "subject_head_in": None if subj_in is None else int(subj_in),
    }


# ----------------------------
# Utilities: build model
# ----------------------------

def build_model_from_inferred(arch: dict):
    """
    Tries to instantiate your DTSVAE with common arg names.
    If your DTSVAE __init__ signature differs, edit the arg_map below.
    """
    from src.models.dts_vae import DTSVAE  # <-- your model class

    sig = inspect.signature(DTSVAE.__init__)
    params = set(sig.parameters.keys())

    # Common arg name variants across repos
    arg_candidates = {
        "input_dim": arch["enc_input"],
        "in_dim": arch["enc_input"],
        "x_dim": arch["enc_input"],
        "n_in": arch["enc_input"],

        "latent_dim": arch["latent_dim"],
        "z_dim": arch["latent_dim"],

        "hidden_dim": arch["enc_hidden"],
        "hid_dim": arch["enc_hidden"],
        "h_dim": arch["enc_hidden"],

        "num_layers": arch["enc_layers"],
        "n_layers": arch["enc_layers"],

        # sometimes decoder shares same hidden/layers
        "dec_hidden_dim": arch["dec_hidden"],
        "dec_num_layers": arch["dec_layers"],

        # optional latent split
        "z_act_dim": arch["activity_head_in"],
        "z_sub_dim": arch["subject_head_in"],
    }

    kwargs = {}
    for k, v in arg_candidates.items():
        if k in params and v is not None:
            kwargs[k] = v

    try:
        return DTSVAE(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Could not instantiate DTSVAE with inferred args {kwargs}.\n"
            f"Edit build_model_from_inferred() to match your DTSVAE __init__ signature.\n"
            f"Signature: {sig}"
        ) from e


def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _get_state_dict(ckpt)
    cfg = ckpt["cfg"]

    # ---- infer dims from weights (reliable) ----
    latent_dim = sd["to_mu.weight"].shape[0]

    enc_wih0 = sd["encoder.lstm.weight_ih_l0"]
    hidden_dim = enc_wih0.shape[0] // 4
    x_dim = enc_wih0.shape[1]

    import re
    enc_layers = 1 + max(
        int(m.group(1))
        for k in sd.keys()
        for m in [re.match(r"encoder\.lstm\.weight_ih_l(\d+)$", k)]
        if m
    )

    # ---- build encoder/decoder using your config classes ----
    from src.models.dts_vae import DTSVAE
    from src.backbones.lstm import (
        LSTMEncoder, LSTMDecoder,
        LSTMEncoderConfig, LSTMDecoderConfig,
    )

    enc_cfg = LSTMEncoderConfig(
        input_size=x_dim,
        hidden_size=hidden_dim,
        num_layers=enc_layers,
        dropout=0.0,
        bidirectional=False,
    )

    dec_cfg = LSTMDecoderConfig(
        output_size=x_dim,
        latent_dim=latent_dim,
        hidden_size=hidden_dim,
        num_layers=enc_layers,
        dropout=0.0,
    )

    encoder = LSTMEncoder(enc_cfg)
    decoder = LSTMDecoder(dec_cfg)

    model = DTSVAE(cfg, encoder, decoder)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    arch = {
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(enc_layers),
        "input_dim": int(x_dim),
    }

    return model, arch, {}



# ----------------------------
# Utilities: get mu latents
# ----------------------------

@torch.no_grad()
def encode_mu(model, x: torch.Tensor) -> torch.Tensor:
    """
    Returns mu for input batch x (B,T,C) or (B,C,T).
    Tries model.encode -> outputs -> fallback to encoder + to_mu.
    """
    # If model has encode()
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        out = model.encode(x)
        # common patterns
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            mu = out[0]
            return mu
        if isinstance(out, dict) and "mu" in out:
            return out["mu"]

    # If forward returns dict/obj with mu
    out = model(x)
    if isinstance(out, dict) and "mu" in out:
        return out["mu"]
    if hasattr(out, "mu"):
        return out.mu

    # Fallback: encoder -> to_mu
    h = model.encoder(x)
    # h might be (B,T,H); take last timestep
    if isinstance(h, (tuple, list)):
        h = h[0]
    if h.ndim == 3:
        h = h[:, -1, :]
    mu = model.to_mu(h)
    return mu


@torch.no_grad()
def collect_latents(model, dataloader, device: str):
    Z_list = []
    y_list = []
    s_list = []
    idx_list = []

    for batch in dataloader:
        if isinstance(batch, dict):
            X = batch["X"]
            y = batch["y"]
            s = batch["s"]
            idx = batch.get("idx", None)
        else:
            X, y, s = batch[:3]
            idx = batch[3] if len(batch) >= 4 else None

        X = X.to(device)
        mu = encode_mu(model, X).detach().cpu().numpy()

        Z_list.append(mu)
        y_list.append(np.asarray(y))
        s_list.append(np.asarray(s))
        if idx is not None:
            idx_list.append(np.asarray(idx))

    Z = np.concatenate(Z_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    s = np.concatenate(s_list, axis=0)
    idx = np.concatenate(idx_list, axis=0) if len(idx_list) else None
    return Z, y, s, idx


# ----------------------------
# Stats: Spearman corr
# ----------------------------

def spearman_corr_matrix(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Spearman correlation between each latent dim in Z (N,D)
    and each feature in X (N,F). Returns (D,F).
    Pure numpy: rank-transform then Pearson.
    """
    # rank along N for each column
    Zr = np.apply_along_axis(_rank_1d, 0, Z)
    Xr = np.apply_along_axis(_rank_1d, 0, X)

    # center
    Zr = Zr - Zr.mean(axis=0, keepdims=True)
    Xr = Xr - Xr.mean(axis=0, keepdims=True)

    # covariance and std
    cov = (Zr.T @ Xr) / (Zr.shape[0] - 1)
    Zstd = Zr.std(axis=0, ddof=1)
    Xstd = Xr.std(axis=0, ddof=1)

    # avoid divide by zero
    Zstd[Zstd == 0] = np.nan
    Xstd[Xstd == 0] = np.nan

    corr = cov / (Zstd[:, None] * Xstd[None, :])
    return corr


def _rank_1d(a: np.ndarray) -> np.ndarray:
    # average ranks for ties
    temp = a.argsort(kind="mergesort")
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a), dtype=float)

    # tie handling: average ranks for equal values
    sa = a[temp]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (ranks[temp[i]] + ranks[temp[j]]) / 2.0
            ranks[temp[i : j + 1]] = avg
        i = j + 1

    return ranks


def topk_per_latent(corr: np.ndarray, feat_names: list[str], k: int = 10):
    rows = []
    D, F = corr.shape
    for d in range(D):
        order = np.argsort(-np.abs(corr[d]))
        for j in order[:k]:
            rows.append({
                "latent_dim": d,
                "feature": feat_names[j],
                "rho": float(corr[d, j]),
                "abs_rho": float(abs(corr[d, j])),
                "feature_idx": int(j),
            })
    return rows


# ----------------------------
# Linear Probes
# ----------------------------

def select_top_features_global(corr: np.ndarray, feat_names: list[str], top_m: int):
    """
    Pick top_m unique features by max abs correlation over latent dims.
    Returns a list of dict rows with feature_idx, feature_name, max_abs_rho, best_latent_dim, best_rho.
    """
    # corr: (D,F)
    abs_corr = np.abs(corr)
    best_dim = abs_corr.argmax(axis=0)              # (F,)
    best_abs = abs_corr.max(axis=0)                 # (F,)
    best_rho = corr[best_dim, np.arange(corr.shape[1])]

    order = np.argsort(-best_abs)
    rows = []
    for j in order[:top_m]:
        rows.append({
            "feature_idx": int(j),
            "feature": feat_names[j],
            "max_abs_rho": float(best_abs[j]),
            "best_latent_dim": int(best_dim[j]),
            "best_rho": float(best_rho[j]),
        })
    return rows


def fit_linear_probe_and_eval(Z_tr, y_tr_feat, Z_va, y_va_feat, Z_te, y_te_feat, Z_trte, y_trte_feat, Z_all, y_all_feat, alpha: float):
    """
    Ridge regression probe: Standardize Z -> Ridge(alpha).
    Returns metrics dict with R2/MAE across splits.
    """
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=alpha, random_state=0)),
    ])
    model.fit(Z_tr, y_tr_feat)

    def eval_split(Z, y, name):
        pred = model.predict(Z)
        return {
            f"r2_{name}": float(r2_score(y, pred)),
            f"mae_{name}": float(mean_absolute_error(y, pred)),
        }

    out = {}
    out.update(eval_split(Z_tr,   y_tr_feat,   "train"))
    out.update(eval_split(Z_va,   y_va_feat,   "val"))
    out.update(eval_split(Z_te,   y_te_feat,   "test"))
    out.update(eval_split(Z_trte, y_trte_feat, "train_test"))
    out.update(eval_split(Z_all,  y_all_feat,  "all"))
    return out


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/wisdm_splits_2_25_w50_st12.npz")
    ap.add_argument("--features", default="data/features/wisdm_catch22_features_2_25_w50_st12.npz")
    ap.add_argument("--ckpt", default="checkpoints/DTSVAE/ckpt_2_25_epoch_small_1550.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--outdir", default="data/probes")
    ap.add_argument("--rank_on", default="train_test", choices=["train", "val", "test", "train_test", "all"],
                help="Which correlation matrix to use when selecting high-|rho| features.")
    ap.add_argument("--probe_topm", type=int, default=88,
                help="How many highest-|rho| features (global) to linearly probe.")
    ap.add_argument("--ridge_alpha", type=float, default=1.0,
                help="Ridge regularization strength for linear probes.")

    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("Loading model checkpoint...")
    model, arch, load_info = load_model(args.ckpt, device=args.device)
    print("Inferred arch:", arch)
    if load_info.get("missing") or load_info.get("unexpected"):
        print("State dict note:", load_info)

    print("Loading splits and making loaders...")
    splits = load_splits_npz(args.splits)
    dl_train, dl_val, dl_test, _ = make_loaders_from_splits(
        splits, batch_size=args.batch_size, num_workers=2, pin_memory=False
    )

    # IMPORTANT: probe val/test (deterministic). train may be misaligned if features were extracted with shuffle=True.
    print("Collecting latents (VAL)...")
    Z_va, y_va, s_va, idx_va = collect_latents(model, dl_val, device=args.device)

    print("Collecting latents (TEST)...")
    Z_te, y_te, s_te, idx_te = collect_latents(model, dl_test, device=args.device)

    print("Collecting latents (TRAIN)...")
    Z_tr, y_tr, s_tr, idx_tr = collect_latents(model, dl_train, device=args.device)

    print("Loading catch22 features...")
    f = np.load(args.features, allow_pickle=True)
    X_va = f["X_va"]
    X_te = f["X_te"]
    X_tr = f["X_tr"]
    feat_names = list(f["feat_names"])

    # Combine sets
    Z_trte = np.concatenate([Z_tr, Z_te], axis=0)
    y_trte = np.concatenate([y_tr, y_te], axis=0)
    s_trte = np.concatenate([s_tr, s_te], axis=0)
    X_trte = np.concatenate([X_tr, X_te], axis=0)

    Z_all = np.concatenate([Z_tr, Z_va, Z_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
    s_all = np.concatenate([s_tr, s_va, s_te], axis=0)
    X_all = np.concatenate([X_tr, X_va, X_te], axis=0)

    # sanity checks
    if X_va.shape[0] != Z_va.shape[0]:
        raise RuntimeError(f"VAL mismatch: features {X_va.shape[0]} vs latents {Z_va.shape[0]}")
    if X_te.shape[0] != Z_te.shape[0]:
        raise RuntimeError(f"TEST mismatch: features {X_te.shape[0]} vs latents {Z_te.shape[0]}")
    if X_tr.shape[0] != Z_tr.shape[0]:
        raise RuntimeError(f"TRAIN mismatch: features {X_tr.shape[0]} vs latents {Z_tr.shape[0]}")

    print("Computing Spearman correlations (VAL)...")
    corr_va = spearman_corr_matrix(Z_va, X_va)

    print("Computing Spearman correlations (TEST)...")
    corr_te = spearman_corr_matrix(Z_te, X_te)

    print("Computing Spearman correlations (TRAIN)...")
    corr_tr = spearman_corr_matrix(Z_tr, X_tr)

    print("Computing Spearman correlations (TRAIN+TEST)...")
    corr_trte = spearman_corr_matrix(Z_trte, X_trte)

    print("Computing Spearman correlations (ALL)...")
    corr_all = spearman_corr_matrix(Z_all, X_all)

    

    # Save latents
    lat_out = os.path.join(args.outdir, "wisdm_latents_val_test.npz")
    np.savez_compressed(
        lat_out,
        Z_va=Z_va, y_va=y_va, s_va=s_va,
        Z_te=Z_te, y_te=y_te, s_te=s_te,
        Z_tr=Z_tr, y_tr=y_tr, s_tr=s_tr,
        Z_trte=Z_trte, y_trte=y_trte, s_trte=s_trte,
        Z_all=Z_all, y_all=y_all, s_all=s_all,
        feat_names=np.array(feat_names, dtype=object),
        arch=np.array([arch], dtype=object),
    )
    print("Saved latents:", lat_out)

    # Save correlation matrices
    corr_va_path = os.path.join(args.outdir, "corr_val.csv")
    corr_te_path = os.path.join(args.outdir, "corr_test.csv")
    corr_tr_path = os.path.join(args.outdir, "corr_train.csv")
    corr_trte_path = os.path.join(args.outdir, "corr_train_test.csv")
    corr_all_path = os.path.join(args.outdir, "corr_all.csv")

    # Write CSV without pandas dependency
    def write_corr_csv(path, corr):
        with open(path, "w", encoding="utf-8") as w:
            w.write("latent_dim," + ",".join([f'"{n}"' for n in feat_names]) + "\n")
            for d in range(corr.shape[0]):
                row = ",".join(["" if np.isnan(x) else f"{x:.6g}" for x in corr[d]])
                w.write(f"{d}," + row + "\n")

    write_corr_csv(corr_va_path, corr_va)
    write_corr_csv(corr_te_path, corr_te)
    write_corr_csv(corr_tr_path, corr_tr)
    write_corr_csv(corr_trte_path, corr_trte)
    write_corr_csv(corr_all_path, corr_all)
    print("Saved:", corr_va_path)
    print("Saved:", corr_te_path)
    print("Saved:", corr_tr_path)
    print("Saved:", corr_trte_path)
    print("Saved:", corr_all_path)

    # Save top-k tables
    top_va = topk_per_latent(corr_va, feat_names, k=args.topk)
    top_te = topk_per_latent(corr_te, feat_names, k=args.topk)
    top_tr = topk_per_latent(corr_tr, feat_names, k=args.topk)
    top_trte = topk_per_latent(corr_trte, feat_names, k=args.topk)
    top_all = topk_per_latent(corr_all, feat_names, k=args.topk)

    def write_topk(path, rows):
        with open(path, "w", encoding="utf-8") as w:
            w.write("latent_dim,feature_idx,feature,rho,abs_rho\n")
            for r in rows:
                w.write(f'{r["latent_dim"]},{r["feature_idx"]},"{r["feature"]}",{r["rho"]:.6g},{r["abs_rho"]:.6g}\n')

    top_va_path = os.path.join(args.outdir, "topk_val.csv")
    top_te_path = os.path.join(args.outdir, "topk_test.csv")
    top_tr_path = os.path.join(args.outdir, "topk_train.csv")
    top_trte_path = os.path.join(args.outdir, "topk_train_test.csv")
    top_all_path = os.path.join(args.outdir, "topk_all.csv")

    write_topk(top_va_path, top_va)
    write_topk(top_te_path, top_te)
    write_topk(top_tr_path, top_tr)
    write_topk(top_trte_path, top_trte)
    write_topk(top_all_path, top_all)
    print("Saved:", top_va_path)
    print("Saved:", top_te_path)
    print("Saved:", top_tr_path)
    print("Saved:", top_trte_path)
    print("Saved:", top_all_path)


    # ----------------------------
    # Linear feature probing
    # ----------------------------
    corr_map = {
        "train": corr_tr,
        "val": corr_va,
        "test": corr_te,
        "train_test": corr_trte,
        "all": corr_all,
    }
    corr_rank = corr_map[args.rank_on]

    print(f"Selecting top-{args.probe_topm} features by |rho| on set={args.rank_on}...")
    top_feats = select_top_features_global(corr_rank, feat_names, top_m=args.probe_topm)

    probe_rows = []
    print("Running linear probes (Ridge) from Z -> selected features...")

    for info in top_feats:
        j = info["feature_idx"]

        ytr = X_tr[:, j]
        yva = X_va[:, j]
        yte = X_te[:, j]
        ytrte = X_trte[:, j]
        yall = X_all[:, j]

        metrics = fit_linear_probe_and_eval(
            Z_tr, ytr,
            Z_va, yva,
            Z_te, yte,
            Z_trte, ytrte,
            Z_all, yall,
            alpha=args.ridge_alpha,
        )

        row = {**info, **metrics}
        probe_rows.append(row)

    # save probe results
    probe_path = os.path.join(args.outdir, f"linear_probe_{args.rank_on}.csv")
    with open(probe_path, "w", encoding="utf-8") as w:
        w.write(
            "feature_idx,feature,max_abs_rho,best_latent_dim,best_rho,"
            "r2_train,mae_train,r2_val,mae_val,r2_test,mae_test,r2_train_test,mae_train_test,r2_all,mae_all\n"
        )
        for r in probe_rows:
            w.write(
                f'{r["feature_idx"]},"{r["feature"]}",{r["max_abs_rho"]:.6g},{r["best_latent_dim"]},{r["best_rho"]:.6g},'
                f'{r["r2_train"]:.6g},{r["mae_train"]:.6g},{r["r2_val"]:.6g},{r["mae_val"]:.6g},'
                f'{r["r2_test"]:.6g},{r["mae_test"]:.6g},{r["r2_train_test"]:.6g},{r["mae_train_test"]:.6g},'
                f'{r["r2_all"]:.6g},{r["mae_all"]:.6g}\n'
            )

    print("Saved linear probe results:", probe_path)


    print("Done.")


if __name__ == "__main__":
    main()
