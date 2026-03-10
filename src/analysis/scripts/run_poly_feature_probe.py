import argparse, os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

def spearman_select_top_features_global(corr: np.ndarray, feat_names: list[str], top_m: int):
    abs_corr = np.abs(corr)
    best_dim = abs_corr.argmax(axis=0)
    best_abs = abs_corr.max(axis=0)
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

def fit_and_eval(model, Z_tr, y_tr, Z_va, y_va, Z_te, y_te):
    model.fit(Z_tr, y_tr)
    def ev(Z, y):
        p = model.predict(Z)
        return float(r2_score(y, p)), float(mean_absolute_error(y, p))
    r2_tr, mae_tr = ev(Z_tr, y_tr)
    r2_va, mae_va = ev(Z_va, y_va)
    r2_te, mae_te = ev(Z_te, y_te)
    return r2_tr, mae_tr, r2_va, mae_va, r2_te, mae_te

def make_probe(model_kind: str, alpha: float, degree: int):
    steps = [("scaler", StandardScaler())]
    if model_kind.startswith("poly"):
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("ridge", Ridge(alpha=alpha, random_state=0)))
    return Pipeline(steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", default="data/features/wisdm_catch22_features.npz")
    ap.add_argument("--latents_npz", default="data/probes/wisdm_latents_val_test.npz")
    ap.add_argument("--corr_csv", default="data/probes/corr_test.csv", help="Correlation matrix to select features from (CSV produced earlier).")
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--model", choices=["ridge","poly2","poly3"], default="poly2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="data/probes/poly_feature_probe.csv")
    args = ap.parse_args()

    # Load latents (val/test)
    ld = np.load(args.latents_npz, allow_pickle=True)
    Z_va, Z_te = ld["Z_va"], ld["Z_te"]

    # Load features (val/test)
    fd = np.load(args.features_npz, allow_pickle=True)
    X_va, X_te = fd["X_va"], fd["X_te"]
    feat_names = list(fd["feat_names"])

    # Use train_test = val+test for fitting if you prefer; here we train on val, test on test by default
    Z_tr, y_src = Z_va, X_va
    Z_test, y_tgt = Z_te, X_te

    # Read corr matrix for selection (expects corr CSV format from our earlier script)
    import pandas as pd
    corr_df = pd.read_csv(args.corr_csv, index_col=0)
    corr = corr_df.to_numpy()  # (D,F)

    top_feats = spearman_select_top_features_global(corr, feat_names, top_m=args.topm)

    degree = 1
    if args.model == "poly2": degree = 2
    if args.model == "poly3": degree = 3

    probe = make_probe(args.model, alpha=args.alpha, degree=degree)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as w:
        w.write("feature_idx,feature,max_abs_rho,best_latent_dim,best_rho,r2_val,mae_val,r2_test,mae_test\n")
        for info in top_feats:
            j = info["feature_idx"]
            y_tr_feat = y_src[:, j]
            y_te_feat = y_tgt[:, j]

            probe = make_probe(args.model, alpha=args.alpha, degree=degree)
            _, _, r2_va, mae_va, r2_te, mae_te = fit_and_eval(probe, Z_tr, y_tr_feat, Z_tr, y_tr_feat, Z_test, y_te_feat)

            w.write(f'{j},"{info["feature"]}",{info["max_abs_rho"]:.6g},{info["best_latent_dim"]},{info["best_rho"]:.6g},'
                    f'{r2_va:.6g},{mae_va:.6g},{r2_te:.6g},{mae_te:.6g}\n')

    print("Saved:", args.out)

if __name__ == "__main__":
    main()
