import numpy as np
import argparse

def corr_independence(Z: np.ndarray):
    C = np.corrcoef(Z, rowvar=False)  # (D,D)
    D = C.shape[0]
    mask = ~np.eye(D, dtype=bool)
    abs_off = np.abs(C[mask])
    return C, {
        "mean_abs_corr_offdiag": float(abs_off.mean()),
        "median_abs_corr_offdiag": float(np.median(abs_off)),
        "max_abs_corr_offdiag": float(abs_off.max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_npz", default="data/probes/wisdm_latents_val_test.npz")
    ap.add_argument("--split", choices=["val","test","val_test"], default="test")
    args = ap.parse_args()

    d = np.load(args.latents_npz, allow_pickle=True)

    if args.split == "val":
        Z = d["Z_va"]
    elif args.split == "test":
        Z = d["Z_te"]
    else:
        Z = np.concatenate([d["Z_va"], d["Z_te"]], axis=0)

    C, summary = corr_independence(Z)
    print("Latent independence summary:", summary)
    print("Correlation matrix:\n", np.round(C, 3))

if __name__ == "__main__":
    main()
