import numpy as np
from pathlib import Path

from src.analysis.hctsa.extractor import (
    extract_hctsa_from_loader,
    ExtractConfig,
)

# your existing data helpers
from src.data.dataset_base import load_splits_npz, make_loaders_from_splits


# -----------------------------
# CONFIG
# -----------------------------

DATA_PATH = "data/processed/wisdm_splits_2_25_w50_st12.npz"
OUT_PATH = "data/features/wisdm_catch22_features_2_25_w50_st12.npz"

BATCH_SIZE = 256


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("Loading splits...")
    splits = load_splits_npz(DATA_PATH)

    print("Building loaders (no shuffle)...")
    dl_train, dl_val, dl_test, _ = make_loaders_from_splits(
        splits,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
    )

    cfg = ExtractConfig(
        include_magnitude=True,   # xyz + |a|
        standardize=False,
        verbose=True,
    )

    print("Extracting catch22 features (TRAIN)...")
    out_tr = extract_hctsa_from_loader(dl_train, config=cfg)

    print("Extracting catch22 features (VAL)...")
    out_va = extract_hctsa_from_loader(dl_val, config=cfg)

    print("Extracting catch22 features (TEST)...")
    out_te = extract_hctsa_from_loader(dl_test, config=cfg)

    print("Saving features...")

    Path("data/features").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_PATH,

        X_tr=out_tr["X_feat"],
        y_tr=out_tr["window_meta"]["y"],
        s_tr=out_tr["window_meta"]["s"],

        X_va=out_va["X_feat"],
        y_va=out_va["window_meta"]["y"],
        s_va=out_va["window_meta"]["s"],

        X_te=out_te["X_feat"],
        y_te=out_te["window_meta"]["y"],
        s_te=out_te["window_meta"]["s"],

        feat_names=np.array(out_tr["feat_names"], dtype=object),
    )

    print("Done.")
    print("Train shape:", out_tr["X_feat"].shape)
    print("Val shape:", out_va["X_feat"].shape)
    print("Test shape:", out_te["X_feat"].shape)


# -----------------------------

if __name__ == "__main__":
    main()
