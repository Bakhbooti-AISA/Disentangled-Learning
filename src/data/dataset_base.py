# src/data/dataset_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import os
from torch.utils.data import Dataset, dataloader


class WindowDataset(Dataset):
    """
    Generic window dataset.
      X: [N, T, C] float32
      y: [N] int64
      subjects: [N] int64
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        *,
        standardize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        if X.ndim != 3:
            raise ValueError(f"X must be [N,T,C], got {X.shape}")
        if y.ndim != 1 or subjects.ndim != 1:
            raise ValueError("y and subjects must be 1D arrays")

        if len(X) != len(y) or len(X) != len(subjects):
            raise ValueError("X, y, subjects must have same length")

        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.subjects = subjects.astype(np.int64, copy=False)

        self.standardize = standardize
        self.mean = mean
        self.std = std

        if self.standardize and (self.mean is None or self.std is None):
            # Compute per-channel mean/std over ALL windows/timesteps (train set should supply these ideally)
            flat = self.X.reshape(-1, self.X.shape[-1])  # [(N*T), C]
            self.mean = flat.mean(axis=0)
            self.std = flat.std(axis=0) + 1e-8

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.standardize:
            x = (x - self.mean) / self.std

        return (
            torch.from_numpy(x),                 # [T,C] float32
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.subjects[idx], dtype=torch.long),
        )


@dataclass(frozen=True)
class SubjectSplit:
    train_subjects: Tuple[int, ...]
    val_subjects: Tuple[int, ...]
    test_subjects: Tuple[int, ...]


def subject_wise_split(
    subjects: np.ndarray,
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SubjectSplit:
    """
    Split by unique subject IDs (no subject overlap between splits).
    """
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split ratios too large; no subjects left for train.")

    train = tuple(int(s) for s in uniq[:n_train])
    val = tuple(int(s) for s in uniq[n_train : n_train + n_val])
    test = tuple(int(s) for s in uniq[n_train + n_val :])

    return SubjectSplit(train_subjects=train, val_subjects=val, test_subjects=test)


def mask_from_subjects(subjects: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
    allowed = np.asarray(list(allowed), dtype=subjects.dtype)
    return np.isin(subjects, allowed)


def subset_arrays(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X[mask], y[mask], subjects[mask]


def save_splits_npz(path, *,
                    X_tr, y_tr, s_tr,
                    X_va, y_va, s_va,
                    X_te, y_te, s_te,
                    mean, std,
                    meta: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # meta goes in as a numpy object array; allow_pickle=True will be needed when loading
    if meta is None:
        meta = {}

    np.savez_compressed(
        path,
        X_tr=X_tr, y_tr=y_tr, s_tr=s_tr,
        X_va=X_va, y_va=y_va, s_va=s_va,
        X_te=X_te, y_te=y_te, s_te=s_te,
        mean=np.asarray(mean),
        std=np.asarray(std),
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved splits to: {path}")


def load_splits_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item() if "meta" in d else {}

    out = {
        "X_tr": d["X_tr"], "y_tr": d["y_tr"], "s_tr": d["s_tr"],
        "X_va": d["X_va"], "y_va": d["y_va"], "s_va": d["s_va"],
        "X_te": d["X_te"], "y_te": d["y_te"], "s_te": d["s_te"],
        "mean": d["mean"], "std": d["std"],
        "meta": meta,
    }
    return out

def make_loaders_from_splits(splits, *, batch_size=256, num_workers=2, pin_memory=True):
    mean, std = splits["mean"], splits["std"]

    ds_train = WindowDataset(splits["X_tr"], splits["y_tr"], splits["s_tr"],standardize=True, mean=mean, std=std)
    ds_val   = WindowDataset(splits["X_va"], splits["y_va"], splits["s_va"],standardize=True, mean=mean, std=std)
    ds_test  = WindowDataset(splits["X_te"], splits["y_te"], splits["s_te"],standardize=True, mean=mean, std=std)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return dl_train, dl_val, dl_test, (ds_train, ds_val, ds_test)