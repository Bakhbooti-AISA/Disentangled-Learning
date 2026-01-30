# src/data/dataset_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
