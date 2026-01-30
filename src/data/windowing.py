# src/data/windowing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


WISDM_LABELS: Tuple[str, ...] = (
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Downstairs",
)

LABEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(WISDM_LABELS)}
ID_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(WISDM_LABELS)}


@dataclass(frozen=True)
class WindowingConfig:
    window_size: int = 100   # 5 seconds @ 20Hz
    stride: int = 50         # 50% overlap
    min_segment_len: int = 100  # ignore segments shorter than one window


def sliding_windows(
    x: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Create sliding windows from a 2D array x: [T, C] -> windows [N, window_size, C].

    Uses a simple loop to avoid tricky stride issues and keep memory predictable.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x.ndim==2 [T,C], got {x.shape}")

    T = x.shape[0]
    if T < window_size:
        return np.empty((0, window_size, x.shape[1]), dtype=x.dtype)

    starts = range(0, T - window_size + 1, stride)
    windows = np.stack([x[s : s + window_size] for s in starts], axis=0)
    return windows


def map_activity_to_id(activity: str) -> int:
    """
    Maps WISDM activity string to integer class id.
    Raises KeyError if unknown.
    """
    return LABEL_TO_ID[activity]
