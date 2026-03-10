"""
src/analysis/hctsa/extractor.py

Feature extraction utilities using pycatch22.

No MATLAB. No hctsa toolbox. Pure Python catch22 features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# -----------------------------
# pycatch22 backend
# -----------------------------
try:
    from pycatch22 import catch22_all
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pycatch22 is required. Install with: pip install pycatch22"
    ) from e


FeatFn = Callable[[np.ndarray], Tuple[np.ndarray, List[str]]]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractConfig:
    channels: Tuple[int, ...] = (0, 1, 2)
    channel_names: Tuple[str, ...] = ("x", "y", "z")
    include_magnitude: bool = True
    standardize: bool = False
    eps: float = 1e-8
    max_windows: Optional[int] = None
    verbose: bool = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _ensure_window_shape(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Expected (T,C) or (C,T), got {window.shape}")

    T, C = window.shape
    if C <= 8:
        return window
    if T <= 8:
        return window.T
    return window


def _standardize_ts(ts: np.ndarray, eps: float) -> np.ndarray:
    mu = float(np.mean(ts))
    sd = float(np.std(ts))
    if not np.isfinite(sd) or sd < eps:
        return ts - mu
    return (ts - mu) / (sd + eps)


def _extract_series_per_channel(
    window: np.ndarray,
    channels: Sequence[int],
    channel_names: Sequence[str],
    include_magnitude: bool,
    standardize: bool,
    eps: float,
) -> List[Tuple[str, np.ndarray]]:

    window = _ensure_window_shape(window)
    T, C = window.shape

    if len(channel_names) != len(channels):
        raise ValueError("channel_names length mismatch")

    selected = window[:, list(channels)].astype(np.float64, copy=False)

    series = []
    for name, idx in zip(channel_names, range(selected.shape[1])):
        ts = selected[:, idx]
        if standardize:
            ts = _standardize_ts(ts, eps)
        series.append((name, ts))

    if include_magnitude:
        mag = np.linalg.norm(selected, axis=1)
        if standardize:
            mag = _standardize_ts(mag, eps)
        series.append(("mag", mag))

    return series


def _flatten_feature_names(per_channel_names):
    out = []
    for ch, names in per_channel_names:
        out.extend([f"{ch}__{n}" for n in names])
    return out


# ---------------------------------------------------------------------
# pycatch22 feature fn
# ---------------------------------------------------------------------

def make_pycatch22_feature_fn() -> FeatFn:
    """
    Returns feature_fn(ts) using pycatch22.
    """

    def _fn(ts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        r = catch22_all(ts.tolist())
        return np.asarray(r["values"], dtype=float), list(r["names"])

    return _fn


# ---------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------

def extract_hctsa_from_loader(
    dataloader: Any,
    *,
    feature_fn: Optional[FeatFn] = None,
    config: ExtractConfig = ExtractConfig(),
) -> Dict[str, Any]:

    if feature_fn is None:
        feature_fn = make_pycatch22_feature_fn()

    X_rows = []
    meta_y, meta_s, meta_idx = [], [], []
    meta_batch, meta_inbatch = [], []

    feat_names_total = None
    n_fail = 0
    n_processed = 0
    running_idx = 0

    for b, batch in enumerate(dataloader):

        if isinstance(batch, dict):
            X = batch["X"]
            y = batch["y"]
            s = batch["s"]
            idx = batch.get("idx", None)
        else:
            X, y, s = batch[:3]
            idx = batch[3] if len(batch) >= 4 else None

        X_np = _to_numpy(X)
        y_np = _to_numpy(y).reshape(-1)
        s_np = _to_numpy(s).reshape(-1)

        B = X_np.shape[0]
        idx_np = (
            np.arange(running_idx, running_idx + B)
            if idx is None
            else _to_numpy(idx).reshape(-1)
        )

        for i in range(B):

            if config.max_windows and n_processed >= config.max_windows:
                break

            try:
                window = _ensure_window_shape(X_np[i])

                series_list = _extract_series_per_channel(
                    window,
                    config.channels,
                    config.channel_names,
                    config.include_magnitude,
                    config.standardize,
                    config.eps,
                )

                feats_all = []
                names_all = []

                for ch_name, ts in series_list:
                    feats, names = feature_fn(ts)
                    feats_all.append(feats)
                    names_all.append((ch_name, names))

                row = np.concatenate(feats_all)

                if feat_names_total is None:
                    feat_names_total = _flatten_feature_names(names_all)

                X_rows.append(row)
                meta_y.append(int(y_np[i]))
                meta_s.append(int(s_np[i]))
                meta_idx.append(int(idx_np[i]))
                meta_batch.append(b)
                meta_inbatch.append(i)

            except Exception as e:
                n_fail += 1
                if config.verbose:
                    print("feature fail:", e)

            n_processed += 1

        running_idx += B

    X_feat = np.vstack(X_rows)

    return {
        "X_feat": X_feat,
        "feat_names": feat_names_total,
        "window_meta": {
            "y": np.array(meta_y),
            "s": np.array(meta_s),
            "idx": np.array(meta_idx),
            "batch_idx": np.array(meta_batch),
            "item_in_batch": np.array(meta_inbatch),
        },
        "status": {
            "num_windows": X_feat.shape[0],
            "num_features": X_feat.shape[1],
            "num_fail_windows": n_fail,
        },
    }
