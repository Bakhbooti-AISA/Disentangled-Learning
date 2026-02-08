"""
src/analysis/hctsa/extractor.py

HCTSA feature extraction utilities.

Design goals:
- No file I/O here (no .npz loading, no caching). Scripts handle that.
- Deterministic: assumes the DataLoader is created with shuffle=False (recommended).
- Backend-agnostic: you provide a `feature_fn(ts)->(feat_vec, feat_names)`.

Why this design?
- hctsa is commonly MATLAB-based and heavy; keeping it behind a callable makes
  the rest of your pipeline clean and testable.
- You can swap in a MATLAB runner, a Python port, or a smaller feature set later.

Typical usage (in your script):

    splits = load_splits_npz("data/processed/wisdm_w50_s12_p.npz")
    dl_train, dl_val, dl_test, _ = make_loaders_from_splits(splits, batch_size=128, ...)

    # Build a feature function (example: MATLAB runner)
    feature_fn = make_matlab_hctsa_feature_fn(
        hctsa_dir="/path/to/hctsa",
        matlab_entrypoint="hctsa_feature_vector"  # a MATLAB function you provide
    )

    out = extract_hctsa_from_loader(
        dl_test,
        feature_fn=feature_fn,
        include_magnitude=True,
        channels=(0, 1, 2),
        channel_names=("x", "y", "z"),
    )

    # out["X_feat"] is (N, F_total), out["feat_names"] is list[str]

MATLAB entrypoint contract
--------------------------
You provide a MATLAB function on the MATLAB path with signature:

    function [feat_vec, feat_names] = hctsa_feature_vector(ts)
        % ts: column vector double (T x 1)
        % feat_vec: row vector double (1 x F)
        % feat_names: cell array of strings (1 x F)
    end

This lets you implement the actual hctsa logic in MATLAB while keeping Python clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

FeatFn = Callable[[np.ndarray], Tuple[np.ndarray, List[str]]]


@dataclass(frozen=True)
class ExtractConfig:
    """
    Configuration for extraction.

    channels: indices of channels to extract from window shaped (T, C)
    channel_names: names matching channels, used for prefixing feature names
    include_magnitude: also compute features on ||a|| (Euclidean norm across selected channels)
    standardize: whether to standardize the time series before feature extraction
    eps: numerical stability for standardization
    max_windows: if set, stop after processing this many windows
    verbose: print progress
    """
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
    """Convert torch Tensor or numpy array to numpy array (CPU)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    # Try array conversion for lists etc.
    return np.asarray(x)


def _ensure_window_shape(window: np.ndarray) -> np.ndarray:
    """
    Ensure window is shaped (T, C).
    Accepts (C, T) and will transpose if needed when C is small.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected window with 2 dims (T,C) or (C,T), got shape {window.shape}")

    T, C = window.shape
    if C <= 8:  # common for inertial (3-9 channels)
        return window  # assume (T, C)
    # If C is large and T is small, it might be (C, T)
    if T <= 8:
        return window.T
    return window


def _standardize_ts(ts: np.ndarray, eps: float) -> np.ndarray:
    """Standardize a 1D array."""
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
    """
    Returns list of (series_name, series_1d).
    """
    window = _ensure_window_shape(window)

    T, C = window.shape
    if any(ch < 0 or ch >= C for ch in channels):
        raise ValueError(f"Channel indices {channels} out of bounds for window with C={C}")

    if len(channel_names) != len(channels):
        raise ValueError("channel_names must have same length as channels")

    series: List[Tuple[str, np.ndarray]] = []
    selected = window[:, list(channels)].astype(np.float64, copy=False)

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


def _flatten_feature_names(per_channel_names: List[Tuple[str, List[str]]]) -> List[str]:
    """
    Prefix feature names with channel name: e.g. "x__CO_f1", "mag__EN_f2"
    """
    out: List[str] = []
    for ch_name, feat_names in per_channel_names:
        out.extend([f"{ch_name}__{fn}" for fn in feat_names])
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def extract_hctsa_from_loader(
    dataloader: Any,
    *,
    feature_fn: FeatFn,
    config: ExtractConfig = ExtractConfig(),
) -> Dict[str, Any]:
    """
    Extract (hctsa or hctsa-like) features from windows produced by a DataLoader.

    Expected batch formats:
      - (X, y, s)
      - (X, y, s, idx)
      - or dict-like with keys containing X/y/s/idx

    X is expected to be shaped (B, T, C) or (B, C, T); each window will be converted to (T, C).

    Returns:
      dict with:
        - X_feat: np.ndarray (N, F_total)
        - feat_names: list[str]
        - window_meta: dict arrays: y, s, idx, batch_idx, item_in_batch
        - status: dict with counters, nan ratios, etc.
    """
    if feature_fn is None:
        raise ValueError("feature_fn must be provided (callable that maps 1D ts -> (feat_vec, feat_names))")

    X_rows: List[np.ndarray] = []
    meta_y: List[int] = []
    meta_s: List[int] = []
    meta_idx: List[int] = []
    meta_batch: List[int] = []
    meta_inbatch: List[int] = []

    feat_names_total: Optional[List[str]] = None
    n_fail_windows = 0
    n_processed = 0

    # A running idx when dataset does not provide idx
    running_idx = 0

    for b, batch in enumerate(dataloader):
        # Unpack batch
        if isinstance(batch, dict):
            X = batch.get("X", batch.get("x", None))
            y = batch.get("y", None)
            s = batch.get("s", batch.get("subject", None))
            idx = batch.get("idx", batch.get("index", None))
        else:
            # tuple/list
            if not isinstance(batch, (tuple, list)):
                raise ValueError("Batch must be tuple/list or dict-like")
            if len(batch) < 3:
                raise ValueError("Batch must contain at least (X, y, s)")
            X, y, s = batch[0], batch[1], batch[2]
            idx = batch[3] if len(batch) >= 4 else None

        X_np = _to_numpy(X)
        y_np = _to_numpy(y).reshape(-1)
        s_np = _to_numpy(s).reshape(-1)

        if X_np.ndim != 3:
            raise ValueError(f"Expected X to have shape (B,T,C) or (B,C,T), got {X_np.shape}")

        B = X_np.shape[0]

        if idx is None:
            idx_np = np.arange(running_idx, running_idx + B, dtype=np.int64)
        else:
            idx_np = _to_numpy(idx).reshape(-1).astype(np.int64, copy=False)

        # Iterate items in batch
        for i in range(B):
            if config.max_windows is not None and n_processed >= config.max_windows:
                break

            window = X_np[i]
            # normalize shape to (T,C)
            window = _ensure_window_shape(window)

            try:
                series_list = _extract_series_per_channel(
                    window,
                    channels=config.channels,
                    channel_names=config.channel_names,
                    include_magnitude=config.include_magnitude,
                    standardize=config.standardize,
                    eps=config.eps,
                )

                per_channel_feats: List[np.ndarray] = []
                per_channel_names: List[Tuple[str, List[str]]] = []

                for ch_name, ts in series_list:
                    feats, names = feature_fn(ts.astype(np.float64, copy=False))
                    feats = np.asarray(feats, dtype=np.float64).reshape(-1)
                    if feats.size == 0:
                        raise RuntimeError(f"feature_fn returned empty features for channel {ch_name}")
                    if not isinstance(names, (list, tuple)) or len(names) != feats.size:
                        raise RuntimeError(
                            f"feature_fn must return (feat_vec, feat_names) with matching lengths; "
                            f"got feats={feats.size}, names={len(names) if isinstance(names,(list,tuple)) else 'n/a'}"
                        )
                    per_channel_feats.append(feats)
                    per_channel_names.append((ch_name, list(names)))

                row = np.concatenate(per_channel_feats, axis=0)

                # Initialize feature names once (assumes feature_fn returns consistent names)
                if feat_names_total is None:
                    feat_names_total = _flatten_feature_names(per_channel_names)

                X_rows.append(row)
                meta_y.append(int(y_np[i]) if y_np.size > i else int(y_np[-1]))
                meta_s.append(int(s_np[i]) if s_np.size > i else int(s_np[-1]))
                meta_idx.append(int(idx_np[i]))
                meta_batch.append(int(b))
                meta_inbatch.append(int(i))

            except Exception as e:
                n_fail_windows += 1
                # Keep alignment: optionally append NaNs row if names known
                if feat_names_total is not None:
                    X_rows.append(np.full((len(feat_names_total),), np.nan, dtype=np.float64))
                    meta_y.append(int(y_np[i]) if y_np.size > i else int(y_np[-1]))
                    meta_s.append(int(s_np[i]) if s_np.size > i else int(s_np[-1]))
                    meta_idx.append(int(idx_np[i]))
                    meta_batch.append(int(b))
                    meta_inbatch.append(int(i))
                if config.verbose:
                    print(f"[extract_hctsa_from_loader] window failed (batch={b}, i={i}, idx={int(idx_np[i])}): {e}")

            n_processed += 1

        running_idx += B

        if config.max_windows is not None and n_processed >= config.max_windows:
            break

        if config.verbose and (b % 10 == 0):
            print(f"[extract_hctsa_from_loader] processed batches={b+1}, windows={n_processed}, failures={n_fail_windows}")

    if feat_names_total is None:
        raise RuntimeError("No features extracted successfully; check feature_fn and input data.")

    X_feat = np.vstack([r.reshape(1, -1) for r in X_rows]).astype(np.float64, copy=False)

    nan_ratio = float(np.mean(~np.isfinite(X_feat))) if X_feat.size else 0.0

    return {
        "X_feat": X_feat,
        "feat_names": feat_names_total,
        "window_meta": {
            "y": np.asarray(meta_y, dtype=np.int64),
            "s": np.asarray(meta_s, dtype=np.int64),
            "idx": np.asarray(meta_idx, dtype=np.int64),
            "batch_idx": np.asarray(meta_batch, dtype=np.int64),
            "item_in_batch": np.asarray(meta_inbatch, dtype=np.int64),
        },
        "status": {
            "num_windows": int(X_feat.shape[0]),
            "num_features": int(X_feat.shape[1]),
            "num_fail_windows": int(n_fail_windows),
            "nan_ratio": nan_ratio,
            "config": config,
        },
    }


# ---------------------------------------------------------------------
# MATLAB runner factory (optional)
# ---------------------------------------------------------------------

def make_matlab_hctsa_feature_fn(
    *,
    hctsa_dir: str,
    matlab_entrypoint: str = "hctsa_feature_vector",
    add_paths: Optional[Sequence[str]] = None,
    start_matlab: bool = True,
    mode: str = "full",                     # <-- add this
) -> FeatFn:
    """
    Create a feature_fn(ts) that calls into MATLAB.

    Requirements:
      - MATLAB installed
      - hctsa on disk (hctsa_dir)
      - You provide a MATLAB function `matlab_entrypoint` with signature:
            function [feat_vec, feat_names] = hctsa_feature_vector(ts)
        where ts is (T x 1) double.

    Notes:
      - We keep the MATLAB engine alive inside the closure.
      - We import matlab.engine lazily inside the factory.

    If you don't want MATLAB, ignore this and provide your own feature_fn.
    """
    try:
        import matlab.engine  # type: ignore
        import matlab  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MATLAB engine for Python is not available. "
            "Install it or provide a non-MATLAB feature_fn."
        ) from e

    eng = matlab.engine.start_matlab() if start_matlab else None

    # Add required paths
    def _ensure_paths():
        nonlocal eng
        if eng is None:
            eng = matlab.engine.start_matlab()
        eng.addpath(hctsa_dir, nargout=0)
        if add_paths:
            for p in add_paths:
                eng.addpath(p, nargout=0)

    _ensure_paths()

    def _feat_fn(ts: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        nonlocal eng
        if eng is None:
            _ensure_paths()

        ts = np.asarray(ts, dtype=np.float64).reshape(-1, 1)  # MATLAB expects column vector
        ts_ml = matlab.double(ts.tolist())

        # Call user-provided MATLAB function
        # Expect: feat_vec (1xF double), feat_names (1xF cellstr / string array)
        mode_ml = mode  # MATLAB engine will accept Python str as char array
        feat_vec_ml, feat_names_ml = getattr(eng, matlab_entrypoint)(ts_ml, mode_ml, nargout=2)

        # Convert outputs
        feat_vec = np.asarray(feat_vec_ml, dtype=np.float64).reshape(-1)
        # MATLAB cellstr -> list[str]
        if isinstance(feat_names_ml, (list, tuple)):
            feat_names = [str(x) for x in feat_names_ml]
        else:
            # Try to coerce MATLAB types (string array/cell)
            try:
                feat_names = [str(x) for x in list(feat_names_ml)]
            except Exception:
                feat_names = [f"feat_{i}" for i in range(feat_vec.size)]

        return feat_vec, feat_names

    return _feat_fn
