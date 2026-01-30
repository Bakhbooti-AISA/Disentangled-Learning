# src/data/wisdm.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .windowing import WindowingConfig, sliding_windows, map_activity_to_id


@dataclass(frozen=True)
class WISDMParseConfig:
    """
    Parsing knobs for WISDM v1.1 raw:
      line: user,activity,timestamp,x,y,z;
    """
    expected_hz: float = 20.0
    # If time gap between consecutive samples exceeds this threshold, start a new segment.
    # Default ~ 0.25s in ns (5 samples @ 20Hz).
    gap_ns: int = 250_000_000

    # If timestamps are missing/garbled, we can ignore gap logic by setting gap_ns=None
    use_gap_splitting: bool = True


@dataclass(frozen=True)
class WISDMPrepared:
    X: np.ndarray         # [N,T,C] float32
    y: np.ndarray         # [N] int64
    subjects: np.ndarray  # [N] int64
    label_map: Dict[str, int]


def _iter_wisdm_rows(raw_path: str) -> Iterator[Tuple[int, str, int, float, float, float]]:
    """
    Streaming generator over parsed rows.
    Yields: (user_id, activity, timestamp_ns, x, y, z)
    Skips malformed lines.
    """
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some lines may end with ';'
            if line.endswith(";"):
                line = line[:-1]

            parts = line.split(",")
            if len(parts) != 6:
                continue

            try:
                user = int(parts[0])
                activity = parts[1]
                ts = int(parts[2])
                ax = float(parts[3])
                ay = float(parts[4])
                az = float(parts[5])
            except Exception:
                continue

            yield user, activity, ts, ax, ay, az


def prepare_wisdm_windows(
    *,
    raw_path: str,
    window_cfg: WindowingConfig = WindowingConfig(),
    parse_cfg: WISDMParseConfig = WISDMParseConfig(),
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> WISDMPrepared:
    """
    Reads raw WISDM file, segments by (user, activity, time gaps), then makes sliding windows.

    Returns arrays:
      X: [N, window, 3] float32
      y: [N] int64 (activity id)
      subjects: [N] int64 (user id)
    """
    # Segment accumulator
    cur_user: Optional[int] = None
    cur_act: Optional[str] = None
    cur_last_ts: Optional[int] = None
    cur_samples: List[Tuple[float, float, float]] = []

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []

    def flush_segment():
        nonlocal cur_samples, cur_user, cur_act
        if cur_user is None or cur_act is None:
            cur_samples = []
            return

        seg_len = len(cur_samples)
        if seg_len < window_cfg.min_segment_len:
            cur_samples = []
            return

        seg = np.asarray(cur_samples, dtype=np.float32)  # [T,3]
        wins = sliding_windows(seg, window_cfg.window_size, window_cfg.stride)  # [Nw, W, 3]
        if wins.shape[0] == 0:
            cur_samples = []
            return

        y_id = map_activity_to_id(cur_act)
        ys = np.full((wins.shape[0],), y_id, dtype=np.int64)
        ss = np.full((wins.shape[0],), int(cur_user), dtype=np.int64)

        X_list.append(wins)
        y_list.append(ys)
        s_list.append(ss)

        cur_samples = []

    n_rows = 0
    n_segments = 0

    for user, act, ts, ax, ay, az in _iter_wisdm_rows(raw_path):
        n_rows += 1

        # Decide if we start a new segment
        new_segment = False
        if cur_user is None:
            new_segment = True
        else:
            if user != cur_user or act != cur_act:
                new_segment = True
            elif parse_cfg.use_gap_splitting and cur_last_ts is not None:
                if ts < cur_last_ts:
                    # Non-monotonic timestamp => new segment
                    new_segment = True
                else:
                    if (ts - cur_last_ts) > parse_cfg.gap_ns:
                        new_segment = True

        if new_segment:
            if cur_user is not None:
                flush_segment()
                n_segments += 1

            cur_user = user
            cur_act = act
            cur_last_ts = ts
            cur_samples = [(ax, ay, az)]
        else:
            cur_last_ts = ts
            cur_samples.append((ax, ay, az))

        if verbose and n_rows % 200_000 == 0:
            print(f"[prepare_wisdm_windows] parsed {n_rows:,} rows; segments so far: {n_segments:,}")

    # flush last segment
    flush_segment()
    n_segments += 1

    if len(X_list) == 0:
        raise RuntimeError("No windows produced. Check raw_path and configs.")

    X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)
    subjects = np.concatenate(s_list, axis=0).astype(np.int64, copy=False)

    if verbose:
        print(f"[prepare_wisdm_windows] done.")
        print(f"  rows parsed:     {n_rows:,}")
        print(f"  segments flushed:{n_segments:,}")
        print(f"  windows:         {X.shape[0]:,}")
        print(f"  window shape:    {X.shape[1:]}  (T,C)")
        # class counts
        uniq, cnt = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
        print(f"  label dist (id->count): {dist}")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            subjects=subjects,
        )
        if verbose:
            print(f"[prepare_wisdm_windows] cached to: {cache_path}")

    return WISDMPrepared(X=X, y=y, subjects=subjects, label_map={"WISDM_LABELS": map_activity_to_id.__globals__["LABEL_TO_ID"]})


def load_cached_wisdm(cache_path: str) -> WISDMPrepared:
    """
    Load cached npz created by prepare_wisdm_windows().
    """
    data = np.load(cache_path)
    X = data["X"].astype(np.float32, copy=False)
    y = data["y"].astype(np.int64, copy=False)
    subjects = data["subjects"].astype(np.int64, copy=False)
    return WISDMPrepared(X=X, y=y, subjects=subjects, label_map={"WISDM_LABELS": map_activity_to_id.__globals__["LABEL_TO_ID"]})
