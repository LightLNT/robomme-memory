"""Host-side sampling plans for RPM multi-resolution memory."""

import numpy as np


GROUPS = np.array(
    [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
    dtype=np.int32,
)


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if k == 0:
        return np.empty((0,), dtype=np.int32)
    if not 0 < k <= n:
        raise ValueError((n, k))
    return np.linspace(0, n - 1, k, dtype=np.int32)


def make_plan(available_ids, cutoff, mode="mixed", max_frames=56, budget=512):
    """Build a fixed-shape memory plan from available historical frame IDs.

    ``mixed`` keeps all frames fine-grained through 32 frames, then gradually
    allocates frames to coarse representation. ``warm`` and ``baseline`` use
    the same maximum 32-frame timeline as the original teacher/baseline.
    """
    ids = np.asarray(available_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("history IDs must be one-dimensional and unique")
    ids = np.sort(ids[ids <= cutoff])

    if mode not in ("mixed", "warm", "baseline"):
        raise ValueError(mode)
    limit = max_frames if mode == "mixed" else 32
    n = min(len(ids), limit)
    selected = ids[_uniform_indices(len(ids), n)] if n else ids

    if mode == "baseline" or (mode == "mixed" and n <= 32):
        n_high = n
    elif mode == "warm":
        n_high = min(24, n)
    else:
        n_high = min(n, (budget - 4 * n) // 12)
    high_rows = set(_uniform_indices(n, n_high).tolist())

    frame_ids = np.full(max_frames, -1, dtype=np.int64)
    frame_ids[:n] = selected
    frame_valid = np.arange(max_frames) < n
    frame_high = np.zeros(max_frames, dtype=np.bool_)
    gather, mass, kpos = [], [], []

    for frame_idx in range(n):
        if frame_idx in high_rows:
            frame_high[frame_idx] = True
            gather.extend(frame_idx * 16 + token_idx for token_idx in range(16))
            mass.extend([1.0] * 16)
            kpos.extend(frame_idx * 16 + token_idx for token_idx in range(16))
        else:
            gather.extend(max_frames * 16 + frame_idx * 4 + region_idx for region_idx in range(4))
            mass.extend([4.0] * 4)
            kpos.extend(
                frame_idx * 16 + float(GROUPS[region_idx].mean())
                for region_idx in range(4)
            )

    token_count = len(gather)
    if token_count > budget:
        raise ValueError("memory plan exceeds the read budget")
    padding = budget - token_count

    return {
        "frame_ids": frame_ids,
        "frame_valid": frame_valid,
        "frame_high": frame_high,
        "mem_gather": np.asarray(gather + [0] * padding, dtype=np.int32),
        "mem_mask": np.arange(budget) < token_count,
        "mem_mass": np.asarray(mass + [1.0] * padding, dtype=np.float32),
        "mem_kpos": np.asarray(kpos + [0.0] * padding, dtype=np.float32),
        "mem_qoffset": np.asarray(max(512, 16 * n), dtype=np.float32),
    }
