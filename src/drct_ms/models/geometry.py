"""Deterministic host-side attention geometry, not neural-network inference.

Derived from DRCT's relative-position indexing and calculate_mask functions.
Upstream license: reference/DRCT/LICENSE (MIT, Chia-Ming Lee, 2024).
"""

import numpy as np


def relative_position_index(window_size: int) -> np.ndarray:
    """Row-major ij coordinates, with the same subtraction direction as DRCT."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    coords = np.stack(np.meshgrid(np.arange(window_size), np.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)
    delta = coords[:, :, None] - coords[:, None, :]
    delta += window_size - 1
    return (delta[0] * (2 * window_size - 1) + delta[1]).astype(np.int32)


def shifted_window_mask(height: int, width: int, window_size: int, shift_size: int) -> np.ndarray:
    """Return [nW, W*W, W*W] additive mask, including the upstream shift=0 path.

    Upstream forward recalculates this mask for *all* blocks when input size
    differs from input_resolution. For shift=0 the resulting mask is all zero.
    """
    if window_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("image and window sizes must be positive")
    if height % window_size or width % window_size:
        raise ValueError("height and width must be multiples of window_size")
    if not 0 <= shift_size < window_size:
        raise ValueError("shift_size must be in [0, window_size)")
    labels = np.zeros((1, height, width, 1), dtype=np.int32)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    for row, hs in enumerate(h_slices):
        for col, ws in enumerate(w_slices):
            labels[:, hs, ws, :] = row * 3 + col
    windows = labels.reshape(1, height // window_size, window_size, width // window_size, window_size, 1)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
    delta = windows[:, None, :] - windows[:, :, None]
    return np.where(delta != 0, -100.0, 0.0).astype(np.float32)
