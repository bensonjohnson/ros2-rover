"""FrameStacker — rover-side fixed-length history buffer for RLPD observations.

Maintains the last `k` observations of a single stream (rgb, bev, or proprio)
so the rover can build the stacked tensors expected by the RLPD ONNX/RKNN
graph. On `reset()` (called when `is_first=True` or a fresh RKNN model is
swapped in), the buffer is zero-filled so the policy never sees stale frames
from a previous episode/model.

Stacked frames are concatenated along axis 0 of the frame shape, matching the
server-side encoder layout:
    rgb_stack    : (3*k, 84, 84)
    bev_stack    : (2*k, 128, 128)
    proprio_stack: (6*k,)   (flat — internally reshaped through (k, 6) for stacking)

This module has no ROS or torch dependencies — pure numpy — so it's safe to
import from a launch-time context.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class FrameStacker:
    """Fixed-size ring buffer of past frames.

    Args:
        k: stack depth (number of past frames, including the current one).
        shape: per-frame shape, e.g. (3, 84, 84) for RGB or (6,) for proprio.
        dtype: numpy dtype to store frames in. Match the rover wire dtype:
            uint8 for rgb/bev, float32 for proprio.
    """

    def __init__(self, k: int, shape: Tuple[int, ...], dtype=np.float32):
        self.k = k
        self.shape = tuple(shape)
        self.dtype = dtype
        self._buf: deque = deque(maxlen=k)
        self.reset()

    def reset(self) -> None:
        """Zero-fill all `k` slots. Call on episode boundary / model swap."""
        self._buf.clear()
        for _ in range(self.k):
            self._buf.append(np.zeros(self.shape, dtype=self.dtype))

    def push(self, frame: np.ndarray) -> None:
        """Append `frame` to the buffer; oldest entry is evicted automatically."""
        if frame.shape != self.shape:
            raise ValueError(f"FrameStacker expected {self.shape}, got {frame.shape}")
        # Cast on push so the stored buffer always has the declared dtype
        self._buf.append(frame.astype(self.dtype, copy=False))

    def get_stacked(self) -> np.ndarray:
        """Return the stacked view.

        For frame shape with rank ≥ 1, concatenates along axis 0
        (so (3,84,84) frames produce (3*k, 84, 84)).
        For rank-1 frames (proprio), concatenates and returns a flat (k*d,) vector.
        """
        return np.concatenate(list(self._buf), axis=0)
