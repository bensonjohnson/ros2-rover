"""Sequence replay buffer for the predictive-coding world model.

The world model is recurrent (temporal predictive coding), so replaying
isolated frames would discard the very context the dynamics model learns. We
instead store the experience stream *in order* and hand back short contiguous
windows. The trainer replays each window forward — re-deriving the latent state
with the current weights — so a few extra learning passes can be squeezed from
each lived moment without corrupting the live recurrent state.

Each entry is (obs, action_prev): the observation at step t, and the action
that was executed leading into it (the transition's input). Stored as float32
numpy to keep memory tiny.
"""

from __future__ import annotations

import numpy as np


class SequenceReplay:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 2,
                 seed: int = 0):
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._act = np.zeros((capacity, action_dim), dtype=np.float32)
        self._n = 0          # number of valid entries
        self._head = 0       # next write index (ring)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._n

    def append(self, obs: np.ndarray, action_prev: np.ndarray) -> None:
        self._obs[self._head] = obs
        self._act[self._head] = action_prev
        self._head = (self._head + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample_sequence(self, length: int):
        """Return (obs[length, O], act[length, A]) for a contiguous window.

        Returns None if not enough contiguous history is available. To avoid
        crossing the ring's write seam (where order is discontinuous), windows
        are drawn only from the contiguous filled prefix before the head.
        """
        if self._n < length:
            return None
        # While not yet wrapped, [0, _n) is contiguous and in order.
        # Once wrapped, the seam sits at _head; sample from the older contiguous
        # span [_head, capacity) or the newer [0, _head) — whichever fits.
        if self._n < self.capacity:
            start = int(self._rng.integers(0, self._n - length + 1))
            sl = slice(start, start + length)
            return self._obs[sl].copy(), self._act[sl].copy()

        # Buffer full: pick a span that does not straddle _head.
        spans = []
        if self.capacity - self._head >= length:
            spans.append((self._head, self.capacity - length))
        if self._head >= length:
            spans.append((0, self._head - length))
        if not spans:
            return None
        lo, hi = spans[int(self._rng.integers(0, len(spans)))]
        start = int(self._rng.integers(lo, hi + 1))
        sl = slice(start, start + length)
        return self._obs[sl].copy(), self._act[sl].copy()
