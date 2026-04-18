"""Rover-side episodic novelty (NGU's episodic component, Badia et al. 2020).

Maintains a ring buffer of recent encoder embeddings over the current
episode; the per-step reward is a k-NN Gaussian-kernel pseudo-count
that *discourages loitering in recently-visited latent states*. The
buffer is cleared on `is_first` so the signal is genuinely episodic.

The complementary *lifetime* side of NGU lives on the server as a
SimHash over the same embeddings, and is multiplied in at the imagination
loss — rover never touches it.

Cheap to run on RK3588: at 30 Hz with buffer=256 and 1024-dim embeddings
the query is ~0.3 ms (one matmul + softmax-like reduction).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class EpisodicNovelty:
    """Ring-buffer k-NN pseudo-count novelty.

    Reward formula (Badia et al. 2020, eq. 2, simplified):

        r_epi = 1 / sqrt( 1 + Σ_k K(z, z_k) )

    where `K` is a Gaussian kernel over a running memory of the last
    `buffer_size` embeddings. Reward is in (0, 1]: 1.0 when the memory
    is empty or the current embedding is far from anything seen, ~0
    when the current embedding is already well-covered.
    """

    def __init__(
        self,
        embed_dim: int,
        buffer_size: int = 256,
        k: int = 10,
        kernel_epsilon: float = 1e-3,
        min_distance: float = 1e-3,
    ):
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.k = k
        self.kernel_epsilon = kernel_epsilon
        self.min_distance = min_distance
        self._buf: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._running_dist_mean: Optional[float] = None

    def reset(self) -> None:
        """Clear the episodic memory. Must be called on `is_first`."""
        self._buf.clear()
        self._running_dist_mean = None

    def step(self, embed: np.ndarray) -> float:
        """Record `embed` in the memory and return its novelty reward in [0, 1]."""
        e = np.asarray(embed, dtype=np.float32).reshape(-1)
        if e.shape[0] != self.embed_dim:
            raise ValueError(
                f"EpisodicNovelty: expected embed dim {self.embed_dim}, got {e.shape[0]}"
            )

        if not self._buf:
            self._buf.append(e)
            return 1.0  # First observation of the episode is maximally novel.

        mem = np.stack(self._buf, axis=0)  # (N, D)
        diffs = mem - e
        sq_dists = np.einsum('nd,nd->n', diffs, diffs)
        # k-NN
        if sq_dists.shape[0] > self.k:
            idx = np.argpartition(sq_dists, self.k)[:self.k]
            sq_dists = sq_dists[idx]

        # Adaptive kernel bandwidth: running mean of squared distances.
        mean_sq = float(sq_dists.mean())
        if self._running_dist_mean is None:
            self._running_dist_mean = mean_sq
        else:
            self._running_dist_mean = 0.99 * self._running_dist_mean + 0.01 * mean_sq
        bandwidth = max(self._running_dist_mean, self.min_distance)

        # Normalized Gaussian kernel → pseudo-count
        norm_sq = sq_dists / bandwidth
        kernel = self.kernel_epsilon / (norm_sq + self.kernel_epsilon)
        pseudo_count = float(kernel.sum())
        reward = 1.0 / float(np.sqrt(1.0 + pseudo_count))

        # Only add sufficiently different embeddings — prevents the buffer from
        # collapsing to a single state when the rover is stopped.
        nearest_sq = float(sq_dists.min())
        if nearest_sq > (self.min_distance * self.min_distance):
            self._buf.append(e)

        return reward
