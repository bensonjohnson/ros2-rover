"""Transient spatial memory — a decaying visit-count grid, RAM only.

"Explore new areas" needs at least a short-term memory of where the rover has
recently been; without one, novelty-seeking degenerates into a random walk.
This grid is deliberately NOT a map:

  - it lives only in RAM and dies with the process (never persisted),
  - counts decay exponentially (tau ~ minutes), so it remembers "where I've
    been lately", not "what this building looks like",
  - it is anchored to wherever odometry happened to start this session, so
    dropping the rover in a brand-new place needs no relocalization — the
    grid is simply all-novel there.

Decay also bounds the damage from skid-steer odometry drift: by the time the
pose estimate has wandered, the cells it mis-attributes have mostly faded.
"""

from __future__ import annotations

import time

import numpy as np


class VisitGrid:
    def __init__(self, cell_size: float = 0.25, extent_m: float = 30.0,
                 tau_s: float = 420.0):
        self.cell_size = float(cell_size)
        self.tau_s = float(tau_s)
        n = max(3, int(round(extent_m / cell_size)))
        n += (n + 1) % 2                      # odd, so the origin is a cell center
        self._n = n
        self._half = n // 2
        self._counts = np.zeros((n, n), dtype=np.float32)
        self._last_decay = time.monotonic()

    # ---- maintenance --------------------------------------------------------

    def decay(self) -> None:
        """Apply exponential forgetting for the time since the last call."""
        now = time.monotonic()
        dt = now - self._last_decay
        if dt <= 0.0:
            return
        self._last_decay = now
        self._counts *= np.float32(np.exp(-dt / self.tau_s))

    def clear(self) -> None:
        """Forget everything (used when the rover detects it was picked up)."""
        self._counts.fill(0.0)

    # ---- access -------------------------------------------------------------

    def _indices(self, xs: np.ndarray, ys: np.ndarray):
        ix = np.clip((np.round(xs / self.cell_size)).astype(np.int64) + self._half,
                     0, self._n - 1)
        iy = np.clip((np.round(ys / self.cell_size)).astype(np.int64) + self._half,
                     0, self._n - 1)
        return ix, iy

    def visit(self, x: float, y: float, amount: float = 1.0) -> None:
        ix, iy = self._indices(np.asarray([x]), np.asarray([y]))
        self._counts[iy[0], ix[0]] += amount

    def novelty(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Novelty in (0, 1] per position: 1 = never (recently) visited."""
        ix, iy = self._indices(np.asarray(xs), np.asarray(ys))
        return 1.0 / (1.0 + self._counts[iy, ix])

    def novelty_at(self, x: float, y: float) -> float:
        return float(self.novelty(np.asarray([x]), np.asarray([y]))[0])

    def sparse(self, min_count: float = 0.05, max_cells: int = 1500) -> list:
        """Visited cells as [dx_cells, dy_cells, count] relative to the origin.

        Only meaningfully-visited cells are returned (decay drives stale ones
        under min_count), capped at the max_cells strongest — small enough to
        ship to the dashboard every poll.
        """
        iy, ix = np.nonzero(self._counts > min_count)
        counts = self._counts[iy, ix]
        if counts.size > max_cells:
            keep = np.argpartition(counts, counts.size - max_cells)[-max_cells:]
            ix, iy, counts = ix[keep], iy[keep], counts[keep]
        return [[int(x - self._half), int(y - self._half), round(float(c), 2)]
                for x, y, c in zip(ix, iy, counts)]
