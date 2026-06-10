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

Visit counts are in SECONDS of occupancy (the caller adds dt per tick), which
keeps the novelty signal 1/(1+count) well-conditioned: a cell driven through
once reads ~0.9, a cell camped in for a minute reads ~0.02.

Obstacles are deliberately NOT accumulated here. World-frame obstacle
accumulation inherits every instant's heading error and the errors only add
up (it smeared phantom walls across the grid). Instead the caller passes a
`blocked_fn` built from the CURRENT lidar scan — ego-centric and
instantaneous, like a BEV: a position is blocked if it lies at or beyond the
nearest return in its direction. Nothing to register, nothing to smear.

Decay also bounds the damage from skid-steer odometry drift: by the time the
pose estimate has wandered, the cells it mis-attributes have mostly faded.
"""

from __future__ import annotations

import math
import time
from collections import deque

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
        self._goal: tuple[int, int] | None = None  # committed novel-cell target

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
        self._goal = None

    # ---- writing ------------------------------------------------------------

    def _indices(self, xs: np.ndarray, ys: np.ndarray):
        ix = np.clip((np.round(xs / self.cell_size)).astype(np.int64) + self._half,
                     0, self._n - 1)
        iy = np.clip((np.round(ys / self.cell_size)).astype(np.int64) + self._half,
                     0, self._n - 1)
        return ix, iy

    def visit(self, x: float, y: float, amount: float = 1.0) -> None:
        """Mark presence; `amount` should be the tick dt (counts = seconds)."""
        ix, iy = self._indices(np.asarray([x]), np.asarray([y]))
        self._counts[iy[0], ix[0]] += amount

    # ---- reading ------------------------------------------------------------

    def novelty(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Novelty in (0, 1] per position: 1 = never (recently) visited."""
        ix, iy = self._indices(np.asarray(xs), np.asarray(ys))
        return 1.0 / (1.0 + self._counts[iy, ix])

    def novelty_at(self, x: float, y: float) -> float:
        return float(self.novelty(np.asarray([x]), np.asarray([y]))[0])

    def novel_bearing(self, x: float, y: float, blocked_fn=None,
                      novel_thresh: float = 0.5,
                      max_radius_m: float = 6.0,
                      dist_weight: float = 0.5,
                      clear_cap_m: float = 2.0) -> float | None:
        """World bearing (rad) toward the BEST reachable novel cell.

        BFS from the rover's cell over cells with under `novel_thresh`
        seconds of recent occupancy. This is the long-range pull out of an
        over-visited region: the local kinematic rollout can only see ~1 m,
        but the BFS sees across the whole grid window.

        Candidates are scored, not taken nearest-first. The visit trail is
        one cell wide, so "nearest novel" is almost always a cramped sliver
        right beside the trail or behind the rover — that made the arrow
        cycle between the true frontier and dead space at its back. Score =
        openness - dist_weight * (bfs_dist / max_radius), where openness is
        the brushfire clearance from anything the CURRENT scan shows,
        capped at `clear_cap_m`. A novel cell in wide-open space (an open
        doorway) outranks a nearer one hugging a wall.

        `blocked_fn(xs, ys) -> bool array` (built from the current scan)
        keeps the search from routing through walls the lidar can see right
        now. Returns None if nothing novel is reachable within max_radius_m.

        Goal hysteresis: a re-search every cycle re-picks the *nearest* novel
        cell, and ties at the blob edge make the bearing thrash. So once a
        goal cell is chosen we stay committed to it until it is reached,
        loses its novelty, leaves range, or the current scan says it sits in
        a wall — only then does the BFS run again.
        """
        ix0, iy0 = self._indices(np.asarray([x]), np.asarray([y]))
        sx, sy = int(ix0[0]), int(iy0[0])
        max_r = int(max_radius_m / self.cell_size)

        if self._goal is not None:
            gx0, gy0 = self._goal
            wx = (gx0 - self._half) * self.cell_size
            wy = (gy0 - self._half) * self.cell_size
            d_cells = max(abs(gx0 - sx), abs(gy0 - sy))
            in_wall = (bool(blocked_fn(np.asarray([wx]), np.asarray([wy]))[0])
                       if blocked_fn is not None else False)
            if (d_cells >= 2 and d_cells <= max_r and not in_wall
                    and self._counts[gy0, gx0] < novel_thresh):
                return math.atan2(wy - y, wx - x)
            self._goal = None

        # Pre-evaluate blockage over the BFS window in one vectorized call.
        x_lo, x_hi = max(0, sx - max_r), min(self._n, sx + max_r + 1)
        y_lo, y_hi = max(0, sy - max_r), min(self._n, sy + max_r + 1)
        if blocked_fn is not None:
            gx, gy = np.meshgrid(np.arange(x_lo, x_hi), np.arange(y_lo, y_hi))
            wxs = (gx - self._half) * self.cell_size
            wys = (gy - self._half) * self.cell_size
            blocked_win = blocked_fn(wxs.ravel(), wys.ravel()).reshape(gx.shape)
        else:
            blocked_win = np.zeros((y_hi - y_lo, x_hi - x_lo), dtype=bool)

        # Brushfire: per-cell distance (in cells) to the nearest scan-blocked
        # cell, capped — the openness half of the candidate score. Vectorized
        # chamfer sweeps (each pass propagates one cell in all 4 directions).
        cap = max(2, int(clear_cap_m / self.cell_size))
        clear_win = np.where(blocked_win, 0, cap).astype(np.int32)
        for _ in range(cap - 1):
            n = clear_win.copy()
            n[1:, :] = np.minimum(n[1:, :], clear_win[:-1, :] + 1)
            n[:-1, :] = np.minimum(n[:-1, :], clear_win[1:, :] + 1)
            n[:, 1:] = np.minimum(n[:, 1:], clear_win[:, :-1] + 1)
            n[:, :-1] = np.minimum(n[:, :-1], clear_win[:, 1:] + 1)
            if np.array_equal(n, clear_win):
                break
            clear_win = n

        seen = np.zeros((self._n, self._n), dtype=bool)
        seen[sy, sx] = True
        q = deque([(sx, sy, 0)])
        best = None
        best_score = -1e9
        while q:
            cx, cy, d = q.popleft()
            # Skip trivially-adjacent cells: a bearing to a neighbor we are
            # practically standing on is noise, not direction.
            if d >= 2 and self._counts[cy, cx] < novel_thresh:
                openness = clear_win[cy - y_lo, cx - x_lo] / cap
                s = openness - dist_weight * (d / max(max_r, 1))
                if s > best_score:
                    best_score = s
                    best = (cx, cy)
            if d >= max_r:
                continue
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if x_lo <= nx < x_hi and y_lo <= ny < y_hi and not seen[ny, nx] \
                        and not blocked_win[ny - y_lo, nx - x_lo]:
                    seen[ny, nx] = True
                    q.append((nx, ny, d + 1))
        if best is None:
            return None
        self._goal = best
        wx = (best[0] - self._half) * self.cell_size
        wy = (best[1] - self._half) * self.cell_size
        return math.atan2(wy - y, wx - x)

    # ---- dashboard export -----------------------------------------------------

    def sparse(self, min_count: float = 0.05, max_cells: int = 1500) -> list:
        """Visited cells as [dx_cells, dy_cells, seconds] relative to origin."""
        iy, ix = np.nonzero(self._counts > min_count)
        vals = self._counts[iy, ix]
        if vals.size > max_cells:
            keep = np.argpartition(vals, vals.size - max_cells)[-max_cells:]
            ix, iy, vals = ix[keep], iy[keep], vals[keep]
        return [[int(x - self._half), int(y - self._half), round(float(v), 2)]
                for x, y, v in zip(ix, iy, vals)]
