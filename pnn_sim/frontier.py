"""Local occupancy + frontier explorer — the exploration drive that works.

Replaces the dead place-novelty curiosity (the FFT-openness place sensor can't
separate real rooms; see active_inference/place_memory). Pure-reactive
gap-following also fails: it chases visible openness (the room CENTRE) and
forgets a doorway the instant it turns away, so it never threads doors
(measured: 0 doorway crossings across 10 sim houses, even with 28 m of travel).

The missing ingredient is a little spatial MEMORY. This builds a local
occupancy grid in the odometry frame, finds FRONTIERS (free cells bordering
unknown = what's beyond a doorway), and pure-pursuits the nearest reachable
one. That persistent target is what lets it navigate to a door it isn't
currently facing. It is NOT SLAM: a rolling local grid, no loop closure, no
global map — and it is drift-tolerant because the target is always a *nearby*
frontier reached in seconds (validated: ~unchanged at 5 deg/s yaw drift).

Pose comes from fused odometry (wheel + IMU + rf2o lidar odom via
robot_localization on the rover; ground truth in sim). Validated in sim
against GROUND-TRUTH map rooms (World.room_id): ~1.4 rooms/house vs 1.0 (zero
doorway crossings) for every reactive controller tried.

Output each tick: a track command [left, right] in [-1, 1], plus the
frontier target bearing and an "actively exploring" flag — the latter two are
what a brain integration feeds into the curiosity observation channel and the
actor's steering prior.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FrontierExplorer:
    def __init__(self, res: float = 0.15, size_m: float = 24.0,
                 max_range: float = 5.0, inflate: int = 1,
                 reach: float = 0.45, replan_every: int = 8,
                 lookahead: float = 0.6, escape_ticks: int = 14,
                 l_free: float = -0.4, l_occ: float = 0.85, l_clamp: float = 4.0):
        # inflate=1 (0.15 m) is deliberate: inflating walls by the robot radius
        # is enough to keep paths off them, but inflate=2 narrows 0.7-1 m doors
        # below the planner's corridor and it refuses to route through them
        # (measured: 0.4 frac vs 0.57). Doorway traversal needs inflate=1.
        self.res = res
        self.n = int(size_m / res)
        self.half = self.n // 2
        self.max_range = max_range
        self.inflate = inflate
        self.reach = reach
        self.replan_every = replan_every
        self.lookahead = lookahead
        self.escape_ticks = escape_ticks
        self.l_free, self.l_occ, self.l_clamp = l_free, l_occ, l_clamp
        self.lo = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        self.origin: np.ndarray | None = None     # world coord at grid centre
        self.path: list | None = None
        self.k = 0
        self.escape = 0
        self.edir = 1.0

    # ---- grid ----------------------------------------------------------

    def _ij(self, x, y):
        return (np.round((np.asarray(x) - self.origin[0]) / self.res)
                .astype(np.int32) + self.half,
                np.round((np.asarray(y) - self.origin[1]) / self.res)
                .astype(np.int32) + self.half)

    def _cell_world(self, c):
        return (self.origin[0] + (c[0] - self.half) * self.res,
                self.origin[1] + (c[1] - self.half) * self.res)

    def update(self, x, y, theta, ranges, bearings):
        """Ray-march the scan into the grid: free along each ray, occupied at
        the hit. `bearings` are robot-relative beam angles (rad, 0 = forward)."""
        if self.origin is None:
            self.origin = np.array([x, y], dtype=np.float64)
        ang = theta + bearings
        cs, sn = np.cos(ang), np.sin(ang)
        rng = np.clip(ranges, 0, self.max_range)
        steps = np.arange(0, self.max_range, self.res)
        px = x + cs[:, None] * steps[None, :]
        py = y + sn[:, None] * steps[None, :]
        ci, cj = self._ij(px, py)
        inb = (ci >= 0) & (ci < self.n) & (cj >= 0) & (cj < self.n)
        free = (steps[None, :] < (rng[:, None] - self.res)) & inb
        np.add.at(self.lo, (ci[free], cj[free]), self.l_free)
        self.seen[ci[free], cj[free]] = True
        hit = ranges < self.max_range
        hi, hj = self._ij(x + cs[hit] * rng[hit], y + sn[hit] * rng[hit])
        ib = (hi >= 0) & (hi < self.n) & (hj >= 0) & (hj < self.n)
        np.add.at(self.lo, (hi[ib], hj[ib]), self.l_occ)
        self.seen[hi[ib], hj[ib]] = True
        np.clip(self.lo, -self.l_clamp, self.l_clamp, out=self.lo)

    def _masks(self):
        occ = self.seen & (self.lo > 0.0)
        free = self.seen & (self.lo <= 0.0)
        unk = ~self.seen
        un = np.zeros_like(unk)
        un[:-1] |= unk[1:]; un[1:] |= unk[:-1]
        un[:, :-1] |= unk[:, 1:]; un[:, 1:] |= unk[:, :-1]
        frontier = free & un
        infl = occ.copy()
        for _ in range(self.inflate):
            nb = infl.copy()
            nb[:-1] |= infl[1:]; nb[1:] |= infl[:-1]
            nb[:, :-1] |= infl[:, 1:]; nb[:, 1:] |= infl[:, :-1]
            infl = nb
        return frontier, infl

    def _plan(self, x, y):
        """BFS over traversable cells to the nearest frontier; return the path
        (list of cells) or None."""
        frontier, infl = self._masks()
        si, sj = self._ij(x, y)
        si, sj = int(si), int(sj)
        if not (0 <= si < self.n and 0 <= sj < self.n):
            return None
        trav = ~infl
        trav[si, sj] = True
        prev = -np.ones((self.n, self.n, 2), np.int32)
        seen = np.zeros((self.n, self.n), bool)
        seen[si, sj] = True
        q = deque([(si, sj)])
        goal = None
        while q:
            i, j = q.popleft()
            if frontier[i, j] and (abs(i - si) + abs(j - sj)) > 2:
                goal = (i, j)
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < self.n and 0 <= b < self.n \
                        and not seen[a, b] and trav[a, b]:
                    seen[a, b] = True
                    prev[a, b] = (i, j)
                    q.append((a, b))
        if goal is None:
            return None
        path = [goal]
        cur = goal
        while tuple(cur) != (si, sj):
            cur = tuple(prev[cur[0], cur[1]])
            if cur == (-1, -1):
                break
            path.append(cur)
        path.reverse()
        return path

    # ---- control -------------------------------------------------------

    def step(self, x, y, theta, ranges, bearings, blocked: bool = False):
        """Advance one tick. Returns (cmd[2], info).

        info = {"target_bearing": rad robot-relative (or None),
                "exploring": bool}. `blocked` should be the safety gate's
        front-clamp state — the authority on whether forward is allowed; when
        set we commit to an escape pivot for several ticks (re-deciding every
        tick just re-wedges against the wall)."""
        self.update(x, y, theta, ranges, bearings)
        self.k += 1

        # Committed escape: pivot toward the open side until clear.
        if self.escape > 0 or blocked:
            if self.escape <= 0:
                la = np.abs((bearings - np.radians(55) + np.pi) % (2 * np.pi)
                            - np.pi) < np.radians(40)
                ra = np.abs((bearings + np.radians(55) + np.pi) % (2 * np.pi)
                            - np.pi) < np.radians(40)
                self.edir = 1.0 if np.mean(ranges[la]) >= np.mean(ranges[ra]) \
                    else -1.0
                self.escape = self.escape_ticks
                self.path = None
            self.escape -= 1
            return (np.array([-0.75 * self.edir, 0.75 * self.edir]),
                    {"target_bearing": None, "exploring": True})

        # (Re)plan to the nearest frontier on schedule / on arrival.
        need = self.path is None or self.k % self.replan_every == 0
        if not need and self.path:
            gx, gy = self._cell_world(self.path[-1])
            if np.hypot(gx - x, gy - y) < self.reach:
                need = True
        if need:
            self.path = self._plan(x, y)

        if not self.path or len(self.path) < 2:
            # No reachable frontier (room/house mapped out): rotate to scan.
            return (np.array([0.3, -0.3]),
                    {"target_bearing": None, "exploring": False})

        # Pure-pursuit a lookahead point along the path.
        wp = self.path[-1]
        for c in self.path:
            wx, wy = self._cell_world(c)
            if np.hypot(wx - x, wy - y) >= self.lookahead:
                wp = c
                break
        wx, wy = self._cell_world(wp)
        tb = (np.arctan2(wy - y, wx - x) - theta + np.pi) % (2 * np.pi) - np.pi
        info = {"target_bearing": float(tb), "exploring": True}

        if abs(tb) > 0.6:                       # pivot in place toward target
            s = 1.0 if tb >= 0 else -1.0
            return np.array([-0.7 * s, 0.7 * s]), info
        turn = float(np.clip(tb / 0.6, -0.8, 0.8))
        fwd = 0.9
        return np.clip(np.array([fwd - turn, fwd + turn]), -1, 1), info

    def reset(self):
        """Forget the map (new building / lift)."""
        self.lo[:] = 0.0
        self.seen[:] = False
        self.origin = None
        self.path = None
        self.escape = 0
