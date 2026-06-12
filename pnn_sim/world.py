"""Procedural 2D worlds for the headless rover sim.

A world is a set of wall segments [x1, y1, x2, y2] — the outer shell of a
house, internal walls with doorway gaps, and box furniture. Segments (not an
occupancy grid) because exact ray-segment intersection is a handful of
vectorized numpy ops: 360 beams x ~100 segments per tick is trivial, and
there is no grid-resolution artifact for the lidar to learn.

Frame convention matches the rover: x forward / CCW-positive headings, and
beam 0 of a scan points along the robot's +x (the safety gate and the
brain's bin layout both assume this).
"""

from __future__ import annotations

import numpy as np


def _box_segments(cx: float, cy: float, w: float, h: float,
                  angle: float = 0.0) -> np.ndarray:
    """Four wall segments of a (rotated) rectangle centered at (cx, cy)."""
    c, s = np.cos(angle), np.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    corners = corners @ np.array([[c, s], [-s, c]]) + [cx, cy]
    segs = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        segs.append([a[0], a[1], b[0], b[1]])
    return np.asarray(segs, dtype=np.float64)


def _wall_with_door(x1, y1, x2, y2, door_at: float, door_w: float) -> list:
    """A wall segment split by a doorway gap. door_at in (0,1) along the wall."""
    p1 = np.array([x1, y1], dtype=np.float64)
    p2 = np.array([x2, y2], dtype=np.float64)
    length = float(np.linalg.norm(p2 - p1))
    if length < door_w * 1.5:
        return [[x1, y1, x2, y2]]
    u = (p2 - p1) / length
    lo = np.clip(door_at * length - door_w / 2.0, 0.1, length - door_w - 0.1)
    a = p1 + u * lo
    b = p1 + u * (lo + door_w)
    return [[p1[0], p1[1], a[0], a[1]], [b[0], b[1], p2[0], p2[1]]]


class World:
    def __init__(self, segments: np.ndarray, bounds: tuple,
                 start_pose: tuple):
        self.segments = np.asarray(segments, dtype=np.float64)  # [M, 4]
        self.bounds = bounds                  # (xmin, ymin, xmax, ymax)
        self.start_pose = start_pose          # (x, y, theta)
        # Precompute segment pieces for the raycaster.
        self._a = self.segments[:, 0:2]                       # [M, 2]
        self._e = self.segments[:, 2:4] - self.segments[:, 0:2]

    # ------------------------------------------------------------------

    def raycast(self, x: float, y: float, angles: np.ndarray,
                max_range: float) -> np.ndarray:
        """Range per beam from (x, y) along world-frame `angles`. [N]"""
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)    # [N, 2]
        q = self._a - np.array([x, y])                            # [M, 2]
        e = self._e
        # Solve p + t d = a + s e per (ray, segment) by Cramer's rule:
        #   t = cross(e, q) / cross(e, d),  s = cross(d, q) / cross(e, d)
        cross_eq = e[:, 0] * q[:, 1] - e[:, 1] * q[:, 0]          # [M]
        cross_ed = (e[None, :, 0] * d[:, None, 1]
                    - e[None, :, 1] * d[:, None, 0])              # [N, M]
        cross_dq = (d[:, None, 0] * q[None, :, 1]
                    - d[:, None, 1] * q[None, :, 0])              # [N, M]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross_eq[None, :] / cross_ed
            s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & np.isfinite(t)
        t = np.where(hit, t, np.inf)
        return np.minimum(t.min(axis=1), max_range)

    def clearance(self, x: float, y: float) -> float:
        """Distance from a point to the nearest wall segment."""
        p = np.array([x, y])
        ap = p - self._a                                          # [M, 2]
        ee = (self._e * self._e).sum(axis=1)                      # [M]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.clip((ap * self._e).sum(axis=1) / np.maximum(ee, 1e-12),
                         0.0, 1.0)
        closest = self._a + tt[:, None] * self._e
        return float(np.sqrt(((p - closest) ** 2).sum(axis=1)).min())


def make_house(rng: np.random.Generator) -> World:
    """Random single-floor house: shell, internal walls with doors, furniture."""
    W = float(rng.uniform(6.0, 11.0))
    H = float(rng.uniform(5.0, 9.0))
    segs: list = [
        [0, 0, W, 0], [W, 0, W, H], [W, H, 0, H], [0, H, 0, 0],
    ]

    # Internal walls (alternating orientation), each with a doorway.
    n_walls = int(rng.integers(1, 4))
    for i in range(n_walls):
        door_w = float(rng.uniform(0.7, 1.0))
        if (i + int(rng.integers(0, 2))) % 2 == 0:
            x = float(rng.uniform(0.25 * W, 0.75 * W))
            segs += _wall_with_door(x, 0, x, H,
                                    float(rng.uniform(0.2, 0.8)), door_w)
        else:
            y = float(rng.uniform(0.25 * H, 0.75 * H))
            segs += _wall_with_door(0, y, W, y,
                                    float(rng.uniform(0.2, 0.8)), door_w)

    # Furniture boxes, kept off the walls a little.
    for _ in range(int(rng.integers(3, 9))):
        bw = float(rng.uniform(0.3, 1.2))
        bh = float(rng.uniform(0.3, 1.2))
        cx = float(rng.uniform(0.5 + bw / 2, W - 0.5 - bw / 2))
        cy = float(rng.uniform(0.5 + bh / 2, H - 0.5 - bh / 2))
        segs += _box_segments(cx, cy, bw, bh,
                              angle=float(rng.uniform(0, np.pi))).tolist()

    world = World(np.asarray(segs), (0.0, 0.0, W, H), (0.0, 0.0, 0.0))

    # Start pose: rejection-sample a spot with decent clearance.
    for _ in range(200):
        x = float(rng.uniform(0.5, W - 0.5))
        y = float(rng.uniform(0.5, H - 0.5))
        if world.clearance(x, y) > 0.35:
            world.start_pose = (x, y, float(rng.uniform(-np.pi, np.pi)))
            break
    return world
