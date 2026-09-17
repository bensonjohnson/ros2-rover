"""Procedural buildings with explicit rooms and door thresholds.

`make_house` (world.py) only knows full-span partition walls, so every
house is a 1-3 wall slab layout and "room" = which side of each wall.
Buildings are richer and carry their ground truth explicitly:

  * a room LABEL grid on a U = 0.25 m lattice (-1 = outside / void)
  * `room_rects` [R, 5] (x0, y0, x1, y1, room): disjoint axis-aligned
    rectangles tiling every room — the GPU room counter is elementwise
    rect containment (no index/gather kernels: the CUDA-graph fault class)
  * `doors` [D, 6] (x1, y1, x2, y2, room_a, room_b): THRESHOLD segments,
    the open span of wall between two different rooms. A rover centre
    crossing that segment = passing through the doorway.
  * reachability: a 0.05 m free-space raster (clearance > robot radius +
    margin) flood-filled from the start pose; `rooms_reachable` and the
    reachable mask give honest coverage ceilings (a room behind a sofa
    that blocks its only door does not count).

Families (make_building(rng, family=...)):
  bsp      recursive split of a rectangle into 3-9 rooms, one door per
           spanning-tree adjacency + occasional loops
  hall     corridor spine with rooms both sides (apartment/office)
  lshape   bsp inside an L/U/notched footprint
  open     few large rooms joined by wide archways, island furniture
  maze     braided grid maze; rooms = 2x2/3x3 zones of maze cells,
           thresholds = maze openings across zone boundaries
  house    legacy make_house slab layout, converted to rects/doors
  mix      weighted draw over all of the above

Door widths: natural doors 0.75/1.0 m (the lattice rounds the real
0.7-1.0 m range), archways 1.5-2.5 m, maze openings = corridor width
1.0-1.5 m. The rover is 0.28 m wide with a 0.24 m gate corridor, so every
threshold is physically passable; furniture keeps 0.6 m clear of them.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .world import World, _box_segments, make_house

U = 0.25                  # label lattice (m)
ROBOT_R = 0.14            # RoverConfig.robot_radius
FREE_MARGIN = 0.04        # reachability clearance margin beyond radius
RASTER = 0.05             # reachability raster (m)
MAX_SEGMENTS = 128        # raycast cost scales with the padded max
FAMILIES = ("bsp", "hall", "lshape", "open", "maze", "house")
MIX_WEIGHTS = {"bsp": 0.25, "hall": 0.2, "lshape": 0.15, "open": 0.1,
               "maze": 0.15, "house": 0.15}


# Curriculum levels (easy -> production). Level 3 == the "mix" distribution.
LEVELS = {
    0: dict(family="mix", doors="wide", start="near_door", max_rooms=3,
            weights={"bsp": 0.4, "open": 0.3, "house": 0.3}),
    1: dict(family="mix", doors="natural", start="near_door", max_rooms=4,
            weights={"bsp": 0.35, "lshape": 0.15, "open": 0.2,
                     "house": 0.3}),
    2: dict(family="mix", doors="natural", start="random", max_rooms=6,
            weights={"bsp": 0.3, "hall": 0.2, "lshape": 0.2, "open": 0.1,
                     "house": 0.2}),
    3: dict(family="mix", doors="natural", start="random", max_rooms=None,
            weights=MIX_WEIGHTS),
}


def make_level(rng: np.random.Generator, level: int) -> "Building":
    return make_building(rng, **LEVELS[int(level)])


class Building(World):
    """World with explicit room labels, room rectangles and door
    thresholds. room_id() returns an int (-1 outside) instead of the
    partition side-tuple; both are hashable, so set()-based room counters
    (eval_checkpoints, distill_policy) work unchanged."""

    def __init__(self, segments, bounds, start_pose, labels, room_rects,
                 doors, family, reach=None, reach_origin=(0.0, 0.0)):
        super().__init__(segments, bounds, start_pose, partitions=None)
        self.labels = labels                    # [ny, nx] int16 or None
        self.room_rects = np.asarray(room_rects, np.float64).reshape(-1, 5)
        self.doors = np.asarray(doors, np.float64).reshape(-1, 6)
        self.family = family
        self.reach = reach                      # [ry, rx] bool raster
        self.reach_origin = reach_origin
        rooms = self.room_rects[:, 4].astype(int)
        self.n_rooms = int(rooms.max()) + 1 if rooms.size else 0
        self.rooms_reachable = self._reachable_rooms()

    def room_id(self, x: float, y: float) -> int:
        r = self.room_rects
        inside = (x >= r[:, 0]) & (x < r[:, 2]) & (y >= r[:, 1]) & (y < r[:, 3])
        hits = r[inside, 4]
        return int(hits[0]) if hits.size else -1

    def reachable(self, x: float, y: float) -> bool:
        if self.reach is None:
            return self.clearance(x, y) > ROBOT_R + FREE_MARGIN
        i = int((y - self.reach_origin[1]) / RASTER)
        j = int((x - self.reach_origin[0]) / RASTER)
        ry, rx = self.reach.shape
        return 0 <= i < ry and 0 <= j < rx and bool(self.reach[i, j])

    def reachable_in_box(self, x0, y0, x1, y1) -> bool:
        """Any reachable raster point inside the axis box (coverage cell
        ceiling: a cell counts if the rover centre can enter it)."""
        if self.reach is None:
            return False
        ox, oy = self.reach_origin
        ry, rx = self.reach.shape
        i0 = max(0, int(np.floor((y0 - oy) / RASTER)))
        i1 = min(ry, int(np.ceil((y1 - oy) / RASTER)))
        j0 = max(0, int(np.floor((x0 - ox) / RASTER)))
        j1 = min(rx, int(np.ceil((x1 - ox) / RASTER)))
        return i1 > i0 and j1 > j0 and bool(self.reach[i0:i1, j0:j1].any())

    def _reachable_rooms(self) -> set:
        if self.reach is None:
            return set(range(self.n_rooms))
        ii, jj = np.nonzero(self.reach)
        if ii.size == 0:
            return set()
        ox, oy = self.reach_origin
        xs = ox + (jj + 0.5) * RASTER
        ys = oy + (ii + 0.5) * RASTER
        out = set()
        for x0, y0, x1, y1, room in self.room_rects:
            if room in out:
                continue
            m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            if m.any():
                out.add(int(room))
        return out


# ---------------------------------------------------------------- geometry

def _label_components(labels: np.ndarray, min_cells: int) -> np.ndarray:
    """Relabel 4-connected same-label components 0..K-1; components below
    min_cells become -1 (void). Guarantees every room is one connected
    region (a notch can cut a split region in two)."""
    ny, nx = labels.shape
    out = np.full_like(labels, -1)
    k = 0
    for sy in range(ny):
        for sx in range(nx):
            if labels[sy, sx] < 0 or out[sy, sx] != -1:
                continue
            lab = labels[sy, sx]
            comp = [(sy, sx)]
            out[sy, sx] = k
            q = deque(comp)
            while q:
                y, x = q.popleft()
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (0 <= yy < ny and 0 <= xx < nx and out[yy, xx] == -1
                            and labels[yy, xx] == lab):
                        out[yy, xx] = k
                        comp.append((yy, xx))
                        q.append((yy, xx))
            if len(comp) < min_cells:
                for y, x in comp:
                    out[y, x] = -2              # mark void, keep k
            else:
                k += 1
    out[out == -2] = -1
    return out


def _edge_labels(labels: np.ndarray):
    """(H_a, H_b) [ny+1, nx]: labels below/above each horizontal edge;
    (V_a, V_b) [ny, nx+1]: labels left/right of each vertical edge."""
    ny, nx = labels.shape
    pad = np.full((ny + 2, nx + 2), -1, labels.dtype)
    pad[1:-1, 1:-1] = labels
    Ha, Hb = pad[0:-1, 1:-1], pad[1:, 1:-1]
    Va, Vb = pad[1:-1, 0:-1], pad[1:-1, 1:]
    return Ha, Hb, Va, Vb


def _runs(mask_1d: np.ndarray):
    """[(start, end_exclusive)] of True runs."""
    out, i, n = [], 0, len(mask_1d)
    while i < n:
        if mask_1d[i]:
            j = i
            while j < n and mask_1d[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _rects_from_labels(labels: np.ndarray):
    """Greedy maximal-rectangle cover of each label (disjoint). Rows of
    equal-label runs are extended downward while identical."""
    ny, nx = labels.shape
    used = np.zeros_like(labels, bool)
    rects = []
    for y in range(ny):
        x = 0
        while x < nx:
            lab = labels[y, x]
            if lab < 0 or used[y, x]:
                x += 1
                continue
            x1 = x
            while x1 < nx and labels[y, x1] == lab and not used[y, x1]:
                x1 += 1
            y1 = y + 1
            while (y1 < ny and np.all(labels[y1, x:x1] == lab)
                   and not used[y1, x:x1].any()):
                y1 += 1
            used[y:y1, x:x1] = True
            rects.append((x * U, y * U, x1 * U, y1 * U, int(lab)))
            x = x1
    return rects


def _segments_from_walls(wall_h: np.ndarray, wall_v: np.ndarray):
    segs = []
    for j in range(wall_h.shape[0]):
        for a, b in _runs(wall_h[j]):
            segs.append([a * U, j * U, b * U, j * U])
    for i in range(wall_v.shape[1]):
        for a, b in _runs(wall_v[:, i]):
            segs.append([i * U, a * U, i * U, b * U])
    return segs


def _thresholds(labels, wall_h, wall_v):
    """Door thresholds = maximal runs of OPEN edges separating two
    different in-building rooms."""
    Ha, Hb, Va, Vb = _edge_labels(labels)
    doors = []
    open_h = ~wall_h & (Ha != Hb) & (Ha >= 0) & (Hb >= 0)
    for j in range(open_h.shape[0]):
        i = 0
        row = open_h[j]
        while i < len(row):
            if not row[i]:
                i += 1
                continue
            pa, pb = Ha[j, i], Hb[j, i]
            k = i
            while k < len(row) and row[k] and Ha[j, k] == pa and Hb[j, k] == pb:
                k += 1
            doors.append([i * U, j * U, k * U, j * U, pa, pb])
            i = k
    open_v = ~wall_v & (Va != Vb) & (Va >= 0) & (Vb >= 0)
    for i in range(open_v.shape[1]):
        col = open_v[:, i]
        j = 0
        while j < len(col):
            if not col[j]:
                j += 1
                continue
            pa, pb = Va[j, i], Vb[j, i]
            k = j
            while k < len(col) and col[k] and Va[k, i] == pa and Vb[k, i] == pb:
                k += 1
            doors.append([i * U, j * U, i * U, k * U, pa, pb])
            j = k
    return doors


def _carve_doors(rng, labels, wall_h, wall_v, door_units, p_loop,
                 prefer_room: int | None = None, max_tries: int = 1):
    """Open one doorway per spanning-tree adjacency (+ loops with p_loop).
    door_units: callable rng -> width in lattice units. Returns False when
    some room has no candidate wall run long enough (regenerate)."""
    Ha, Hb, Va, Vb = _edge_labels(labels)
    cand = {}                 # (a, b) -> [("h"|"v", line, lo, hi)]
    shared_h = (Ha != Hb) & (Ha >= 0) & (Hb >= 0)
    for j in range(shared_h.shape[0]):
        i, row = 0, shared_h[j]
        while i < len(row):
            if not row[i]:
                i += 1
                continue
            pa, pb = Ha[j, i], Hb[j, i]
            k = i
            while k < len(row) and row[k] and Ha[j, k] == pa and Hb[j, k] == pb:
                k += 1
            cand.setdefault((min(pa, pb), max(pa, pb)), []).append(("h", j, i, k))
            i = k
    shared_v = (Va != Vb) & (Va >= 0) & (Vb >= 0)
    for i in range(shared_v.shape[1]):
        j, col = 0, shared_v[:, i]
        while j < len(col):
            if not col[j]:
                j += 1
                continue
            pa, pb = Va[j, i], Vb[j, i]
            k = j
            while k < len(col) and col[k] and Va[k, i] == pa and Vb[k, i] == pb:
                k += 1
            cand.setdefault((min(pa, pb), max(pa, pb)), []).append(("v", i, j, k))
            j = k

    n_rooms = int(labels.max()) + 1
    parent = list(range(n_rooms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = list(cand.keys())
    rng.shuffle(pairs)
    if prefer_room is not None:           # corridor links first
        pairs.sort(key=lambda p: 0 if prefer_room in p else 1)

    def open_door(pair):
        width = door_units(rng)
        runs = [r for r in cand[pair] if r[3] - r[2] >= width + 2]
        if not runs:
            return False
        kind, line, lo, hi = runs[int(rng.integers(len(runs)))]
        start = int(rng.integers(lo + 1, hi - width))
        if kind == "h":
            wall_h[line, start:start + width] = False
        else:
            wall_v[start:start + width, line] = False
        return True

    for pair in pairs:
        ra, rb = find(pair[0]), find(pair[1])
        if ra != rb:
            if open_door(pair):
                parent[ra] = rb
        elif rng.random() < p_loop:
            open_door(pair)
    roots = {find(r) for r in range(n_rooms)}
    return len(roots) == 1


def _bsp(rng, x0, y0, x1, y1, min_side, max_rooms, out):
    """Recursive split in lattice units; appends leaf rects."""
    w, h = x1 - x0, y1 - y0
    can_v, can_h = w >= 2 * min_side, h >= 2 * min_side
    big = w * h > (min_side * 1.6) ** 2
    if len(out) + 1 >= max_rooms or not (can_v or can_h) or (
            not big and rng.random() < 0.5):
        out.append((x0, y0, x1, y1))
        return
    vertical = can_v and (not can_h or (w > h if rng.random() < 0.75
                                        else rng.random() < 0.5))
    if vertical:
        s = int(rng.integers(x0 + min_side, x1 - min_side + 1))
        _bsp(rng, x0, y0, s, y1, min_side, max_rooms, out)
        _bsp(rng, s, y0, x1, y1, min_side, max_rooms, out)
    else:
        s = int(rng.integers(y0 + min_side, y1 - min_side + 1))
        _bsp(rng, x0, y0, x1, s, min_side, max_rooms, out)
        _bsp(rng, x0, s, x1, y1, min_side, max_rooms, out)


def _natural_door(rng) -> int:
    return int(rng.choice([3, 4]))              # 0.75 / 1.0 m


def _archway(rng) -> int:
    return int(rng.integers(6, 11))             # 1.5-2.5 m


# ---------------------------------------------------------------- families

def _plan_bsp(rng, footprint=None, max_rooms=None):
    small = max_rooms is not None and max_rooms <= 4
    nx = int(round(rng.uniform(*((5.0, 8.0) if small else (7.0, 13.0))) / U))
    ny = int(round(rng.uniform(*((4.5, 7.0) if small else (6.0, 11.0))) / U))
    leaves = []
    cap = int(rng.integers(3, 10))
    if max_rooms is not None:
        cap = min(cap, max_rooms)
    _bsp(rng, 0, 0, nx, ny, min_side=9, max_rooms=cap, out=leaves)
    labels = np.full((ny, nx), -1, np.int32)
    for k, (a, b, c, d) in enumerate(leaves):
        labels[b:d, a:c] = k
    if footprint is not None:
        labels[~footprint(nx, ny)] = -1
    return labels


def _footprint_cut(rng):
    kind = rng.choice(["L", "U", "notch2"])

    def fp(nx, ny):
        m = np.ones((ny, nx), bool)
        cw = int(nx * rng.uniform(0.3, 0.5))
        ch = int(ny * rng.uniform(0.3, 0.5))
        if kind == "L":
            cy = 0 if rng.random() < 0.5 else ny - ch
            cx = 0 if rng.random() < 0.5 else nx - cw
            m[cy:cy + ch, cx:cx + cw] = False
        elif kind == "U":
            cx = (nx - cw) // 2
            m[ny - ch:, cx:cx + cw] = False
        else:
            m[:ch, :cw] = False
            m[ny - ch // 2:, nx - cw // 2:] = False
        return m
    return fp


def _plan_hall(rng):
    long_ = int(round(rng.uniform(9.0, 14.0) / U))
    side = int(round(rng.uniform(2.5, 4.0) / U))
    corridor = int(rng.integers(4, 7))           # 1.0-1.5 m
    ny = 2 * side + corridor
    labels = np.full((ny, long_), -1, np.int32)
    k = 0
    labels[side:side + corridor, :] = k          # corridor = room 0
    k += 1
    for y0, y1 in ((0, side), (side + corridor, ny)):
        x = 0
        while x < long_:
            w = int(rng.integers(10, 17))        # 2.5-4.0 m rooms
            if long_ - (x + w) < 10:
                w = long_ - x
            labels[y0:y1, x:x + w] = k
            k += 1
            x += w
    if rng.random() < 0.5:                       # swap axes
        labels = labels.T.copy()
    return labels, 0


def _plan_open(rng):
    nx = int(round(rng.uniform(8.0, 13.0) / U))
    ny = int(round(rng.uniform(7.0, 11.0) / U))
    leaves = []
    _bsp(rng, 0, 0, nx, ny, min_side=14, max_rooms=int(rng.integers(2, 5)),
         out=leaves)
    labels = np.full((ny, nx), -1, np.int32)
    for k, (a, b, c, d) in enumerate(leaves):
        labels[b:d, a:c] = k
    return labels


def _plan_maze(rng, wide: bool = False):
    """Braided maze. Returns labels + wall masks with maze walls already
    carved (zones label the rooms; openings across zones = thresholds)."""
    cell = 6 if wide else int(rng.choice([4, 5, 6]))   # 1.0/1.25/1.5 m
    mx = int(rng.integers(6, 10))
    my = int(rng.integers(5, 9))
    while mx * cell * U > 13.0:
        mx -= 1
    while my * cell * U > 11.0:
        my -= 1
    zone = int(rng.choice([2, 3]))
    nx, ny = mx * cell, my * cell
    labels = np.zeros((ny, nx), np.int32)
    zx = -(-mx // zone)
    for cy in range(my):
        for cx in range(mx):
            labels[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell] = \
                (cy // zone) * zx + (cx // zone)
    # all cell edges start as walls
    wall_h = np.zeros((ny + 1, nx), bool)
    wall_v = np.zeros((ny, nx + 1), bool)
    for cy in range(my + 1):
        wall_h[cy * cell, :] = True
    for cx in range(mx + 1):
        wall_v[:, cx * cell] = True
    # recursive backtracker
    seen = np.zeros((my, mx), bool)
    stack = [(int(rng.integers(my)), int(rng.integers(mx)))]
    seen[stack[0]] = True

    def knock(a, b):
        (ay, ax), (by, bx) = a, b
        if ay == by:
            x = max(ax, bx) * cell
            wall_v[ay * cell:(ay + 1) * cell, x] = False
        else:
            y = max(ay, by) * cell
            wall_h[y, ax * cell:(ax + 1) * cell] = False

    while stack:
        y, x = stack[-1]
        nbrs = [(y + dy, x + dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= y + dy < my and 0 <= x + dx < mx and not seen[y + dy, x + dx]]
        if not nbrs:
            stack.pop()
            continue
        n = nbrs[int(rng.integers(len(nbrs)))]
        knock((y, x), n)
        seen[n] = True
        stack.append(n)
    # braid: knock extra walls to create loops (a pet maze, not a puzzle)
    for _ in range(int(mx * my * rng.uniform(0.08, 0.2))):
        y, x = int(rng.integers(my)), int(rng.integers(mx))
        dy, dx = [(0, 1), (1, 0)][int(rng.integers(2))]
        if y + dy < my and x + dx < mx:
            knock((y, x), (y + dy, x + dx))
    # keep the outer shell closed
    wall_h[0, :] = wall_h[-1, :] = True
    wall_v[:, 0] = wall_v[:, -1] = True
    labels = _label_components(labels, 1)
    return labels, wall_h, wall_v


def _walls_from_labels(labels):
    Ha, Hb, Va, Vb = _edge_labels(labels)
    wall_h = (Ha != Hb)
    wall_v = (Va != Vb)
    return wall_h, wall_v


# ---------------------------------------------------------------- assembly

def _free_raster(segs: np.ndarray, nx_m: float, ny_m: float):
    rx = int(np.ceil(nx_m / RASTER))
    ry = int(np.ceil(ny_m / RASTER))
    xs = (np.arange(rx) + 0.5) * RASTER
    ys = (np.arange(ry) + 0.5) * RASTER
    a = segs[:, 0:2]
    e = segs[:, 2:4] - segs[:, 0:2]
    ee = np.maximum((e * e).sum(1), 1e-12)
    d2min = np.full((ry, rx), np.inf)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    best = np.full(P.shape[0], np.inf)
    for k in range(0, segs.shape[0], 16):
        ak, ek, eek = a[k:k + 16], e[k:k + 16], ee[k:k + 16]
        ap = P[:, None, :] - ak[None]
        t = np.clip((ap * ek[None]).sum(2) / eek[None], 0.0, 1.0)
        c = ak[None] + t[..., None] * ek[None]
        best = np.minimum(best, ((P[:, None, :] - c) ** 2).sum(2).min(1))
    d2min = best.reshape(ry, rx)
    return d2min > (ROBOT_R + FREE_MARGIN) ** 2


def _flood(free: np.ndarray, seed_ij) -> np.ndarray:
    reach = np.zeros_like(free)
    if not free[seed_ij]:
        return reach
    reach[seed_ij] = True
    while True:
        grow = reach.copy()
        grow[1:, :] |= reach[:-1, :]
        grow[:-1, :] |= reach[1:, :]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= free
        if (grow == reach).all():
            return reach
        reach = grow


def _furniture(rng, labels, doors, family):
    """Boxes inside rooms, >= 0.6 m from any threshold and 0.35 m off
    walls. Mazes stay empty (the corridors are the obstacle)."""
    if family == "maze":
        return []
    ny, nx = labels.shape
    n = {"open": (3, 8), "hall": (2, 6)}.get(family, (3, 9))
    boxes = []
    d = np.asarray(doors, float).reshape(-1, 6)
    for _ in range(int(rng.integers(*n)) * 3):
        if len(boxes) >= n[1]:
            break
        bw, bh = rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2)
        cx, cy = rng.uniform(0, nx * U), rng.uniform(0, ny * U)
        rad = 0.5 * np.hypot(bw, bh)
        # whole footprint + standoff inside one room
        x0, x1 = cx - rad - 0.35, cx + rad + 0.35
        y0, y1 = cy - rad - 0.35, cy + rad + 0.35
        if x0 < 0 or y0 < 0 or x1 > nx * U or y1 > ny * U:
            continue
        patch = labels[int(y0 / U):int(np.ceil(y1 / U)),
                       int(x0 / U):int(np.ceil(x1 / U))]
        if patch.size == 0 or patch.min() < 0 or patch.min() != patch.max():
            continue
        if d.size:
            mid = 0.5 * (d[:, 0:2] + d[:, 2:4])
            if (np.hypot(mid[:, 0] - cx, mid[:, 1] - cy) < rad + 0.6 +
                    0.5 * np.hypot(d[:, 2] - d[:, 0], d[:, 3] - d[:, 1])).any():
                continue
        if any(np.hypot(cx - b[0], cy - b[1]) < rad + b[2] + 0.5 for b in boxes):
            continue
        boxes.append((cx, cy, rad, bw, bh, float(rng.uniform(0, np.pi))))
    return boxes


def _pick_start(rng, S, W, H, free, room_of, doors, mode):
    """(x, y, heading, raster_i, raster_j) or None.
    mode 'random': any free spot in a room with clearance > 0.35 m.
    mode 'near_door': 0.8-2.0 m in front of a random threshold, heading at
    its midpoint +/- 30 deg (curriculum level 0-1: the doorway is in view)."""
    ii, jj = np.nonzero(free)
    if ii.size == 0:
        return None
    probe = World(S, (0.0, 0.0, W, H), (0.0, 0.0, 0.0))
    d = np.asarray(doors, float).reshape(-1, 6)
    for _ in range(300):
        if mode == "near_door" and d.shape[0]:
            k = d[int(rng.integers(d.shape[0]))]
            mid = 0.5 * (k[0:2] + k[2:4])
            t = (k[2:4] - k[0:2]) / max(1e-9, np.hypot(*(k[2:4] - k[0:2])))
            nrm = np.array([-t[1], t[0]]) * (1 if rng.random() < 0.5 else -1)
            p = mid + nrm * rng.uniform(0.8, 2.0) + t * rng.uniform(-0.4, 0.4)
            x, y = float(p[0]), float(p[1])
            i, j = int(y / RASTER), int(x / RASTER)
            if not (0 <= i < free.shape[0] and 0 <= j < free.shape[1]
                    and free[i, j]):
                continue
            heading = float(np.arctan2(mid[1] - y, mid[0] - x)
                            + rng.uniform(-0.52, 0.52))
        else:
            q = int(rng.integers(ii.size))
            i, j = int(ii[q]), int(jj[q])
            x, y = (j + 0.5) * RASTER, (i + 0.5) * RASTER
            heading = float(rng.uniform(-np.pi, np.pi))
        if room_of(x, y) >= 0 and probe.clearance(x, y) > 0.35:
            return (x, y, heading, i, j)
    return None


def _assemble(rng, family, labels, wall_h, wall_v,
              start_mode: str = "random") -> Building | None:
    ny, nx = labels.shape
    segs = _segments_from_walls(wall_h, wall_v)
    doors = _thresholds(labels, wall_h, wall_v)
    boxes = _furniture(rng, labels, doors, family)
    base = np.asarray(segs, float)
    rects = _rects_from_labels(labels)
    W, H = nx * U, ny * U

    for n_keep in range(len(boxes), -1, -1):
        allsegs = [base] + [_box_segments(b[0], b[1], b[3], b[4], b[5])
                            for b in boxes[:n_keep]]
        S = np.concatenate(allsegs, 0)
        if S.shape[0] > MAX_SEGMENTS:
            continue
        free = _free_raster(S, W, H)

        def room_of(x, y):
            return int(labels[min(ny - 1, int(y / U)), min(nx - 1, int(x / U))])
        start = _pick_start(rng, S, W, H, free, room_of, doors, start_mode)
        if start is None:
            continue
        reach = _flood(free, (start[3], start[4]))
        b = Building(S, (0.0, 0.0, W, H), start[:3], labels, rects, doors,
                     family, reach=reach)
        if len(b.rooms_reachable) == b.n_rooms:
            return b
    return None


def make_building(rng: np.random.Generator, family: str = "mix",
                  max_tries: int = 40, doors: str = "natural",
                  start: str = "random", max_rooms: int | None = None,
                  weights: dict | None = None) -> Building:
    """One building of `family` (see module doc); regenerates until every
    room is reachable from the start pose and segments <= MAX_SEGMENTS.

    Difficulty knobs (curriculum, see LEVELS):
      doors      'natural' 0.75/1.0 m | 'wide' 1.5-2.5 m archways (maze:
                 1.5 m corridors; house: 1.5-2.0 m gaps)
      start      'random' | 'near_door' (0.8-2.0 m from a threshold, facing
                 it +/- 30 deg)
      max_rooms  cap for bsp/lshape/open (hall/maze are always >= 3)
      weights    family weights for family='mix' (default MIX_WEIGHTS)"""
    wts = weights or MIX_WEIGHTS
    for _ in range(max_tries):
        fam = family
        if fam == "mix":
            names = list(wts)
            p = np.array([wts[k] for k in names], float)
            fam = names[int(rng.choice(len(names), p=p / p.sum()))]
        if fam == "house":
            dw = (1.5, 2.0) if doors == "wide" else (0.7, 1.0)
            b = house_to_building(make_house(rng, door_w_range=dw), rng=rng,
                                  start_mode=start)
            if b is not None and (max_rooms is None
                                  or b.n_rooms <= max(max_rooms, 2)):
                return b
            continue
        if fam == "maze":
            labels, wall_h, wall_v = _plan_maze(
                rng, wide=doors == "wide")
            b = _assemble(rng, fam, labels, wall_h, wall_v, start)
        else:
            prefer = None
            if fam == "bsp":
                labels = _plan_bsp(rng, max_rooms=max_rooms)
            elif fam == "lshape":
                labels = _plan_bsp(rng, footprint=_footprint_cut(rng),
                                   max_rooms=max_rooms)
            elif fam == "open":
                labels = _plan_open(rng)
            elif fam == "hall":
                labels, prefer = _plan_hall(rng)
            else:
                raise ValueError(f"unknown building family {fam!r}")
            labels = _label_components(labels, min_cells=36)   # >= 2.25 m2
            if labels.max() < 0:
                continue
            wall_h, wall_v = _walls_from_labels(labels)
            door = (_archway if fam == "open" or doors == "wide"
                    else _natural_door)
            p_loop = {"open": 0.6, "bsp": 0.15, "lshape": 0.15,
                      "hall": 0.1}[fam]
            if not _carve_doors(rng, labels, wall_h, wall_v, door, p_loop,
                                prefer_room=prefer):
                continue
            b = _assemble(rng, fam, labels, wall_h, wall_v, start)
        if b is not None and (max_rooms is None or fam in ("hall", "maze")
                              or b.n_rooms <= max_rooms):
            return b
    raise RuntimeError(f"make_building({family!r}) failed {max_tries} tries")


def house_to_building(w: World, rng=None,
                      start_mode: str = "random") -> Building | None:
    """Legacy slab house -> Building: rects = the partition grid, doors =
    gaps in the segments lying on each partition line (room pair read on
    either side of the gap), reachability raster from the real segments."""
    x0, y0, x1, y1 = w.bounds
    xs = sorted([x0, x1] + [p for k, p in w.partitions if k == "v"])
    ys = sorted([y0, y1] + [p for k, p in w.partitions if k == "h"])
    rects, ids = [], {}
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
            rid = ids.setdefault(w.room_id(cx, cy), len(ids))
            rects.append((xs[i], ys[j], xs[i + 1], ys[j + 1], rid))
    doors = []
    S = w.segments
    for kind, pos in w.partitions:
        if kind == "v":
            on = S[(np.abs(S[:, 0] - pos) < 1e-6) & (np.abs(S[:, 2] - pos) < 1e-6)]
            iv = sorted((min(a, b), max(a, b)) for a, b in on[:, [1, 3]])
            lo_all, hi_all = y0, y1
        else:
            on = S[(np.abs(S[:, 1] - pos) < 1e-6) & (np.abs(S[:, 3] - pos) < 1e-6)]
            iv = sorted((min(a, b), max(a, b)) for a, b in on[:, [0, 2]])
            lo_all, hi_all = x0, x1
        edges = [lo_all] + [v for ab in iv for v in ab] + [hi_all]
        for g0, g1 in zip(edges[0::2], edges[1::2]):
            if g1 - g0 < 0.3:
                continue
            m = 0.5 * (g0 + g1)
            if kind == "v":
                ra, rb = w.room_id(pos - 0.05, m), w.room_id(pos + 0.05, m)
                seg = [pos, g0, pos, g1]
            else:
                ra, rb = w.room_id(m, pos - 0.05), w.room_id(m, pos + 0.05)
                seg = [g0, pos, g1, pos]
            doors.append(seg + [ids.get(ra, -1), ids.get(rb, -1)])
    free = _free_raster(S, x1 - x0, y1 - y0)
    pose = w.start_pose
    si = min(free.shape[0] - 1, int(pose[1] / RASTER))
    sj = min(free.shape[1] - 1, int(pose[0] / RASTER))
    if start_mode == "near_door" and rng is not None:
        tmp = Building(S, w.bounds, pose, None, rects, doors, "house")
        st = _pick_start(rng, S, x1 - x0, y1 - y0, free, tmp.room_id,
                         [d for d in doors if d[4] >= 0 and d[5] >= 0],
                         "near_door")
        if st is None:
            return None
        pose, si, sj = st[:3], st[3], st[4]
    reach = _flood(free, (si, sj))
    return Building(S, w.bounds, pose, None, rects, doors, "house",
                    reach=reach)
