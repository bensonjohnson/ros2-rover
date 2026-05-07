"""Session-scoped occupancy tracker for the DreamerV3 rover.

Drives two reward channels (both folded into the `coverage` scalar that
ships in the rover's reward vector — `frontier` is a separate scalar with
PBRS semantics, kept distinct):

* `coverage`  — combined per-tick signal:
    1. `α · (cells transitioning UNKNOWN → KNOWN this tick)` — dense gradient
       toward driving into virgin space (any new cell anywhere).
    2. `β · (newly-visited perimeter cells this tick)` — perimeter-coverage
       shaping. A *perimeter cell* is a FREE cell within `K` cells of an
       OCCUPIED cell, and it counts as "visited" when the rover is within
       `R` cells of it. This rewards the rover for systematically sweeping
       along walls (the natural way to find doorways and openings) while
       *not* paying it to drive into walls (visit ≠ collision). Stops
       firing once the perimeter of the current room is fully covered;
       refills automatically when a new room is entered.

* `frontier` — potential-based shaping `γ Φ(s') − Φ(s)` with
  `Φ(s) = −d(robot, nearest BFS-frontier)`. A frontier is a FREE cell that
  has at least one UNKNOWN 4-neighbour, *reachable from the robot through
  FREE cells via 8-connectivity BFS*. The 8-connectivity step lets the BFS
  squeeze through 1-cell discretization gaps in door frames (lidar
  occasionally marks single OCCUPIED cells across thin obstacles); the
  frontier definition itself stays 4-connected so we don't pretend that
  diagonal-touching FREE cells "border" UNKNOWN. PBRS-net-zero (Ng et al.
  1999), so it doesn't bias the optimal policy — when no frontier is
  reachable (sealed room), `phi=0` and the channel is silent, which is
  the *correct* behaviour: there's nothing reachable to head toward, so
  the perimeter-coverage shaping above does the bootstrapping.

The grid is intentionally **world-fixed and per-session**: it resets on
operator command or when the robot teleports (large odometry jump). The
cross-environment "have I been here" memory lives in the server-side
SimHash over encoder embeddings; that signal is environment-agnostic and
won't pre-fill the grid when the rover is carried outside.

Performance: a 200×200 uint8 grid with `cv2.line()` per LiDAR return runs
under 2% CPU on an RK3588 at 30 Hz with ~600 returns per scan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # only used for ray-tracing; fallback marks endpoints only
    HAS_CV2 = False


UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


@dataclass
class CoverageStep:
    coverage_delta: float       # combined coverage + perimeter reward this tick (after clip)
    frontier_distance: float    # metric distance to nearest BFS-reachable frontier, ∞ if none
    phi: float                  # potential = -frontier_distance (clipped)
    frontier_angle: float       # bearing to nearest frontier in robot frame, [-π, π]
    has_frontier: bool


class CoverageTracker:
    """Incremental world-fixed occupancy grid + frontier distance.

    Coordinate convention:
        - World axes follow odom: x forward, y left, yaw CCW.
        - Grid origin is `(origin_x, origin_y)` (centered on first observed
          pose). `(row, col)` ↔ world `(x, y)` via `_world_to_cell`.
    """

    def __init__(
        self,
        grid_size: int = 200,
        resolution: float = 0.05,        # m / cell → 200 × 0.05 = 10 m square
        max_range: float = 5.0,
        coverage_clip: tuple[float, float] = (0.0, 0.5),
        coverage_alpha: float = 0.001,   # one cell = 0.001 reward (so 500 cells = clip)
        teleport_threshold: float = 5.0, # metres
        frontier_max_search: float = 8.0,
        # Perimeter-coverage shaping:
        perimeter_radius_cells: int = 3, # FREE cells within this many cells of OCCUPIED
        perimeter_visit_radius_cells: int = 20,  # rover within this many cells = "visited"
        perimeter_alpha: float = 0.005,  # reward per newly-visited perimeter cell
    ):
        self.grid_size = grid_size
        self.resolution = resolution
        self.max_range = max_range
        self.coverage_clip = coverage_clip
        self.coverage_alpha = coverage_alpha
        self.teleport_threshold = teleport_threshold
        self.frontier_max_search = frontier_max_search
        self.perimeter_radius_cells = int(perimeter_radius_cells)
        self.perimeter_visit_radius_cells = int(perimeter_visit_radius_cells)
        self.perimeter_alpha = float(perimeter_alpha)

        self.grid = np.full((grid_size, grid_size), UNKNOWN, dtype=np.uint8)
        # World-frame origin of cell (0, 0). Set on first scan.
        self.origin_x: Optional[float] = None
        self.origin_y: Optional[float] = None
        self._last_pose: Optional[tuple[float, float]] = None
        self._known_count = 0
        # Perimeter "have I been near this perimeter cell" memory. Session-
        # scoped, persists across episodes within the same grid; reset() clears.
        self.perimeter_visited = np.zeros((grid_size, grid_size), dtype=bool)

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def reset(self, reason: str = 'manual') -> None:
        """Hard reset (operator command or environment switch)."""
        self.grid.fill(UNKNOWN)
        self.origin_x = None
        self.origin_y = None
        self._last_pose = None
        self._known_count = 0
        self.perimeter_visited.fill(False)

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def step(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> CoverageStep:
        """Integrate one LiDAR scan and return coverage + frontier signals."""
        if self.origin_x is None:
            half = self.grid_size * self.resolution * 0.5
            self.origin_x = robot_x - half
            self.origin_y = robot_y - half
            self._last_pose = (robot_x, robot_y)

        # Detect teleport / "carried to a new place" → reset
        if self._last_pose is not None:
            dx = robot_x - self._last_pose[0]
            dy = robot_y - self._last_pose[1]
            if (dx * dx + dy * dy) > self.teleport_threshold * self.teleport_threshold:
                self.reset(reason='teleport')
                return self.step(ranges, angle_min, angle_increment, robot_x, robot_y, robot_yaw)
        self._last_pose = (robot_x, robot_y)

        # Robot cell
        rr, rc = self._world_to_cell(robot_x, robot_y)
        if not self._in_bounds(rr, rc):
            # Robot has driven off the session grid; treat as new environment.
            self.reset(reason='out_of_grid')
            return self.step(ranges, angle_min, angle_increment, robot_x, robot_y, robot_yaw)

        # Endpoints in world frame, then cell frame
        ranges = np.asarray(ranges, dtype=np.float32)
        n = ranges.shape[0]
        if n == 0:
            return self._empty_step(robot_x, robot_y, robot_yaw, rr, rc)

        angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment + robot_yaw
        valid = np.isfinite(ranges) & (ranges > 0.05)
        # Cap to max_range for "seen empty up to here" — beyond max_range we can't claim free.
        capped_ranges = np.clip(ranges, 0.0, self.max_range)
        is_hit = valid & (ranges <= self.max_range)
        is_pass_through = valid & (ranges > self.max_range)

        ex = robot_x + capped_ranges * np.cos(angles)
        ey = robot_y + capped_ranges * np.sin(angles)
        end_cells_r, end_cells_c = self._world_to_cell_arr(ex, ey)
        in_grid = (
            (end_cells_r >= 0) & (end_cells_r < self.grid_size) &
            (end_cells_c >= 0) & (end_cells_c < self.grid_size)
        )

        prev_known = self._known_count

        # Free cells: ray-trace robot → endpoint. Use cv2.line on a temp mask
        # so we can flip UNKNOWN → FREE without touching existing OCCUPIED.
        if HAS_CV2:
            free_mask = np.zeros_like(self.grid)
            for i in range(n):
                if not (valid[i] and in_grid[i]):
                    continue
                cv2.line(free_mask, (rc, rr), (int(end_cells_c[i]), int(end_cells_r[i])), 1, 1)
            unknown_now = (self.grid == UNKNOWN)
            transition = unknown_now & (free_mask > 0)
            self.grid[transition] = FREE
            self._known_count += int(transition.sum())

        # Mark hit endpoints as OCCUPIED (overrides FREE if needed)
        hit_idx = np.flatnonzero(is_hit & in_grid)
        if hit_idx.size > 0:
            rs = end_cells_r[hit_idx]
            cs = end_cells_c[hit_idx]
            new_known = (self.grid[rs, cs] == UNKNOWN)
            self._known_count += int(new_known.sum())
            self.grid[rs, cs] = OCCUPIED

        delta = self._known_count - prev_known
        # Perimeter-coverage shaping bonus: count newly-visited perimeter
        # cells within visit radius of the robot's current cell.
        perim_new = self._update_perimeter_visited(rr, rc)

        coverage_reward = float(np.clip(
            delta * self.coverage_alpha + perim_new * self.perimeter_alpha,
            self.coverage_clip[0], self.coverage_clip[1],
        ))

        # Frontier (BFS-reachable through FREE, 8-conn expansion / 4-conn detection)
        frontier_d, frontier_a, has_f = self._nearest_frontier(rr, rc, robot_yaw)
        phi = -frontier_d if has_f else 0.0

        return CoverageStep(
            coverage_delta=coverage_reward,
            frontier_distance=frontier_d,
            phi=phi,
            frontier_angle=frontier_a,
            has_frontier=has_f,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def _world_to_cell_arr(self, x: np.ndarray, y: np.ndarray):
        col = ((x - self.origin_x) / self.resolution).astype(np.int32)
        row = ((y - self.origin_y) / self.resolution).astype(np.int32)
        return row, col

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _empty_step(self, x, y, yaw, rr, rc) -> CoverageStep:
        # Even on an empty scan we still credit any perimeter cells that
        # come into visit radius (e.g., from prior scans) — keeps the
        # signal continuous if a single scan drops out.
        perim_new = self._update_perimeter_visited(rr, rc)
        coverage_reward = float(np.clip(
            perim_new * self.perimeter_alpha,
            self.coverage_clip[0], self.coverage_clip[1],
        ))
        d, a, has_f = self._nearest_frontier(rr, rc, yaw)
        phi = -d if has_f else 0.0
        return CoverageStep(coverage_delta=coverage_reward, frontier_distance=d, phi=phi,
                            frontier_angle=a, has_frontier=has_f)

    # ------------------------------------------------------------------
    # Perimeter-coverage shaping
    # ------------------------------------------------------------------

    def _update_perimeter_visited(self, rr: int, rc: int) -> int:
        """Mark perimeter cells within visit-radius of the robot, return count
        of newly-marked cells (this tick).

        Perimeter cells are FREE cells whose distance-transform to the nearest
        OCCUPIED cell is in (0, perimeter_radius_cells]. We compute that mask
        every tick (cheap on a 200×200 grid) and intersect it with a circular
        visit window around the robot, then with `~perimeter_visited` to count
        only *new* visits.
        """
        if not HAS_CV2:
            return 0
        occupied = (self.grid == OCCUPIED)
        if not occupied.any():
            return 0
        # cv2.distanceTransform: 0 cells are sources, non-zero are queried.
        # We want distance from each cell to the nearest OCCUPIED cell, so
        # invert the OCCUPIED mask before passing in.
        non_occ = (1 - occupied.astype(np.uint8))
        dist_to_occ = cv2.distanceTransform(non_occ, cv2.DIST_L2, 3)
        is_perimeter = (
            (self.grid == FREE)
            & (dist_to_occ > 0.0)
            & (dist_to_occ <= float(self.perimeter_radius_cells))
        )

        # Circular visit window around the robot's cell — squared distance
        # avoids the sqrt and any need to allocate a float grid.
        rs = np.arange(self.grid_size).reshape(-1, 1) - rr
        cs = np.arange(self.grid_size).reshape(1, -1) - rc
        in_visit = (rs * rs + cs * cs) <= (self.perimeter_visit_radius_cells ** 2)

        new_perim = is_perimeter & in_visit & (~self.perimeter_visited)
        n_new = int(new_perim.sum())
        if n_new > 0:
            self.perimeter_visited[new_perim] = True
        return n_new

    # ------------------------------------------------------------------
    # Frontier search — BFS through FREE, 8-connectivity expansion
    # ------------------------------------------------------------------

    _FRONTIER_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    _BFS_NEIGHBORS = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    def _nearest_frontier(self, rr: int, rc: int, yaw: float) -> tuple[float, float, bool]:
        """Return (metric_distance, robot-frame bearing, found_flag).

        A frontier cell is a FREE cell with at least one UNKNOWN 4-neighbour.
        BFS expands through FREE cells with **8-connectivity**, so the search
        can squeeze through 1-cell-wide diagonal gaps in door frames left by
        lidar discretization. The frontier check itself stays 4-connected to
        keep the strict "FREE-and-touching-UNKNOWN" definition. First frontier
        found is approximately the geodesic-shortest one (good enough for
        PBRS shaping; only monotonicity matters, not Euclidean optimality).
        """
        if not self._in_bounds(rr, rc) or self.grid[rr, rc] == OCCUPIED:
            return float('inf'), 0.0, False

        max_radius = int(self.frontier_max_search / self.resolution)
        visited = np.zeros_like(self.grid, dtype=bool)
        visited[rr, rc] = True
        q = deque()
        q.append((rr, rc, 0))

        while q:
            r, c, d = q.popleft()
            if d > max_radius:
                break

            # Frontier check (4-conn): is (r, c) a FREE cell with an UNKNOWN
            # 4-neighbour? Diagonal "touch" through a wall corner doesn't
            # count — that would let a frontier "exist" across a true wall.
            if self.grid[r, c] == FREE:
                for dr, dc in self._FRONTIER_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(nr, nc):
                        continue
                    if self.grid[nr, nc] == UNKNOWN:
                        metric_d = d * self.resolution
                        wx = self.origin_x + (c + 0.5) * self.resolution
                        wy = self.origin_y + (r + 0.5) * self.resolution
                        rx = wx - (self.origin_x + (rc + 0.5) * self.resolution)
                        ry = wy - (self.origin_y + (rr + 0.5) * self.resolution)
                        angle_world = np.arctan2(ry, rx)
                        bearing = _wrap_pi(angle_world - yaw)
                        return metric_d, float(bearing), True

            # BFS expansion (8-conn): walk through FREE cells, allowing
            # diagonal hops to traverse 1-cell-wide door-frame artifacts.
            for dr, dc in self._BFS_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (not self._in_bounds(nr, nc)) or visited[nr, nc]:
                    continue
                visited[nr, nc] = True
                if self.grid[nr, nc] == FREE:
                    q.append((nr, nc, d + 1))

        # No frontier reachable within the search radius. Correct outcome
        # in a sealed room — perimeter-coverage shaping handles bootstrap.
        return float('inf'), 0.0, False


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))
