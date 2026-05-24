#!/usr/bin/env python3
"""Convert a 2D occupancy grid into coverage geometry.

The boustrophedon planner in ``coverage_planner.CoveragePlanner`` works on a
*boundary polygon* plus a list of *obstacle polygons*. SLAM (slam_toolbox)
produces a ``nav_msgs/OccupancyGrid`` instead. This module is the bridge:

    OccupancyGrid  --->  (free-space boundary polygon, [obstacle polygons])

It extracts the largest connected region of known-free space, takes its outer
contour as the boundary, and treats interior holes (occupied islands and
unmapped pockets) as obstacles. Everything is returned in *world / map* metric
coordinates as ``(x, y)`` tuples so the core has no ROS dependency and can be
unit-tested offline. Use :func:`points_to_ros` to convert to
``geometry_msgs/Point`` when feeding the planner.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

XY = Tuple[float, float]


def grid_to_polygons(
    grid: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    *,
    free_max: int = 25,
    occupied_min: int = 50,
    close_kernel_cells: int = 2,
    min_region_area_m2: float = 0.25,
    min_obstacle_area_m2: float = 0.05,
    simplify_eps_m: float = 0.05,
) -> Tuple[List[XY], List[List[XY]]]:
    """Extract a coverage boundary and obstacle polygons from an occupancy grid.

    Args:
        grid: (H, W) int array, ROS occupancy convention
            (-1 unknown, 0 free, 100 occupied). data[row, col] with row along
            +y and col along +x relative to ``origin``.
        resolution: meters per cell.
        origin_x, origin_y: world coords of the grid origin (cell [0, 0] lower-left
            corner), i.e. ``grid.info.origin.position``.
        free_max: cells with 0 <= value <= free_max are treated as free.
        occupied_min: cells with value >= occupied_min are treated as occupied.
        close_kernel_cells: morphological-close radius (cells) to bridge speckle.
        min_region_area_m2: ignore free regions smaller than this (noise).
        min_obstacle_area_m2: ignore interior holes smaller than this.
        simplify_eps_m: Douglas-Peucker simplification tolerance (meters).

    Returns:
        (boundary, obstacles) where ``boundary`` is a list of (x, y) world points
        (empty if no usable free region) and ``obstacles`` is a list of such
        polygons. Caller can pass these to ``CoveragePlanner.plan_coverage_path``.
    """
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {grid.shape}")

    free = ((grid >= 0) & (grid <= free_max)).astype(np.uint8)
    if not free.any():
        return [], []

    # Bridge small gaps so a noisy scan line doesn't split the room.
    if close_kernel_cells > 0:
        k = 2 * close_kernel_cells + 1
        kernel = np.ones((k, k), np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected free region (the room the robot is in).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if num <= 1:
        return [], []
    # label 0 is background; pick the largest non-background component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8)

    cell_area = resolution * resolution
    if stats[largest, cv2.CC_STAT_AREA] * cell_area < min_region_area_m2:
        return [], []

    # RETR_CCOMP: top-level outer contour(s) + their holes (2-level hierarchy).
    contours, hierarchy = cv2.findContours(
        region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return [], []
    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]
    eps_px = max(simplify_eps_m / resolution, 1.0)

    def to_world(contour: np.ndarray) -> List[XY]:
        pts = cv2.approxPolyDP(contour, eps_px, closed=True).reshape(-1, 2)
        # contour points are (col, row) == (x-index, y-index); map to cell centers.
        return [
            (origin_x + (c + 0.5) * resolution, origin_y + (r + 0.5) * resolution)
            for c, r in pts
        ]

    # Outer boundary = the largest top-level (parent == -1) contour by area.
    outer_idx = -1
    outer_area = 0.0
    for i, h in enumerate(hierarchy):
        if h[3] == -1:  # top-level
            a = cv2.contourArea(contours[i])
            if a > outer_area:
                outer_area, outer_idx = a, i
    if outer_idx < 0:
        return [], []

    boundary = to_world(contours[outer_idx])
    if len(boundary) < 3:
        return [], []

    # Holes of the outer contour = interior obstacles / unmapped pockets.
    obstacles: List[List[XY]] = []
    child = hierarchy[outer_idx][2]
    while child != -1:
        if cv2.contourArea(contours[child]) * cell_area >= min_obstacle_area_m2:
            poly = to_world(contours[child])
            if len(poly) >= 3:
                obstacles.append(poly)
        child = hierarchy[child][0]  # next sibling

    return boundary, obstacles


def from_occupancy_grid(msg, **kwargs) -> Tuple[List[XY], List[List[XY]]]:
    """Adapter for a ``nav_msgs/OccupancyGrid`` message.

    Keyword args are forwarded to :func:`grid_to_polygons`.
    """
    info = msg.info
    grid = np.asarray(msg.data, dtype=np.int16).reshape(
        (info.height, info.width)
    )
    return grid_to_polygons(
        grid,
        resolution=info.resolution,
        origin_x=info.origin.position.x,
        origin_y=info.origin.position.y,
        **kwargs,
    )


def points_to_ros(points: List[XY]):
    """Convert (x, y) tuples to a list of ``geometry_msgs/Point`` (z=0)."""
    from geometry_msgs.msg import Point

    return [Point(x=float(x), y=float(y), z=0.0) for x, y in points]
