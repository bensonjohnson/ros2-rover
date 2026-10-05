"""Hardware-true geometry: body shape, sensor mounts, perimeter sampling.

SIM_PLATFORM stage 4a (docs/SIM_PLATFORM.md §4/§6). Every mode here is
OPT-IN: the legacy ``RoverConfig`` still defaults to a collision CIRCLE
(radius 0.14 m) centered on the robot and to a lidar whose raycast origin
IS the robot center, so default runs execute the original expressions
byte-for-byte. The hardware-true variants reproduce the as-built tractor:

    plan-view footprint 0.2667 (x, forward) x 0.186 (y) m
    (outer track edges +-0.093 m)           -- docs §4 table, derived from
                                               urdf/tractor.urdf.xacro
    lidar (LD19) mount  x = +0.07635 m      -- URDF laser_joint (0.057 m
                                               behind the bumper)
    camera (D435i) mount x = +0.12085 m     -- URDF camera_joint

Collision for a rect body is PERIMETER POINT SAMPLING: the footprint
outline (4 corners + points along each edge, allocated proportional to
edge length) is rotated into the world at the pose and each sample's
distance-to-nearest-wall-segment is evaluated with the SAME batched
point-clearance kernel the legacy circle uses. The body is "in contact"
when ``min over samples < margin`` with ``margin = max_spacing / 2``.

Why that margin: distance-to-walls is 1-Lipschitz, so for any boundary
point p the nearest sample s satisfies |d(p) - d(s)| <= |p - s| <=
max_spacing/2. Hence ``sampled_min`` overshoots the true min by at most
max_spacing/2, and comparing ``sampled_min < margin`` can never MISS a
true contact (no false negatives) — it only reports contact slightly
early inside a band of width ``margin`` (documented in
rover_sim/tests/test_geom.py). ``max_spacing`` for n samples of a
``length x width`` rectangle is at most ``max(length, width) / (n/4)``
and is computed exactly by :func:`rect_max_sample_spacing`.

This module imports only numpy/torch — it must NOT import rover.py or
batched.py (they import it), so there is no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BodyConfig:
    """Robot collision footprint. Defaults == the legacy circle exactly.

    shape='circle'   -> :attr:`radius` is the collision radius (0.14 m).
    shape='rect'     -> the footprint is :attr:`length` (x, forward) x
                        :attr:`width` (y) m, perimeter-sampled with
                        :attr:`collision_samples` points.
    """

    shape: str = "circle"
    radius: float = 0.14
    length: float = 0.2667
    width: float = 0.186
    collision_samples: int = 32


# Hardware-true body (docs/SIM_PLATFORM.md §4 table).
HW_BODY = BodyConfig(shape="rect", radius=0.14, length=0.2667, width=0.186,
                     collision_samples=32)

# Sensor mounts, forward (+x) offsets from the base center (docs §4 table;
# URDF laser_joint / camera_joint). The sim is 2D, so only x matters here.
HW_LIDAR_MOUNT_X = 0.07635     # LD19: 0.057 m behind the front bumper
HW_CAMERA_MOUNT_X = 0.12085    # D435i (used by stage 4b, not yet wired)


def rect_perimeter_offsets(length: float, width: float, n: int) -> np.ndarray:
    """Deterministic [n, 2] float64 perimeter sample offsets of a centered
    ``length x width`` rectangle, in perimeter order (so consecutive rows
    are adjacent samples and the closing edge wraps to row 0).

    Layout: the 4 corners are always included; the remaining ``n - 4``
    samples are distributed across the 4 edges in proportion to edge
    length (largest-remainder rounding, ties broken by edge index) and
    placed at equal fractions ``j / (k + 1)`` along each edge (k = the
    edge's sample count). A closed walk over the rows visits the boundary
    once with no duplicate corners.

    Max sample spacing (the largest gap between consecutive boundary
    samples) is ``max_edge(length / (k_e + 1))``; see
    :func:`rect_max_sample_spacing`. For the hardware body (0.2667 x
    0.186, n = 32) this is 0.02963 m, half-spacing 0.01482 m.
    """
    hx, hy = length / 2.0, width / 2.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    edge_lens = [float(length), float(width), float(length), float(width)]
    perim = 2.0 * (length + width)

    n = max(int(n), 4)
    m = n - 4
    counts = [0, 0, 0, 0]
    if m > 0 and perim > 0.0:
        raw = [m * Le / perim for Le in edge_lens]
        counts = [int(np.floor(r)) for r in raw]
        rem = m - sum(counts)
        order = sorted(range(4), key=lambda i: (-(raw[i] - counts[i]), i))
        for i in order[:rem]:
            counts[i] += 1

    pts: list[tuple[float, float]] = []
    for (a, b), k in zip(edges, counts):
        pts.append(a)                              # this edge's start corner
        for j in range(1, k + 1):
            f = j / (k + 1.0)
            pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return np.asarray(pts, dtype=np.float64)


def rect_max_sample_spacing(length: float, width: float, n: int) -> float:
    """Largest gap between consecutive perimeter samples (perimeter walk).

    The sampled-distance overshoot of the exact rect-to-wall distance is
    bounded by half of this value (1-Lipschitz argument in the module
    docstring); :class:`RectBody` uses exactly ``this / 2`` as its
    collision margin.
    """
    off = rect_perimeter_offsets(length, width, n)
    if off.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(np.diff(np.vstack([off, off[:1]]), axis=0), axis=1)
    return float(d.max())


class RectBody:
    """Graph-safe perimeter sampler + collision evaluator for a rect body.

    Buffers are allocated ONCE here (bind/init time), never mid-tick: the
    [B, n] sample-point scratch is written in place with ``out=`` /
    in-place ops, mirroring the static-buffer discipline used elsewhere for
    CUDA-graph replay. The only fresh per-tick tensors are the [B] cos/sin
    of the heading (the surrounding scan already allocates far larger
    intermediates).
    """

    def __init__(self, body: BodyConfig, batch: int, device):
        self.length = float(body.length)
        self.width = float(body.width)
        self.n = int(body.collision_samples)
        off = rect_perimeter_offsets(self.length, self.width, self.n)
        self.off = torch.as_tensor(off, device=device, dtype=torch.float32)
        self.max_spacing = rect_max_sample_spacing(self.length, self.width,
                                                   self.n)
        self.margin = 0.5 * self.max_spacing
        # Preallocated, reused scratch: rotated world-frame sample points.
        self._sx = torch.empty(batch, self.n, device=device,
                               dtype=torch.float32)
        self._sy = torch.empty(batch, self.n, device=device,
                               dtype=torch.float32)

    def sample_points(self, px: torch.Tensor, py: torch.Tensor,
                      th: torch.Tensor):
        """Rotate the [n, 2] offsets by heading ``th`` [B] and translate to
        (px, py) [B], writing into the reused scratch. Returns [B, n] x/y.

        world = center + R(th) @ offset, with R = [[cos, -sin], [sin, cos]].
        """
        ca = torch.cos(th).unsqueeze(1)
        sa = torch.sin(th).unsqueeze(1)
        ox = self.off[:, 0].unsqueeze(0)
        oy = self.off[:, 1].unsqueeze(0)
        torch.addcmul(px.unsqueeze(1), ca, ox, out=self._sx)
        self._sx.addcmul_(sa, oy, value=-1.0)
        torch.addcmul(py.unsqueeze(1), sa, ox, out=self._sy)
        self._sy.addcmul_(ca, oy, value=1.0)
        return self._sx, self._sy

    def clearance(self, clearance_fn, px: torch.Tensor, py: torch.Tensor,
                  th: torch.Tensor) -> torch.Tensor:
        """Min over perimeter samples of ``clearance_fn`` (a [P] -> [P]
        point-clearance kernel). Returns [B]. Compare against
        :attr:`margin` for the (conservative) contact verdict."""
        sx, sy = self.sample_points(px, py, th)
        b = px.shape[0]
        d = clearance_fn(sx.reshape(-1), sy.reshape(-1)).reshape(b, self.n)
        return d.min(dim=1).values
