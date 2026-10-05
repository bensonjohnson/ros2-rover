"""Egocentric 2D camera renderer — SIM_PLATFORM stage 4b (docs §4/§6).

The camera obs channel: one horizontal ray fan per env, centred on the
heading, walls drawn as range-shaded bands. This is a 2D sim, so every ROW
of a frame is identical — a "pixel" is fully described by its COLUMN — and
the shading is a function of the ray range alone:

    brightness = round(255 * (max_range - r) / max_range)      [r clamped]

so a nearer wall is brighter, a wall at ``max_range`` (and a no-hit ray,
whose range clamps to ``max_range``) is black (0), and a wall at the camera
is 255. ``channels=1`` is grayscale; ``channels=3`` replicates the same luma
across R/G/B (RGB-by-replication).

Two producers share the exact same fan + shading math:

* ``CameraRenderer`` — batched torch, reads the engine's cached wall tensors
  (``env._a`` / ``env._e``) with the SAME cross-product raycast as
  ``batched.py scan()`` and writes into a PREALLOCATED uint8 ``[B, H, W, C]``
  buffer. Graph-capture-safe by construction: static output buffer,
  ``out=``/in-place shading, no host sync, no data-dependent control flow.
  One env per row b, column ``j`` at world angle
  ``theta_b - hfov/2 + j * hfov/(W-1)`` (column 0 = leftmost); ``W == 1``
  degenerates to a single forward ray.
* ``render_np`` — numpy twin for CPU tools/tests, one env at a time through
  the ``World.raycast`` API.

Everything here is OPT-IN. With ``mount_x == 0.0`` the ray origin is the
robot centre, matching the legacy ray geometry; pass
``geom.HW_CAMERA_MOUNT_X`` (0.12085 m) to opt into the D435i mount. The
optional ``noise_std`` / ``dropout_p`` default to 0.0 (no-ops) and — to keep
the documented row-constant property — are applied per COLUMN (one draw per
ray), documented at the shading site.

    python -m rover_sim.core.render --bench [--device cuda]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CameraConfig:
    """Camera fan + shading config. Defaults == a 96x96 grayscale 80-degree
    D435i-like camera with a 6 m clip, centred on the robot (mount_x=0.0,
    the legacy ray origin). All fields are opt-in knobs."""

    width: int = 96
    height: int = 96
    channels: int = 1              # 1 = grayscale, 3 = RGB-by-replication
    hfov_rad: float = 1.396        # ~80 deg, matches the D435i note (docs §4)
    max_range: float = 6.0         # clip: r >= max_range -> black (no-hit)
    noise_std: float = 0.0         # pixel-space gaussian on brightness (opt-in)
    dropout_p: float = 0.0         # black-pixel dropout (opt-in)
    mount_x: float = 0.0           # forward offset of the render origin;
    #                                0.0 = robot centre (legacy geometry),
    #                                geom.HW_CAMERA_MOUNT_X = hardware truth

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("camera width/height must be >= 1")
        if self.channels not in (1, 3):
            raise ValueError("camera channels must be 1 (gray) or 3 (RGB)")
        if self.hfov_rad <= 0.0:
            raise ValueError("camera hfov_rad must be > 0")
        if self.max_range <= 0.0:
            raise ValueError("camera max_range must be > 0")

    @property
    def shape(self) -> tuple:
        return (self.height, self.width, self.channels)


def column_angles(cfg: CameraConfig) -> np.ndarray:
    """[W] world-frame offsets from the heading, column 0 = leftmost,
    symmetric about 0 (W == 1 -> a single forward ray at offset 0)."""
    if cfg.width == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-cfg.hfov_rad / 2.0, cfg.hfov_rad / 2.0, cfg.width)


class CameraRenderer:
    """Batched torch camera fan -> preallocated uint8 ``[B, H, W, C]``.

    Buffers are allocated ONCE here (bind/init time), never mid-tick; only
    the per-tick raycast intermediates are fresh, exactly the established
    pattern of ``batched.py scan()`` (which is far larger at 360 beams). The
    OUTPUT is a caller-owned static buffer written in place with ``copy_`` /
    ``out=`` ops, so a captured CUDA graph replays with no fresh output
    allocation and no host sync.
    """

    def __init__(self, cfg: CameraConfig, batch: int, device):
        self.cfg = cfg
        self.B = int(batch)
        self.device = torch.device(device)
        self.col_offsets = torch.as_tensor(column_angles(cfg),
                                           dtype=torch.float32,
                                           device=self.device)
        # static shading scratch [B, W] (reused every call; out= target).
        self._br = torch.empty(self.B, cfg.width, dtype=torch.float32,
                               device=self.device)

    @torch.no_grad()
    def ranges(self, env) -> torch.Tensor:
        """Per-column ray ranges ``[B, W]`` (clamped to ``max_range``) from
        ``env._a`` / ``env._e`` / ``env.x`` / ``env.y`` / ``env.theta``.

        Mirrors ``batched.py scan()``'s cross-product solve
        (t = cross(e, q) / cross(e, d), s = cross(d, q) / cross(e, d)) with
        the camera fan in place of the lidar beams."""
        cfg = self.cfg
        th = env.theta
        ang = th.unsqueeze(1) + self.col_offsets            # [B, W]
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)
        p = torch.stack([env.x, env.y], dim=1)              # [B, 2]
        if cfg.mount_x != 0.0:
            # Same default-off mount branch as the stage-4a lidar mount.
            p = p + cfg.mount_x * torch.stack(
                [torch.cos(th), torch.sin(th)], dim=1)
        a = env._a                                          # [B, M, 2]
        e = env._e
        q = a - p.unsqueeze(1)                              # [B, M, 2]
        cross_eq = (e[:, :, 0] * q[:, :, 1]
                    - e[:, :, 1] * q[:, :, 0])              # [B, M]
        cross_ed = (e[:, None, :, 0] * sin_a[:, :, None]
                    - e[:, None, :, 1] * cos_a[:, :, None])  # [B, W, M]
        cross_dq = (cos_a[:, :, None] * q[:, None, :, 1]
                    - sin_a[:, :, None] * q[:, None, :, 0])  # [B, W, M]
        t = cross_eq.unsqueeze(1) / cross_ed
        s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & torch.isfinite(t)
        t = torch.where(hit, t, torch.full_like(t, torch.inf))
        return t.min(dim=2).values.clamp(max=cfg.max_range)  # [B, W]

    @torch.no_grad()
    def render(self, env, out: torch.Tensor) -> torch.Tensor:
        """Shade ``ranges(env)`` into ``out`` ``[B, H, W, C]`` uint8, in
        place. Rows are constant and every channel of a column equals the
        column's luma (broadcast write). Returns ``out``."""
        cfg = self.cfg
        r = self.ranges(env)                                # [B, W]
        W = cfg.width
        br = self._br
        torch.div(r, cfg.max_range, out=br)
        br.neg_().add_(1.0).mul_(255.0)                     # 255*(1 - r/max_range)
        # Optional, per COLUMN (one draw per ray) so all H rows of a column
        # stay identical — a "pixel" here is its column (2D sim).
        if cfg.noise_std:
            br.add_(torch.randn(self.B, W, device=self.device) * cfg.noise_std)
        br.round_().clamp_(0.0, 255.0)
        if cfg.dropout_p:
            br.masked_fill_(torch.rand(self.B, W, device=self.device)
                            < cfg.dropout_p, 0.0)
        # [B, W] -> [B, 1, W, 1] broadcast over height and channels.
        out.copy_(br.to(torch.uint8).view(br.shape[0], 1, W, 1))
        return out


def render_np(world, x: float, y: float, th: float,
              cfg: CameraConfig) -> np.ndarray:
    """Numpy twin: one env's frame ``[H, W, C]`` uint8 via ``World.raycast``.

    Same fan, origin (mount-shift) and shading as :class:`CameraRenderer`;
    float64 here vs float32 in torch, so only the range comparison is
    tolerance-matched (see rover_sim/tests/test_render.py R1)."""
    ang = th + column_angles(cfg)                          # [W]
    ox, oy = float(x), float(y)
    if cfg.mount_x != 0.0:
        ox += cfg.mount_x * math.cos(th)
        oy += cfg.mount_x * math.sin(th)
    r = world.raycast(ox, oy, ang, cfg.max_range)          # [W]
    br = 255.0 * (1.0 - np.clip(r / cfg.max_range, 0.0, 1.0))
    if cfg.noise_std:
        br = br + np.random.default_rng().standard_normal(cfg.width) \
            * cfg.noise_std
    br = np.round(br).clip(0.0, 255.0)
    if cfg.dropout_p:
        drop = np.random.default_rng().random(cfg.width) < cfg.dropout_p
        br[drop] = 0.0
    col = br.astype(np.uint8)[None, :, None]               # [1, W, 1]
    frame = np.repeat(col, cfg.height, axis=0)             # [H, W, 1]
    return np.repeat(frame, cfg.channels, axis=2)          # [H, W, C]


def _bench(device: str, batch: int, iters: int) -> None:
    """Time the torch renderer at B=batch on a small walled scene."""
    import time

    cfg = CameraConfig()
    dev = torch.device(device)
    B = int(batch)
    segs = np.array([[3.0, -3.0, 3.0, 3.0],     # right wall
                     [-3.0, -3.0, -3.0, 3.0],   # left wall
                     [-3.0, 3.0, 3.0, 3.0],     # top wall
                     [-3.0, -3.0, 3.0, -3.0]],  # bottom wall
                    dtype=np.float32)

    class _E:
        pass

    env = _E()
    env._a = torch.as_tensor(np.tile(segs[:, 0:2], (B, 1, 1)), device=dev)
    env._e = torch.as_tensor(np.tile(segs[:, 2:4] - segs[:, 0:2], (B, 1, 1)),
                             device=dev)
    env.x = torch.zeros(B, device=dev)
    env.y = torch.zeros(B, device=dev)
    env.theta = torch.zeros(B, device=dev)
    out = torch.empty(B, cfg.height, cfg.width, cfg.channels,
                      dtype=torch.uint8, device=dev)
    rend = CameraRenderer(cfg, B, dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(3):
        rend.render(env, out)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        rend.render(env, out)
    sync()
    dt = (time.perf_counter() - t0) / float(iters)
    rays = B * cfg.width
    print(f"render bench: B={B} {cfg.width}x{cfg.height}x{cfg.channels} "
          f"device={device} iters={iters} {dt * 1e3:.3f} ms/tick "
          f"{rays / dt / 1e6:.2f} Mrays/s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true",
                    help="time the torch renderer at B=8192")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    if args.bench:
        _bench(args.device, args.batch, args.iters)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
