"""Hardware-true geometry tests (docs/SIM_PLATFORM.md §6 stage 4a).

    python3 -m rover_sim.tests.test_geom [--device cpu]

G1  Rect perimeter-sampling exactness: the sampled min point-clearance is
    compared against an INDEPENDENT exact segment-to-rectangle distance
    (crossing-aware segment-segment distances + segment-endpoint-to-rect).
    Asserts the analytic error bound `max_spacing / 2` and the contact
    verdict pattern (conservative margin => zero missed contacts).
G2  Lidar mount: (a) parity — with lidar_mount_x = 0.0 the batched scan is
    bit-identical to the original (mount-free) raycast expression;
    (b) geometry — a flat wall ahead: the front-beam range shifts by ~dx.
G3  Gate presets: GATE_PRESETS['sim_evo'] == GateConfig() field-by-field;
    'monitor_default' == the ROS monitor's declare_parameter values.
G4  (CUDA only) rect body + mount capture a CUDA graph in RolloutEngine and
    report finite metrics. Parent runs this on the Spark.
G5  Legacy-default regression: RoverConfig() keeps the circle + center
    raycast, and a default scan equals the legacy expression (G2a).
G6  Batched rect integration: verdicts through the real env.step path
    (per-env wall broadcast; regression for the flattened-sample bug that
    only showed at B > 1).

E1-E2-class determinism is CPU-pinned; G4 owns the GPU. Scan paths covered
for the mount shift: BatchedEnv.scan (Fp32 base), Fp32Env.scan, Fp16Env
eager scan, rayscan.fused_scan, and SimRover.scan (numpy). G4 exercises the
Fp32 engine path; the fp16/fused/GPU paths are exercised by the parent.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    # torch built with CUDA support but no usable driver (e.g. the Hermes
    # SBC): ANY torch.cuda.* call aborts. On CPU rollouts this call is
    # semantically False — shim it (same as evo.equivalence_probe).
    torch.cuda.is_current_stream_capturing = lambda: False

from ..core.geom import (BodyConfig, HW_BODY, HW_LIDAR_MOUNT_X,
                         rect_perimeter_offsets, rect_max_sample_spacing)
from ..core.world import World
from ..core.rover import RoverConfig
from ..core.gate import GateConfig, GATE_PRESETS
from ..core.batched import BatchedEnv

HW_L, HW_W = 0.2667, 0.186


# ---------------------------------------------------------------- exact geom
def _pt_seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, c):
    return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)


def _seg_intersect(p1, q1, p2, q2):
    d1 = _orient(p2, q2, p1)
    d2 = _orient(p2, q2, q1)
    d3 = _orient(p1, q1, p2)
    d4 = _orient(p1, q1, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if abs(d1) < 1e-12 and _on_seg(p2, q2, p1):
        return True
    if abs(d2) < 1e-12 and _on_seg(p2, q2, q1):
        return True
    if abs(d3) < 1e-12 and _on_seg(p1, q1, p2):
        return True
    if abs(d4) < 1e-12 and _on_seg(p1, q1, q2):
        return True
    return False


def _seg_seg_dist(p1, q1, p2, q2):
    if _seg_intersect(p1, q1, p2, q2):
        return 0.0
    return min(_pt_seg_dist(p1, p2, q2), _pt_seg_dist(q1, p2, q2),
               _pt_seg_dist(p2, p1, q1), _pt_seg_dist(q2, p1, q1))


def _rect_corners(cx, cy, th, L, W):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    lc = np.array([[-L / 2, -W / 2], [L / 2, -W / 2],
                   [L / 2, W / 2], [-L / 2, W / 2]])
    return lc @ R.T + [cx, cy]


def _pt_rect_dist(p, cx, cy, th, L, W):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    lp = R.T @ (p - np.array([cx, cy]))
    dx = max(abs(lp[0]) - L / 2, 0.0)
    dy = max(abs(lp[1]) - W / 2, 0.0)
    return float(np.hypot(dx, dy))


def exact_rect_distance(world, cx, cy, th, L, W):
    """Independent exact min distance from the rect (boundary + region) to
    any wall segment: min over {endpoint-to-rect, rect-edge-to-segment}."""
    corners = _rect_corners(cx, cy, th, L, W)
    best = np.inf
    for seg in world.segments:
        a, b = seg[0:2], seg[2:4]
        best = min(best, _pt_rect_dist(a, cx, cy, th, L, W),
                   _pt_rect_dist(b, cx, cy, th, L, W))
        for i in range(4):
            best = min(best, _seg_seg_dist(corners[i], corners[(i + 1) % 4],
                                           a, b))
    return best


def sampled_perimeter_min(world, off, cx, cy, th):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts = off @ R.T + [cx, cy]
    return min(world.clearance(float(p[0]), float(p[1])) for p in pts)


# ---------------------------------------------------------------------- G1
def g1_exactness():
    # A small arena with a corner case (two interior walls meeting the
    # shell; poses can straddle a corner and cross both walls).
    segs = np.array([
        [-2.0, 0.0, 5.0, 0.0],      # shell floor
        [5.0, 0.0, 5.0, 4.0],       # shell right
        [-2.0, 0.0, -2.0, 4.0],     # shell left
        [0.5, 2.0, 4.0, 2.0],       # interior horizontal wall
        [3.0, 0.5, 3.0, 2.0],       # interior vertical (T corner)
    ], dtype=np.float64)
    world = World(segs, (-2.0, 0.0, 5.0, 4.0), (0.0, 1.0, 0.0))
    rng = np.random.default_rng(20261004)
    n_poses = 900
    cx = rng.uniform(-1.6, 4.6, n_poses)
    cy = rng.uniform(0.3, 3.7, n_poses)
    th = rng.uniform(-np.pi, np.pi, n_poses)

    for n in (32, 64, 128):
        off = rect_perimeter_offsets(HW_L, HW_W, n)
        maxsp = rect_max_sample_spacing(HW_L, HW_W, n)
        bound = maxsp / 2.0
        errs, faint, tot_mis, fneg, fpos, band = [], [], 0, 0, 0, 0
        for k in range(n_poses):
            samp = sampled_perimeter_min(world, off, cx[k], cy[k], th[k])
            ex = exact_rect_distance(world, cx[k], cy[k], th[k], HW_L, HW_W)
            errs.append(abs(samp - ex))
            faint.append((samp, ex))
            sc = samp < bound                  # conservative margin
            ec = ex <= 0.0
            if sc != ec:
                tot_mis += 1
                if ec and not sc:
                    fneg += 1
                elif sc and not ec:
                    fpos += 1
            if 0.0 < ex < bound:
                band += 1
        errs = np.array(errs)
        maxerr = float(errs.max())
        assert maxerr <= bound + 1e-9, (n, maxerr, bound)
        # Conservative margin: a sampled contact is never a false negative,
        # and every mismatch is a false POSITIVE inside the band.
        assert fneg == 0, (n, "missed contacts", fneg)
        assert tot_mis == fpos, (n, tot_mis, fpos)
        assert fpos <= band
        # Torch RectBody must reproduce the numpy sampled min (integration).
        rb_ok = True
        for k in range(0, n_poses, 97):
            rb = _torch_rect_min(world, n, cx[k], cy[k], th[k])
            rb_ok &= abs(rb - faint[k][0]) < 1e-5
        assert rb_ok, (n, "RectBody/numpy sampled-min mismatch")
        rate = 100.0 * (1.0 - tot_mis / n_poses)
        print(f"G1 ok: n={n:3d} max_spacing={maxsp:.5f} half={bound:.5f} "
              f"max_abs_err={maxerr:.6f} (<= {bound:.6f}) | verdict agreement "
              f"{rate:.3f}% (fneg={fneg}, fpos={fpos} in band={band}) "
              f"| RectBody==numpy")


def _torch_rect_min(world, n, cx, cy, th):
    from ..core.geom import RectBody
    body = BodyConfig(shape="rect", length=HW_L, width=HW_W,
                      collision_samples=n)
    rb = RectBody(body, 1, "cpu")

    def clearance_fn(px, py):        # grid contract: [B, n] -> [B, n]
        return torch.tensor([[world.clearance(float(a), float(b))
                              for a, b in zip(rx, ry)]
                             for rx, ry in zip(px.tolist(), py.tolist())])
    px = torch.tensor([cx], dtype=torch.float32)
    py = torch.tensor([cy], dtype=torch.float32)
    tt = torch.tensor([th], dtype=torch.float32)
    d = float(rb.clearance(clearance_fn, px, py, tt)[0])
    return d


# ---------------------------------------------------------------------- G2
def _legacy_scan_ref(env):
    """The original mount-free batched raycast, inline (no mount branch)."""
    c = env.cfg
    ang = env.theta.unsqueeze(1) + env._beam_offsets
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
    p = torch.stack([env.x, env.y], dim=1)
    q = env._a - p.unsqueeze(1)
    cross_eq = (env._e[:, :, 0] * q[:, :, 1] - env._e[:, :, 1] * q[:, :, 0])
    cross_ed = (env._e[:, None, :, 0] * d[:, :, None, 1]
                - env._e[:, None, :, 1] * d[:, :, None, 0])
    cross_dq = (d[:, :, None, 0] * q[:, None, :, 1]
                - d[:, :, None, 1] * q[:, None, :, 0])
    t = cross_eq.unsqueeze(1) / cross_ed
    s = cross_dq / cross_ed
    hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & torch.isfinite(t)
    t = torch.where(hit, t, torch.full_like(t, torch.inf))
    r = t.min(dim=2).values.clamp(max=c.lidar_max_range)
    return r.clamp(min=0.02)          # noise=0, dropout=0 -> just the clamp


def _wall_world(d):
    return World(np.array([[d, -8.0, d, 8.0]], dtype=np.float64),
                 (-10.0, -10.0, d + 10.0, 10.0), (0.0, 0.0, 0.0))


def _mk_env(batch, cfg, world):
    env = BatchedEnv(batch, cfg, seed=1, device="cpu")
    env._worlds = [world] * batch
    env._build_segments()
    env.x.zero_()
    env.y.zero_()
    env.theta.zero_()
    env.noise = lambda std, *shape: torch.zeros(
        shape if shape else (env.B,), device=env.x.device)
    return env


def g2_mount():
    d = 3.0
    world = _wall_world(d)
    B = 4

    base = RoverConfig()
    base.lidar_dropout_p = 0.0
    assert base.lidar_mount_x == 0.0
    env0 = _mk_env(B, base, world)
    r0 = env0.scan()

    # (a) parity: mount 0.0 == the legacy mount-free expression, bit-exact.
    ref = _legacy_scan_ref(env0)
    assert torch.equal(r0, ref), "mount=0.0 scan not bit-identical to legacy"
    assert torch.equal(env0.scan(), r0), "scan not deterministic (mount 0.0)"
    print(f"G2a ok: lidar_mount_x=0.0 scan bit-identical to legacy "
          f"(torch.equal, {tuple(r0.shape)})")

    # (b) geometry: mount shifts the ray origin forward; front beam range
    # drops by ~dx for a flat wall directly ahead.
    cfg1 = RoverConfig()
    cfg1.lidar_dropout_p = 0.0
    cfg1.lidar_mount_x = HW_LIDAR_MOUNT_X
    env1 = _mk_env(B, cfg1, world)
    r1 = env1.scan()

    front0 = r0[:, 0]
    front1 = r1[:, 0]
    exp0, exp1 = d, d - HW_LIDAR_MOUNT_X
    assert torch.allclose(front0, torch.full_like(front0, exp0), atol=1e-4), \
        front0.tolist()
    assert torch.allclose(front1, torch.full_like(front1, exp1), atol=1e-4), \
        front1.tolist()
    shift = (front0 - front1)
    assert torch.allclose(shift, torch.full_like(shift, HW_LIDAR_MOUNT_X),
                          atol=1e-4), shift.tolist()
    print(f"G2b ok: front beam {float(front0[0]):.5f} -> "
          f"{float(front1[0]):.5f} m (shift {float(shift[0]):.5f} "
          f"~= dx {HW_LIDAR_MOUNT_X}); expected {exp0:.5f} -> {exp1:.5f}")


# ---------------------------------------------------------------------- G3
def g3_presets():
    defaults = GateConfig()
    sim = GateConfig(**GATE_PRESETS["sim_evo"])
    for f in defaults.__dataclass_fields__:
        assert getattr(sim, f) == getattr(defaults, f), f
    print("G3 ok: GATE_PRESETS['sim_evo'] == GateConfig() (all "
          f"{len(defaults.__dataclass_fields__)} fields)")

    mon = GateConfig(**GATE_PRESETS["monitor_default"])
    # Values from src/tractor_bringup/.../lidar_safety_monitor.py
    # declare_parameter defaults: L54 .20, L55 .40, L56 .40, L57 .05, L58 4.0.
    assert mon.stop_distance == 0.20
    assert mon.stop_distance_rear == 0.40
    assert mon.slow_distance == 0.40
    assert mon.hysteresis == 0.05
    assert mon.max_eval_distance == 4.0
    # Fields the monitor shares with the sim defaults (L70/L71 + others).
    assert mon.robot_front_offset == 0.06
    assert mon.robot_half_width == 0.12
    assert mon.min_block_points == 3 and mon.block_scans == 2
    assert mon.min_valid_range == 0.05 and mon.min_block_duration == 0.3
    assert mon.track_width == 0.154
    assert mon.side_stop_distance is None       # no monitor equivalent
    print("G3 ok: GATE_PRESETS['monitor_default'] == monitor defaults "
          "(0.20/0.40, slow 0.40, hyst 0.05, max_eval 4.0)")


# ---------------------------------------------------------------------- G4
def g4_capture(dev):
    if not str(dev).startswith("cuda"):
        print("G4 skip: capture path is CUDA-only (--device cuda)")
        return
    from ..runner import FunctionPolicy, RolloutEngine
    cfg = RoverConfig()
    cfg.body = HW_BODY
    cfg.lidar_mount_x = HW_LIDAR_MOUNT_X
    eng = RolloutEngine(1, 8, seed=777_000, noise_seed=12345, device=dev,
                        rover_cfg=cfg, fp16=False)
    m = eng.run(FunctionPolicy(lambda o, p: torch.tanh(o[:, :2]),
                               graph_safe=True), 300, graph=True)
    assert not getattr(eng, "_graph_broken", False), "capture failed"
    assert getattr(eng, "_graphs", None), "no graph captured"
    for k in ("dist_m", "collisions", "rooms", "cells"):
        assert torch.isfinite(m[k]).all(), k
    eng.drop_graphs()
    print(f"G4 ok: rect body + mount captured a graph "
          f"(dist_m {float(m['dist_m'].mean()):.3f}, "
          f"collisions {int(m['collisions'].sum())})")


# ---------------------------------------------------------------------- G5
def g5_defaults():
    c = RoverConfig()
    assert c.robot_radius == 0.14, c.robot_radius
    assert c.body is None
    assert c.lidar_mount_x == 0.0
    assert c.camera_mount_x == 0.0
    b = BodyConfig()
    assert b.shape == "circle" and b.radius == 0.14
    assert (b.length, b.width) == (0.2667, 0.186)
    assert HW_BODY.shape == "rect"
    assert (HW_BODY.length, HW_BODY.width) == (0.2667, 0.186)
    assert abs(HW_BODY.width / 2.0 - 0.093) < 1e-12      # outer track edge
    # Legacy default scan equals the mount-free expression (bit-exact).
    env = _mk_env(2, RoverConfig(), _wall_world(3.0))
    env.cfg.lidar_dropout_p = 0.0
    assert torch.equal(env.scan(), _legacy_scan_ref(env))
    # Legacy circle collision is unchanged (no rect buffers).
    assert env._rect is None
    print("G5 ok: legacy defaults intact (r=0.14, body=None, mounts=0.0; "
          "default scan bit-identical)")


# ---------------------------------------------------------------------- G6
def g6_batched_rect():
    """Rect collision through the real batched env (per-env wall broadcast).

    Regression: the first implementation flattened [B, n] samples and would
    crash for B > 1 against the [B, M, 2] wall tensor (caught in parent
    audit; the CUDA gate had not run yet).
    """
    segs = np.array([[3.0, -8.0, 3.0, 8.0]], dtype=np.float64)
    world = World(segs, (-2.0, -10.0, 12.0, 10.0), (0.0, 0.0, 0.0))
    cfg = RoverConfig()
    cfg.body = HW_BODY
    cfg.lidar_mount_x = HW_LIDAR_MOUNT_X
    cfg.lidar_dropout_p = 0.0

    # (a) deterministic verdicts through env.step (B = 4).
    B, xs = 4, [2.6, 2.8, 2.95, 3.2]
    env = _mk_env(B, cfg, world)
    env.x[:] = torch.tensor(xs, dtype=torch.float32)
    env.y[:] = 0.0
    env.theta[:] = 0.0
    env.step(torch.tensor([[0.5, 0.5]] * B, dtype=torch.float32), dt=1e-4)
    # rect spans x +- 0.13335; wall at 3.0: gaps 0.2666 / 0.0667 -> no
    # contact; front edge across the wall -> contact; wall behind the rear
    # edge (gap 0.0667) -> no contact.
    exp = [False, False, True, False]
    assert env.collided.tolist() == exp, env.collided.tolist()

    # (b) random x sweep vs the independent exact distance: sampled min
    # within the Lipschitz bound, no true contact missed.
    rng = np.random.default_rng(7)
    B2 = 64
    xs2 = rng.uniform(2.0, 3.4, B2)
    ex = np.array([exact_rect_distance(world, float(x), 0.0, 0.0, HW_L, HW_W)
                   for x in xs2])
    env2 = _mk_env(B2, cfg, world)
    env2.x[:] = torch.tensor(xs2, dtype=torch.float32)
    env2.y[:] = 0.0
    env2.theta[:] = 0.0
    margin = env2._rect.margin
    rect2 = env2._rect
    assert rect2 is not None
    dd = rect2.clearance(env2._clearance_grid, env2.x, env2.y,
                         env2.theta).numpy().astype(np.float64)
    maxerr = float(np.abs(dd - ex).max())
    assert maxerr <= margin + 1e-4, (maxerr, margin)
    missed = int(((ex <= 0.0) & ~(dd < margin)).sum())
    assert missed == 0, missed
    r = env2.scan()
    assert r.shape == (B2, cfg.n_beams) and torch.isfinite(r).all()
    print(f"G6 ok: batched rect B={B} verdicts {exp}; random sweep B={B2} "
          f"max|sampled-exact|={maxerr:.6f} <= margin {margin:.6f}, "
          f"missed contacts={missed}; mount scan finite")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    g1_exactness()
    g2_mount()
    g3_presets()
    g5_defaults()
    g6_batched_rect()
    g4_capture(args.device)
    print("ALL GEOM CHECKS PASSED")


if __name__ == "__main__":
    main()
