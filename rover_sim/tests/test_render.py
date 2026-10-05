"""Egocentric camera renderer tests (docs/SIM_PLATFORM.md §6 stage 4b).

    python3 -m rover_sim.tests.test_render [--device cpu]

R1  PARITY: the torch CameraRenderer and the numpy twin `render_np` on
    fixed synthetic worlds/poses — per-column ranges agree within 1e-4 and
    the uint8 frames within +-1. A flat wall ahead at distance d has the
    hand-derived column brightness; no-hit columns are 0; mount_x > 0
    shifts every column's expected range by the analytic (d - mount_x).
R2  DETERMINISM: same pose+config -> identical uint8 output twice (CPU).
R3  CONTRACT: a camera spec on a camera-less producer raises a clear
    ValueError; an engine WITH camera presents `channel_views["camera"]`
    as [B, H, W, C] uint8 sharing storage, readable through the normal
    run() policy path; flat() is unchanged (same length/slices).
R4  (CUDA only) ONE engine with camera, fp16+fused+compile, run(graph=True):
    no `_graph_broken`, graphs captured, camera buffer has nonzero variance,
    metrics finite; drop_graphs() after. Parent runs this on the Spark.
R5  LEGACY: a default engine (camera=None) allocates no camera buffer and
    presents no `"camera"` key anywhere; flat() is the pre-4b layout.
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

from ..core.render import (CameraConfig, CameraRenderer, column_angles,
                           render_np)
from ..core.world import World
from ..runner import (DEFAULT_OBS_SPEC, FunctionPolicy, ObsSpec,
                      RolloutEngine, channel_views, flat)
from ..runner import OBS_DIM


# ------------------------------------------------------------------ helpers
def _walled_world(d=2.0):
    """A single vertical wall at x=d, tall enough to span the fan."""
    segs = np.array([[d, -6.0, d, 6.0]], dtype=np.float64)
    return World(segs, (-6.0, -6.0, 6.0, 6.0), (0.0, 0.0, 0.0))


class _Env:
    """Minimal batched env view for CameraRenderer (the fields it reads)."""


def _mk_env(segs, xs, ys, ths):
    segs = np.asarray(segs, dtype=np.float32)
    B = len(xs)
    e = _Env()
    e._a = torch.as_tensor(np.tile(segs[:, 0:2], (B, 1, 1)))
    e._e = torch.as_tensor(np.tile(segs[:, 2:4] - segs[:, 0:2], (B, 1, 1)))
    e.x = torch.as_tensor(np.asarray(xs, np.float32))
    e.y = torch.as_tensor(np.asarray(ys, np.float32))
    e.theta = torch.as_tensor(np.asarray(ths, np.float32))
    return e


def _expected_brightness(cfg, r):
    br = 255.0 * (1.0 - np.clip(r / cfg.max_range, 0.0, 1.0))
    return np.round(br).clip(0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------- R1
def r1_parity():
    cfg = CameraConfig(width=7, height=5, channels=1, hfov_rad=1.0,
                       max_range=6.0)
    d = 2.0
    world = _walled_world(d)

    # (a) torch ranges vs the numpy World.raycast twin on several poses,
    #     including rotations, within 1e-4.
    rng = np.random.default_rng(20261005)
    xs = rng.uniform(-0.3, 0.3, 6)
    ys = rng.uniform(-0.3, 0.3, 6)
    ths = rng.uniform(-0.4, 0.4, 6)
    segs = world.segments
    env = _mk_env(segs, xs, ys, ths)
    rend = CameraRenderer(cfg, len(xs), "cpu")
    tr = rend.ranges(env).numpy().astype(np.float64)       # [B, W]
    ref = np.stack([world.raycast(float(xs[b]), float(ys[b]),
                                  float(ths[b]) + column_angles(cfg),
                                  cfg.max_range)
                    for b in range(len(xs))])
    assert np.allclose(tr, ref, atol=1e-4), np.abs(tr - ref).max()
    print(f"R1a ok: torch vs numpy ranges agree within 1e-4 "
          f"(max|dx|={np.abs(tr - ref).max():.2e}, B={len(xs)}, "
          f"W={cfg.width})")

    # (b) analytic flat wall directly ahead (theta=0, mount 0): column j at
    #     angle a_j hits x=d at r_j = d / cos(a_j); brightness by hand.
    a = column_angles(cfg)
    r_wall = d / np.cos(a)
    exp_br = _expected_brightness(cfg, r_wall)
    frame_np = render_np(world, 0.0, 0.0, 0.0, cfg)
    assert frame_np.shape == (cfg.height, cfg.width, cfg.channels)
    got = frame_np[0, :, 0]
    assert np.array_equal(got, exp_br), (got.tolist(), exp_br.tolist())
    # rows constant, channels replicated
    assert np.array_equal(frame_np, frame_np[0:1].repeat(cfg.height, 0))
    out = torch.zeros(1, cfg.height, cfg.width, cfg.channels,
                      dtype=torch.uint8)
    CameraRenderer(cfg, 1, "cpu").render(
        _mk_env(segs, [0.0], [0.0], [0.0]), out)
    gt = out[0, 0, :, 0].numpy()
    assert np.abs(gt.astype(int) - exp_br.astype(int)).max() <= 1, \
        (gt.tolist(), exp_br.tolist())
    center = cfg.width // 2
    exp_center = int(round(255.0 * (cfg.max_range - d) / cfg.max_range))
    assert gt[center] == exp_center
    print(f"R1b ok: flat wall d={d} — numpy == hand-derived brightness "
          f"{exp_br.tolist()}; torch within +-1; center col {gt[center]} "
          f"== 255*(6-{d})/6 = {exp_center}; rows constant")

    # (c) no-hit columns are black: a short wall only the centre ray sees.
    segs_short = np.array([[d, -0.05, d, 0.05]], dtype=np.float64)
    ws = World(segs_short, (-6.0, -6.0, 6.0, 6.0), (0.0, 0.0, 0.0))
    frame_s = render_np(ws, 0.0, 0.0, 0.0, cfg)
    col = frame_s[0, :, 0]
    hit = np.array(np.isfinite(ws.raycast(0.0, 0.0, a, cfg.max_range))
                   & (ws.raycast(0.0, 0.0, a, cfg.max_range) < cfg.max_range))
    assert (col[~hit] == 0).all() and col[center] > 0, col.tolist()
    print(f"R1c ok: no-hit columns black ({int((~hit).sum())} of "
          f"{cfg.width}); centre col {int(col[center])}")

    # (d) mount_x > 0 shifts the origin forward: r_j = (d - mount_x)/cos(a_j).
    mx = 0.1
    cfgm = CameraConfig(width=7, height=5, channels=1, hfov_rad=1.0,
                        max_range=6.0, mount_x=mx)
    r_m = (d - mx) / np.cos(a)
    exp_m = _expected_brightness(cfgm, r_m)
    got_m = render_np(world, 0.0, 0.0, 0.0, cfgm)[0, :, 0]
    assert np.array_equal(got_m, exp_m), (got_m.tolist(), exp_m.tolist())
    outm = torch.zeros(1, cfgm.height, cfgm.width, cfgm.channels,
                       dtype=torch.uint8)
    CameraRenderer(cfgm, 1, "cpu").render(
        _mk_env(segs, [0.0], [0.0], [0.0]), outm)
    gtm = outm[0, 0, :, 0].numpy()
    assert np.abs(gtm.astype(int) - exp_m.astype(int)).max() <= 1
    # nearer wall with the mount -> strictly brighter than mount-free
    assert (gtm >= gt).all() and gtm[center] > gt[center]
    print(f"R1d ok: mount_x={mx} — numpy == (d-mx)/cos(a) brightness; "
          f"centre {gt[center]} -> {gtm[center]} (forward shift brightens)")


# ---------------------------------------------------------------------- R2
def r2_determinism():
    cfg = CameraConfig(width=16, height=12, channels=3)
    world = _walled_world(2.5)
    segs = world.segments
    env = _mk_env(segs, [0.1, -0.2], [0.0, 0.1], [0.05, -0.1])
    rend = CameraRenderer(cfg, 2, "cpu")
    o1 = torch.zeros(2, cfg.height, cfg.width, cfg.channels, dtype=torch.uint8)
    o2 = torch.zeros_like(o1)
    rend.render(env, o1)
    rend.render(env, o2)
    assert torch.equal(o1, o2), "torch render not deterministic"
    assert o1.float().std().item() > 0, "render is all-constant"
    f1 = render_np(world, 0.1, 0.0, 0.05, cfg)
    f2 = render_np(world, 0.1, 0.0, 0.05, cfg)
    assert np.array_equal(f1, f2), "numpy render not deterministic"
    print(f"R2 ok: identical uint8 output twice (torch {tuple(o1.shape)} "
          f"std={o1.float().std().item():.1f}; numpy {f1.shape})")


# ---------------------------------------------------------------------- R3
def r3_contract():
    # (a) declared-but-unavailable camera -> clear ValueError on channel_views
    spec_cam = ObsSpec(camera=True)
    obs = torch.rand(3, OBS_DIM)
    msg = None
    try:
        channel_views(obs, spec_cam)
        raise AssertionError("expected ValueError for missing camera buffer")
    except ValueError as ex:
        msg = str(ex)
    assert msg is not None and "camera" in msg
    print(f"R3a ok: camera spec without producer raises ValueError "
          f"({msg[:58]}...)")

    # flat() layout unchanged by the camera request
    assert spec_cam.dim == OBS_DIM
    assert spec_cam.slices() == DEFAULT_OBS_SPEC.slices()
    buf = torch.zeros(3, 4, 5, 1, dtype=torch.uint8)
    views = channel_views(obs, spec_cam, camera=buf)
    assert views["camera"] is buf
    assert flat(views) is obs and flat(views).shape[1] == OBS_DIM
    print(f"R3b ok: camera view shares storage; flat() len {OBS_DIM} "
          f"unchanged; slices identical")

    # (b) engine WITH camera through the normal run() policy path
    class CamPolicy:
        obs_spec = ObsSpec(camera=True)
        graph_safe = False

        def __init__(self, expected_buf):
            self._buf = expected_buf

        def bind(self, P, G):
            pass

        def reset(self):
            pass

        def step(self, obs, prev_act):
            cam = obs["camera"]
            assert cam.shape == (eng.B, 8, 6, 3), cam.shape
            assert cam.dtype == torch.uint8, cam.dtype
            assert cam.data_ptr() == self._buf.data_ptr(), "not shared"
            assert flat(obs).shape == (eng.B, OBS_DIM)
            return torch.tanh(flat(obs)[:, :2])

    cam_cfg = CameraConfig(width=6, height=8, channels=3)
    eng = RolloutEngine(1, 3, seed=777_000, noise_seed=12345, device="cpu",
                        cells=False, camera=cam_cfg)
    assert eng._camera_buf is not None
    m = eng.run(CamPolicy(eng._camera_buf), 40)
    for k in ("dist_m", "collisions", "rooms", "cells"):
        assert torch.isfinite(m[k]).all(), k
    assert eng._camera_buf.float().std().item() > 0, "camera buffer never rendered"
    print(f"R3c ok: engine camera [B,H,W,C]={tuple(eng._camera_buf.shape)} "
          f"uint8, shared storage, read via run(); metrics finite")

    # (c) engine WITHOUT camera + camera spec -> early ValueError from run()
    eng2 = RolloutEngine(1, 2, seed=1, device="cpu", cells=False)
    try:
        eng2.run(CamPolicy(torch.zeros(2, 8, 6, 3, dtype=torch.uint8)), 5)
        raise AssertionError("expected run() ValueError for camera-less engine")
    except ValueError as ex:
        assert "camera" in str(ex)
    print("R3d ok: run() with camera spec on camera-less engine raises "
          "ValueError")


# ---------------------------------------------------------------------- R4
def r4_capture(dev):
    if not str(dev).startswith("cuda"):
        print("R4 skip: capture path is CUDA-only (--device cuda)")
        return
    cam_cfg = CameraConfig(width=32, height=32, channels=1)
    eng = RolloutEngine(1, 8, seed=777_000, noise_seed=12345, device=dev,
                        fp16=True, fused=True, compile=True, camera=cam_cfg)

    class Pol:
        obs_spec = ObsSpec(camera=True)
        graph_safe = True

        def bind(self, P, G):
            pass

        def reset(self):
            pass

        def step(self, obs, prev_act):
            return torch.tanh(flat(obs)[:, :2])

    m = eng.run(Pol(), 300, graph=True)
    assert not getattr(eng, "_graph_broken", False), "capture failed"
    assert getattr(eng, "_graphs", None), "no graph captured"
    assert eng._camera_buf.float().std().item() > 0.0, "camera buffer constant"
    for k in ("dist_m", "collisions", "rooms", "cells"):
        assert torch.isfinite(m[k]).all(), k
    eng.drop_graphs()
    print(f"R4 ok: fp16+fused+compile camera graph captured "
          f"(dist_m {float(m['dist_m'].mean()):.3f}, "
          f"cam std {eng._camera_buf.float().std().item():.1f})")


# ---------------------------------------------------------------------- R5
def r5_legacy():
    kw = dict(seed=777_000, noise_seed=12345, device="cpu", cells=False)
    eng = RolloutEngine(1, 4, **kw)
    assert eng.camera is None
    assert eng._camera_buf is None and eng._cam_renderer is None

    seen = {}

    class NoCam:
        obs_spec = DEFAULT_OBS_SPEC
        graph_safe = False

        def bind(self, P, G):
            pass

        def reset(self):
            pass

        def step(self, obs, prev_act):
            seen["keys"] = set(obs)
            seen["flat_dim"] = flat(obs).shape[1]
            return torch.tanh(flat(obs)[:, :2])

    m = eng.run(NoCam(), 30)
    assert "camera" not in seen["keys"], seen["keys"]
    assert seen["flat_dim"] == OBS_DIM, seen["flat_dim"]
    assert set(seen["keys"]) == {"scan_bins", "proprio", "prev_act", "_flat"}
    # legacy closure path is camera-free too
    m2 = eng.run_games(lambda o, p: torch.tanh(o[:, :2]), 30)
    assert "camera" not in m and "camera" not in m2
    print(f"R5 ok: default engine camera=None — no buffer, channel keys "
          f"{sorted(seen['keys'])}, flat {seen['flat_dim']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    r1_parity()
    r2_determinism()
    r3_contract()
    r5_legacy()
    r4_capture(args.device)
    print("ALL RENDER CHECKS PASSED")


if __name__ == "__main__":
    main()
