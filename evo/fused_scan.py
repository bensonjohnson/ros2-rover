#!/usr/bin/env python3
"""Fused Triton lidar scan — kills the [B, 360, M] DRAM round-trips.

The fp16 arena scan materializes ~6 intermediates of [B, 360 beams, M segs]
per tick (t, s, hit masks...) ~250 MB PER TENSOR at production B=8192: the
rollout is memory-traffic-bound, which is also why CUDA graphs only bought
1.06x. A real raytracer (or an RT core) keeps the scene on-chip and keeps
rays in registers — same idea, Triton: one program per (env, 64 beams),
segments streamed once into registers, hit-test math per (beam, seg) tile,
ONE [B,360] store. fp32 internally (more accurate than the fp16 chain).

Noise + dropout stay torch-side (same generator call order as Fp16Env —
the paired-noise semantics runs 1-7 depend on must not change).

Acceptance gates (run 8 is the live job; this benches alongside it —
TIMINGS ARE CONTENTION-INFLATED, trust the RATIO):
  P) geometry parity vs noise-zeroed Fp16Env.scan: 99.5% of beams within
     0.05 m (fp16 rounding + grazing-beam differences; walls must match)
  B) scan-only speedup at B=8192 (eager, both paths)

    python3 -m evo.fused_scan --device cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import triton
import triton.language as tl

from .arena import Fp16Env
from pnn_sim.rover import RoverConfig


@triton.jit
def _ray_scan(A, E, X, Y, TH, OUT,
              M, stride_ab, inv_nbeams, max_range, n_beams,
              BB: tl.constexpr, BM: tl.constexpr):
    """grid = (B, n_beams // BB); OUT [B, n_beams] = clean geometric ranges."""
    b = tl.program_id(0)
    pib = tl.program_id(1)
    beams = pib * BB + tl.arange(0, BB)
    b_ok = beams < n_beams
    theta = tl.load(TH + b)
    px = tl.load(X + b)
    py = tl.load(Y + b)
    ang = theta + beams.to(tl.float32) * inv_nbeams * (2.0 * 3.141592653589793)
    dx = tl.cos(ang)
    dy = tl.sin(ang)
    tmin = tl.full((BB,), float("inf"), tl.float32)
    for m0 in range(0, M, BM):
        ms = m0 + tl.arange(0, BM)
        m_ok = ms < M
        ab = b * stride_ab + ms
        ax = tl.load(A + ab * 2 + 0, mask=m_ok, other=0.0)
        ay = tl.load(A + ab * 2 + 1, mask=m_ok, other=0.0)
        ex = tl.load(E + ab * 2 + 0, mask=m_ok, other=0.0)
        ey = tl.load(E + ab * 2 + 1, mask=m_ok, other=0.0)
        # dummy pads sit at ~1e6: in fp32 they produce a huge-but-finite t
        # (fp16 dropped them via inf/NaN + isfinite). Exclude explicitly.
        m_ok = m_ok & (ax < 9.0e5)
        qax = ax - px
        qay = ay - py
        ceq = ex * qay - ey * qax                     # [BM]
        ced = (ex[None, :] * dy[:, None]
               - ey[None, :] * dx[:, None])           # [BB, BM]
        cdq = (dx[:, None] * qay[None, :]
               - dy[:, None] * qax[None, :])
        t = ceq[None, :] / ced
        s = cdq / ced
        hit = ((t > 1e-9) & (s >= 0.0) & (s <= 1.0)
               & tl.isfinite(t) & m_ok[None, :])
        t = tl.where(hit, t, float("inf"))
        tmin = tl.minimum(tmin, tl.min(t, axis=1))
    r = tl.minimum(tmin, max_range)
    tl.store(OUT + b * n_beams + beams, r, mask=b_ok)


def fused_scan(env: Fp16Env, out: torch.Tensor | None = None) -> torch.Tensor:
    """Clean (noise/dropout-free) batched scan via the fused kernel."""
    B = env.B
    nb = env.cfg.n_beams
    M = env._a.shape[1]
    r = torch.empty(B, nb, device=env.x.device, dtype=torch.float32) \
        if out is None else out
    BB, BM = 64, 32
    _ray_scan[(B, triton.cdiv(nb, BB))](
        env._a, env._e, env.x, env.y, env.theta, r,
        M, M, 1.0 / nb, env.cfg.lidar_max_range, nb,
        BB=BB, BM=BM)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = args.device
    B = args.batch

    rng = np.random.default_rng(0)
    from pnn_sim.world import make_house
    env = Fp16Env(B, RoverConfig(), seed=42, device=dev)
    env._worlds = [make_house(rng) for _ in range(B)]
    env._build_segments()
    # randomize poses a bit (rover mid-game state)
    env.x.uniform_(0.5, 10.5); env.y.uniform_(0.5, 10.5)
    env.theta.uniform_(-np.pi, np.pi)

    # ---- P) geometry parity: eager fp16 scan with noise/dropout zeroed --
    env.noise = lambda std, *shape: torch.zeros(
        shape if shape else (B,), device=env.x.device)
    env._dropout = lambda r, p: r.clamp(min=0.02)
    ref = env.scan()                                   # [B, nb] clean-ish
    got = fused_scan(env)
    ref_c = torch.where(torch.isfinite(ref), ref, torch.tensor(12.0,
                        device=dev)).clamp(min=0.02)
    d = (got - ref_c).abs()
    ok995 = float((d <= 0.05).float().mean())
    worst = float(d.max())
    print(f"PARITY: beams within 0.05 m = {ok995*100:.3f}%  "
          f"worst={worst:.3f} m  (PASS >= 99.5%)")
    torch.manual_seed(1)
    n1 = fused_scan(env)                                # determinism
    n2 = fused_scan(env)
    det = bool(torch.equal(n1, n2))
    print(f"DETERMINISM: identical across calls = {det}")

    # ---- B) speed: fused vs fp16 eager scan, same env, eager ----------
    # rebuild env with live noise for the timed eager path (fair: BOTH do
    # noise+dropout; the fused side adds them torch-side after the kernel)
    env2 = Fp16Env(B, RoverConfig(), seed=42, device=dev)
    env2._worlds = [make_house(np.random.default_rng(0)) for _ in range(B)]
    env2._build_segments()
    env2.x.uniform_(0.5, 10.5); env2.y.uniform_(0.5, 10.5)
    env2.theta.uniform_(-np.pi, np.pi)
    M = env2._a.shape[1]
    out_buf = torch.empty(B, env2.cfg.n_beams, device=dev, dtype=torch.float32)

    def eager_scan():
        return env2.scan()

    def fused_full():
        r = fused_scan(env2, out=out_buf)
        r = r + env2.noise(env2.cfg.lidar_noise_std, B, env2.cfg.n_beams)
        return env2._dropout(r, env2.cfg.lidar_dropout_p)

    for fn, name in ((eager_scan, "fp16 eager"), (fused_full, "fused triton")):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        print(f"{name:12s} scan: {dt*1000:7.2f} ms/tick   "
              f"({B*env2.cfg.n_beams/dt/1e9:5.1f} Mrays/s)")
    print(f"M={M} segs/env, B={B}, NOTE: contention with live run 8 — "
          "ratios portable, absolutes inflated")


if __name__ == "__main__":
    main()
