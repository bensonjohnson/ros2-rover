#!/usr/bin/env python3
"""Smoke/regression guard for the visual place fingerprint (vis_fingerprint).

This is NOT the acceptance test — the fingerprint's real calibration is the
offline bag sweep (bag_vis_replay.py) on house data, exactly like the lidar
temperament was set on real LD19 bags, not synthetic rooms. This only proves
the code decodes, is deterministic, and does not regress into the two known
bad designs (v1 radial-bin and v2 brightness-centroid both let a horizontal
pan move the descriptor more than a room change — see module docstring).

  python3 pnn_sim/tools/vis_fp_check.py
"""
import sys
import numpy as np
sys.path.insert(0, '/home/benson/projects/ros2-rover/src/tractor_bringup')
from tractor_bringup.active_inference import vis_fingerprint as V
from tractor_bringup.active_inference.vis_fingerprint import (
    visual_fingerprint, VisFingerprintEMA, _decode_luma)


class FakeImg:
    """Minimal sensor_msgs/Image stand-in (numpy + no rclpy needed here)."""
    def __init__(self, arr, enc, width=None):
        self.encoding = enc
        self.height = arr.shape[0]
        self.width = width if width is not None else arr.shape[1]
        self.data = arr.tobytes()


def synth_room(seed, style, h=240, w=320):
    r = np.random.default_rng(seed)
    g = np.zeros((h, w), np.uint8)
    if style == "hallway":
        g[:] = 40; g[:, w//3:2*w//3] = 200
    elif style == "corner":
        g[:h//2, :] = 180; g[h//2:, w//2:] = 220; g[h//2:, :w//2] = 60
    elif style == "doorway":
        g[:] = 90; g[h//4:3*h//4, w//2-w//8:w//2+w//8] = 250
    elif style == "window":
        g[:] = 70; g[::14, :] = 160            # horizontal blinds
    elif style == "clutter":
        for _ in range(40):
            x, y = r.integers(0, w-30), r.integers(0, h-30)
            g[y:y+30, x:x+30] = int(r.integers(0, 255))
    noise = r.integers(0, 8, (h, w), dtype=np.uint8)
    return (g*0.9 + noise).clip(0, 255).astype(np.uint8)


def yuyv_from_gray(g):
    """Pack a gray image as YUYV: [Y0 U Y1 V] per pixel PAIR (4B/2px)."""
    h, w = g.shape
    y = g.astype(np.uint8)
    u = np.full((h, w//2), 128, np.uint8); v = u.copy()
    y2 = y.reshape(h, w//2, 2)
    out = np.zeros((h, w//2, 4), np.uint8)
    out[:, :, 0] = y2[:, :, 0]; out[:, :, 1] = u
    out[:, :, 2] = y2[:, :, 1]; out[:, :, 3] = v
    return out.reshape(h, 2*w)


def FP(g):
    return visual_fingerprint(
        FakeImg(yuyv_from_gray(g), "yuv422_yuy2", width=g.shape[1]))


def main():
    fails = []
    H, Wd = V.H, V.W

    # 1) YUYV decode: luma matches the same crop+block-average the decoder does
    g = synth_room(1, "hallway")
    sh, sw = g.shape[0]//H, g.shape[1]//Wd
    gt = g[:H*sh, :Wd*sw].astype(np.float32).reshape(H, sh, Wd, sw).mean(axis=(1, 3))
    err = float(np.abs(_decode_luma(
        FakeImg(yuyv_from_gray(g), "yuv422_yuy2", width=g.shape[1]))*255 - gt).max())
    print(f"1. YUYV luma decode max-err {err:.2f} (want <2)")
    if err >= 2: fails.append("decode")

    # 2) fp shape + unit norm
    fp = FP(g)
    print(f"2. fp dim {fp.shape} == ({V.FP_DIM},), norm {float(np.linalg.norm(fp)):.3f}")
    if fp.shape != (V.FP_DIM,): fails.append("dim")

    # 3) determinism: identical input -> identical fp
    same = float(np.linalg.norm(FP(g) - FP(g)))
    print(f"3. determinism {same:.2e} (want 0)")
    if same > 0: fails.append("determinism")

    # 4) PAN robustness (the killer from v1/v2): horizontal image shift from a
    #    pivot-in-place must stay well under the room-room gap.
    f0 = FP(g)
    pan = [float(np.linalg.norm(f0 - FP(np.roll(g, k, axis=1)))) for k in (8, 24, 48)]
    print(f"4. pan-shift dists {['%.3f' % d for d in pan]} (want all < 0.15)")
    if max(pan) >= 0.15: fails.append("pan")

    # 5) equalization invariance. Histogram equalization is, by construction,
    #    invariant to any monotone brightness map that preserves the 256-level
    #    ranking. Assert THAT (near 0). Gamma/clipping done in 8-bit integer
    #    space collides quantization bins = information loss, the same class as
    #    sensor saturation; those are reported, not gated. (Real AE holds SNR
    #    via analog gain, so it is the injective case; watch the residual on
    #    real bags where a genuine AE-driven shift would show up.)
    gA = synth_room(2, "corner")
    fA = FP(gA)
    # an injective monotone map on the 8-bit levels actually present:
    # scale [0,255] -> [0,1] -> x/(x+0.15) -> back, and verify distinct levels
    # survive as distinct uint8 (else it is a lossy test, not an invariance one)
    vals = np.unique(gA)
    m = vals.astype(np.float32)/255.0
    m = (m/(m+0.15))
    m8 = np.round(m*255.0).astype(np.uint8)
    injective = len(np.unique(m8)) == len(vals)
    lut = np.zeros(256, np.uint8); lut[vals] = m8
    d_mono = float(np.linalg.norm(fA - FP(lut[gA])))
    gam = lambda g, k: (255.0*(np.clip(g,0,255)/255.0)**k).astype(np.uint8)
    d_clip = float(np.linalg.norm(fA - FP((gA.astype(np.float32)*2.5).clip(0,255).astype(np.uint8))))
    print(f"5. monotone-brightness (injective={injective}) d {d_mono:.4f} "
          f"(want <0.02 — equalizer invariance); lossy x2.5-clip {d_clip:.3f} (info destroyed, reported)")
    if injective and d_mono >= 0.02: fails.append("equalizer-invariance")

    # 6) discrimination: visually different scenes must separate beyond noise
    names = ("hallway", "corner", "window", "clutter")
    fps = {n: FP(synth_room(7, n)) for n in names}
    sep = 9.0
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = float(np.linalg.norm(fps[names[i]] - fps[names[j]]))
            sep = min(sep, d)
    # same-view noise floor (AE + sensor noise)
    g0 = synth_room(9, "doorway"); base = FP(g0)
    jit = []
    for s in range(8):
        r = np.random.default_rng(100+s)
        gs = (g0.astype(np.int16)+r.integers(-10, 11, g0.shape)).clip(0, 255).astype(np.uint8)
        gs = (gs.astype(np.float32)*float(r.uniform(0.7, 1.4))).clip(0, 255).astype(np.uint8)
        jit.append(float(np.linalg.norm(base - FP(gs))))
    ratio = sep/max(max(jit), 1e-6)
    print(f"6. min room-room {sep:.3f}, same-view jitter max {max(jit):.3f} "
          f"-> separation {ratio:.1f}x  (INFORMATIONAL — feature selection on "
          f"synthetic rooms is unreliable; the lidar temperament needed real "
          f"bags to move tau 0.6->6.0, acceptance for this is bag_vis_replay)")

    # 7) EMA tames flicker on a fixed view
    FO = FP(g0); e = VisFingerprintEMA(tau_s=0.5); raw = []; sm = []; t = 0.0
    for s in range(20):
        r = np.random.default_rng(200+s)
        gs = (g0.astype(np.int16)+r.integers(-14, 15, g0.shape)).clip(0, 255).astype(np.uint8)
        gs = (gs.astype(np.float32)*float(r.uniform(0.6, 1.6))).clip(0, 255).astype(np.uint8)
        t += 0.1; raw.append(float(np.linalg.norm(FP(gs)-FO)))
        sm.append(float(np.linalg.norm(e.update(FP(gs), t)-FO)))
    print(f"7. EMA flicker {np.mean(raw[5:]):.3f} -> {np.mean(sm[5:]):.3f} (want drop)")
    if not np.mean(sm[5:]) < np.mean(raw[5:]): fails.append("ema")

    print("\nRESULT:", "FAIL " + ",".join(fails) if fails else "PASS (smoke only — real acceptance is bag_vis_replay on house data)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
