#!/usr/bin/env python3
"""Visual place fingerprint — exposure- and heading-tolerant, numpy-only.

The lidar fingerprint (FFT of the openness vector) gives pose-free place
identity, but on real LD19 data room-to-room separation is only ~5x the
noise floor. The camera is the second channel meant to widen that margin —
the ArduCam 1080P is physically on the rover but nothing in the active-
inference stack consumed images until this module.

Requirements the place-memory failure modes impose on the descriptor:
  * Exposure invariant: the low-light ArduCam hunts auto-exposure; raw
    brightness would drift the fingerprint on a still rover. Fix: histogram
    equalization before any statistics.
  * Heading/pan tolerant: the rover pivots in place (documented behavior), and
    a carried tour sweeps heading. A pivot TURNS the scene across the sensor —
    a translation, not a rotation, of the image. Fix: the same trick that
    makes the lidar fingerprint heading-invariant: |FFT| magnitude spectra,
    which are blind to circular shift. Every statistic here is GLOBAL (no
    image-position binning).
  * Cheap and dependency-free: computed once per lidar tick (10 Hz) at
    48x64 resolution; no torch, no cv2 (YUYV decoded by hand).

Layout (FP_DIM=23): 48x64 equalized, 3x3-smoothed luminance ->
  (a) |FFT| harmonics of the luminance column profile (pan-invariant layout),
  (b) canonicalized gradient-orientation histogram (rolled to its circular
      centroid; double-angle for the pi period) — the room "style" channel,
  (c) 5 global texture/brightness scalars. L2-normalized. A query-side EMA
  (same chase arithmetic as the lidar fp) smooths frame-to-frame flicker.
"""

from __future__ import annotations

import numpy as np

H, W = 48, 64
N_ORIENT = 8
FP_DIM = 10 + N_ORIENT + 5   # 23: FFT layout + canonicalized orientation + stats


def _decode_luma(img) -> np.ndarray:
    """sensor_msgs/Image -> HxW float32 luminance in [0,1].

    Handles the v4l2_camera passthrough encodings (yuv422_yuy2 / yuyv —
    raw YUYV means the CAMERA NODE does zero per-frame conversion, which is
    the difference between 30 fps and 6 fps on this hardware) plus rgb8/bgr8
    and mono8 fallbacks. Downsample by striding, not interpolation.
    """
    enc = img.encoding
    h, w = img.height, img.width
    buf = np.frombuffer(img.data, dtype=np.uint8)
    if enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        # YUYV packs [Y0 U Y1 V] per PIXEL PAIR: 4 bytes / 2 px = 2 bytes/px,
        # so the buffer is h * 2w bytes. Taking [:,:,0] of a (h, w//2, 2)
        # reshape would interleave U/V into the luma — reshape by QUADS.
        quads = buf[:h * 2 * w].reshape(h, w // 2, 4)
        y = np.stack([quads[:, :, 0], quads[:, :, 2]],
                     axis=-1).reshape(h, w)            # full-res luma
    elif enc in ("rgb8",):
        rgb = buf[:h * w * 3].reshape(h, w, 3)
        y = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    elif enc in ("bgr8",):
        rgb = buf[:h * w * 3].reshape(h, w, 3)
        y = (0.114 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.299 * rgb[:, :, 2])
    elif enc in ("mono8",):
        y = buf[:h * w].reshape(h, w)
    else:
        raise ValueError(f"unsupported image encoding: {enc!r}")
    # Downsample to HxW by averaging exact blocks (crop any remainder rows/
    # cols first so the reshape is always exact — no strided aliasing).
    sh, sw = h // H, w // W
    y = y[:H * sh, :W * sw]
    g = y.astype(np.float32).reshape(H, sh, W, sw).mean(axis=(1, 3))
    return g / 255.0


def equalize(g: np.ndarray) -> np.ndarray:
    """256-bin histogram equalization on a [0,1] float grid."""
    q = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(q.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    total = cdf[-1]
    if total <= 0:
        return g
    lo = cdf[hist > 0][0] if np.any(hist > 0) else 0.0
    lut = ((cdf - lo) / max(total - lo, 1.0)).astype(np.float32)
    return lut[q]


def _smooth3(g: np.ndarray) -> np.ndarray:
    """3x3 binomial blur (no cv2). Real cameras run this in the ISP; we work
    on raw YUYV luma, and without it the gradient statistics of low-structure
    scenes are sensor-noise dominated (synthetic gate: same-view jitter 0.66).
    Separable [1 2 1] with replicate edges, then /16."""
    k = np.array([1.0, 2.0, 1.0])
    # horizontal
    gp = np.pad(g, ((0, 0), (1, 1)), mode="edge")
    out = (gp[:, :-2] * k[0] + gp[:, 1:-1] * k[1] + gp[:, 2:] * k[2])
    # vertical
    gp = np.pad(out, ((1, 1), (0, 0)), mode="edge")
    out = (gp[:-2, :] * k[0] + gp[1:-1, :] * k[1] + gp[2:, :] * k[2])
    return out / 16.0


def _spectrum_cols(profile: np.ndarray, k: int) -> np.ndarray:
    """First `k` harmonics of |rfft| of a column-profile. The lidar place
    fingerprint is rotation-invariant because |FFT| ignores circular shift;
    the same theorem covers CAMERA PAN: turning in place slides the scene
    across the sensor = translates the column profile, so |FFT| of it is the
    pan-robust encoding. Hann-windowed to damp edge wrap-around."""
    n = profile.size
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))
    spec = np.abs(np.fft.rfft((profile - profile.mean()) * win))[1:k + 1]
    return spec / (n / 4.0)


def visual_fingerprint(img) -> np.ndarray:
    """sensor_msgs/Image -> unit-norm place fingerprint.

    Two descriptors, each robust by CONSTRUCTION (the failure of v1/v2 was
    tuning robustness into features that are position-dependent at heart):

    A) Column-profile |FFT| (10 dims): luminance column means (equalized),
       minus mean, first 10 harmonics of the magnitude spectrum. A pivot in
       place translates the scene across the sensor == circular shift of the
       column profile == |FFT| invariant. This is the EXACT mechanism the
       lidar PlaceMemory uses for heading invariance, transposed to image
       columns. Brightness enters via the (equalized) profile mean and
       gradient stats, not via any positional moment.

    B) Orientation histogram (8 dims) + 4 global gradient stats: canonical
       (rolled to circular centroid) orientation distribution and edge/
       texture strength of the 3x3-smoothed equalized image — the room
       "style" channels (blinds vs clutter vs bare wall) the FFT cannot see.

    All statistics are GLOBAL. v1's radial image-position bins and v2's
    brightness centroid both failed the pan gate (moved more under a 48px
    roll than a room change) and are gone; see pnn_sim/tools/vis_fp_check.py
    — which is a smoke guard, not acceptance. Acceptance is bag_vis_replay
    on house data, like the lidar temperament was calibrated.
    """
    g = equalize(_decode_luma(img))          # exposure-invariant grid
    gs = _smooth3(g)                         # sensor-noise floor removal

    # A) pan-invariant scene layout: |FFT| of the column profile
    col = gs.mean(axis=0)                    # W-dim vertical-edge layout
    spec = _spectrum_cols(col, 10)

    # B) texture/style from gradients
    gx = (gs[:, 1:] - gs[:, :-1])[:-1, :]
    gy = (gs[1:, :] - gs[:-1, :])[:, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) % np.pi
    hist = np.zeros(N_ORIENT, dtype=np.float64)
    np.add.at(hist, np.minimum((ang / np.pi * N_ORIENT).astype(np.int64),
                               N_ORIENT - 1), np.log1p(mag))
    tot = hist.sum()
    if tot <= 0:
        hist[:] = 1.0; tot = float(N_ORIENT)
    h = hist / tot
    ks = np.array([0.25, 0.5, 0.25])
    hs = np.convolve(np.r_[h[-1:], h, h[:1]], ks, mode="valid")
    phi = (np.arange(N_ORIENT) + 0.5) / N_ORIENT * np.pi
    c_sin = float((hs * np.sin(2 * phi)).sum())
    c_cos = float((hs * np.cos(2 * phi)).sum())
    cc = np.arctan2(c_sin, c_cos) % np.pi
    h = np.roll(h, -int(round(cc / np.pi * N_ORIENT)) % N_ORIENT)
    # circular variance of the (normalized) orientation histogram: 1 - |mean
    # resultant|. hs sums to ~1 (it is a smoothed PDF), so NO division by the
    # raw log-magnitude total `tot` (v1 bug: scale mixed PDF and weight units).
    circ_var = 1.0 - abs(complex(c_cos, c_sin))
    edge_frac = float((mag > 0.05).mean())
    m_mean = float(np.log1p(mag.mean()))
    m_std = float(mag.std())

    fp = np.concatenate([
        spec,                                     # 10 pan-invariant layout
        2.0 * h,                                  #  8 canonicalized orientation
        np.array([edge_frac, m_mean, m_std,       #  3 texture strength
                  circ_var,                       #  1 orientation concentration
                  float(col.mean())]),            #  1 equalized brightness level
    ])
    nrm = np.linalg.norm(fp)
    return (fp / nrm).astype(np.float32) if nrm > 0 else fp.astype(np.float32)


class VisFingerprintEMA:
    """Query-side EMA of the visual fingerprint (tau in seconds, called at
    the lidar tick rate with the freshest frame). Same lesson as the lidar
    side: smoothing belongs on the QUERY; stored references stay frozen
    (slot_blend=0) or you re-create the chase equilibrium that collapses all
    places into one."""

    def __init__(self, tau_s: float = 1.0):
        self.tau_s = float(tau_s)
        self._ema: np.ndarray | None = None
        self._last: float | None = None

    def update(self, fp: np.ndarray, now: float) -> np.ndarray:
        fp = np.asarray(fp, dtype=np.float32)
        if self._ema is None or self.tau_s <= 0.0:
            self._ema = fp.copy()
        else:
            dt = max(0.0, now - (self._last if self._last is not None else now))
            a = min(1.0, dt / self.tau_s)
            self._ema += a * (fp - self._ema)
        self._last = now
        nrm = np.linalg.norm(self._ema)
        return (self._ema / nrm).astype(np.float32) if nrm > 0 else self._ema.copy()
