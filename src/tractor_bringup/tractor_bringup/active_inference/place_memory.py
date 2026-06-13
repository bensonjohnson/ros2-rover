"""Topological place memory — room fingerprints, no coordinates at all.

"Have I been in this room lately?" is a question about what the room LOOKS
like, not where it is. Metric spatial memory anchored to skid-steer odometry
drifts within minutes, and every goal-directed behavior built on it inherits
that drift. This module removes the pose from the question entirely.

The fingerprint is the magnitude spectrum of the binned lidar scan: a
rotation of the rover circularly shifts the bins, and |FFT| is invariant to
circular shifts — so the descriptor is IMMUNE to heading drift by
construction. Low frequencies only: they encode the coarse shape/size of the
visible space (what distinguishes rooms) and ignore furniture-level detail
(what varies within one).

Like the visit grid, this is deliberately not a map:
  - RAM only, dies with the process,
  - place weights decay (tau ~ 15 min): it remembers "rooms I've been in
    lately", not the building,
  - a room may legitimately produce 2-3 fingerprints (it looks different
    from its corners) — the semantics are "regions that look the same",
    which is exactly enough for room-to-room exploration.
"""

from __future__ import annotations

import time

import numpy as np


class PlaceMemory:
    def __init__(self, n_freq: int = 10, match_thresh: float = 0.35,
                 tau_s: float = 900.0, max_places: int = 64,
                 time_fn=time.monotonic, shape_weight: float = 1.0,
                 fam_scale_s: float = 20.0, fp_ema_tau_s: float = 0.6):
        # time_fn: clock for the presence-decay. The rover uses wall time;
        # a faster-than-realtime simulator must pass its own sim clock or the
        # 15-min decay runs against the wrong timescale.
        # shape_weight: emphasis on the FFT *shape* harmonics (room geometry)
        # vs channel-0 mean openness (room size). Tempting to raise so
        # similarly-open rooms separate by wall layout — and that works in
        # clean sim — but on the REAL rover's lidar the shape harmonics are
        # noise-dominated: within ONE room the fingerprint jitters (rotation +
        # dropouts) ~5x more than in sim, and scaling shape up scales that
        # noise too, shattering one room into many phantom places (novelty
        # then false-fires while stationary, killing the drive to move).
        # Keep at 1.0: lean on the stable mean-openness channel. Real
        # room-to-room separation needs better-than-FFT-openness features or
        # multi-room real calibration; see [[pnn-sim-harness]].
        # fam_scale_s: novelty of a matched place decays from 1 to 0 over
        # this many seconds of accumulated presence, so entering a new room
        # yields a SUSTAINED (learnable) novelty signal instead of a
        # single-tick spike that is extinguished the moment it is stored.
        # fp_ema_tau_s: temporal smoothing of the fingerprint before matching.
        # Real lidar fingerprints jitter ~5x more than sim per scan (rotation,
        # dropouts), which over-segments one room into phantom places. EMA-ing
        # the FINGERPRINT (not the raw scan — |FFT| is rotation-invariant, so
        # smoothing it doesn't blur angular structure while spinning) over
        # ~tau seconds knocks that per-scan noise down by ~sqrt(N), so a
        # stationary/rotating rover stays one place. 0 disables.
        self.n_freq = int(n_freq)
        self.match_thresh = float(match_thresh)
        self.tau_s = float(tau_s)
        self.max_places = int(max_places)
        self._time_fn = time_fn
        self.shape_weight = float(shape_weight)
        self.fam_scale_s = float(fam_scale_s)
        self.fp_ema_tau_s = float(fp_ema_tau_s)
        self._fp_ema: np.ndarray | None = None
        self._fps: list[np.ndarray] = []     # unit-norm fingerprints
        self._weights: list[float] = []      # seconds of presence, decaying
        self._last = self._time_fn()
        self.novelty = 1.0                   # of the most recent update

    def fingerprint(self, scan) -> np.ndarray:
        """Rotation-invariant room descriptor from the binned scan.

        Channel 0 is mean openness (room SIZE), the rest are FFT magnitudes
        of the mean-removed scan in amplitude units (room SHAPE). The
        vector is deliberately NOT unit-normalized and compared by
        Euclidean distance: normalization makes two featureless rooms of
        different sizes (small square vs big open) collapse onto the same
        direction and read identical.
        """
        s = np.asarray(scan, dtype=np.float64)
        m = float(s.mean())
        harm = np.abs(np.fft.rfft(s - m))[1:self.n_freq] / (s.size / 2.0)
        return np.concatenate([[m], self.shape_weight * harm])

    def update(self, scan) -> float:
        """Fold the current scan in; return place novelty in [0, 1].

        1.0 = nothing remembered looks like this (a new room);
        0.0 = dead match for a recently-occupied place.
        """
        now = self._time_fn()
        dt = max(0.0, now - self._last)
        self._last = now

        if self._weights:
            k = float(np.exp(-dt / self.tau_s))
            self._weights = [w * k for w in self._weights]
            keep = [i for i, w in enumerate(self._weights) if w > 0.05]
            if len(keep) < len(self._weights):
                self._fps = [self._fps[i] for i in keep]
                self._weights = [self._weights[i] for i in keep]

        fp = self.fingerprint(scan)
        # Temporal denoise (rotation-invariant domain) before matching.
        if self.fp_ema_tau_s > 0.0:
            if self._fp_ema is None:
                self._fp_ema = fp.copy()
            else:
                a = min(1.0, dt / self.fp_ema_tau_s)
                self._fp_ema += a * (fp - self._fp_ema)
            fp = self._fp_ema.copy()    # stored downstream; don't alias the EMA
        if not self._fps:
            self._fps.append(fp)
            self._weights.append(max(dt, 0.1))
            self.novelty = 1.0
            return 1.0

        # Euclidean distance to every remembered place.
        d = np.asarray([float(np.linalg.norm(fp - p)) for p in self._fps])
        i = int(np.argmin(d))
        dmin = float(d[i])
        self.novelty = float(np.clip(dmin / self.match_thresh, 0.0, 1.0))

        if dmin < self.match_thresh:
            # Recognized: reinforce, and let the stored fingerprint track
            # slow appearance changes (doors opening, furniture moved).
            self._weights[i] += dt
            self._fps[i] = 0.98 * self._fps[i] + 0.02 * fp
            # Sustained novelty: a place is still "new" until enough presence
            # has accumulated there. A freshly-created room decays from ~1 to
            # 0 over fam_scale_s; a place not visited lately (weight decayed)
            # reads novel again — exactly "rooms I've been in lately".
            fresh = float(np.clip(1.0 - self._weights[i] / self.fam_scale_s,
                                  0.0, 1.0))
            self.novelty = max(self.novelty, fresh)
        else:
            self._fps.append(fp)
            self._weights.append(max(dt, 0.1))
            if len(self._fps) > self.max_places:
                j = int(np.argmin(self._weights))
                self._fps.pop(j)
                self._weights.pop(j)
        return self.novelty

    def n_places(self) -> int:
        return len(self._fps)

    def clear(self) -> None:
        """Forget everything (rover picked up / moved to a new building)."""
        self._fps.clear()
        self._weights.clear()
        self._fp_ema = None
        self.novelty = 1.0
