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
                 fam_scale_s: float = 20.0, fp_ema_tau_s: float = 0.6,
                 slot_blend: float = 0.02, create_drift_gate: float = 0.0,
                 vis_weight: float = 0.0):
        # vis_weight: weight of the CAMERA channel in the place distance
        # (d = lidar_d + vis_weight * vis_d, vis fp unit-norm so vis_d<=2).
        # 0.0 (default) = lidar-only, byte-identical to the validated field
        # temperament. The real-room lidar separation is only ~5x noise, so
        # the visual fingerprint (vis_fingerprint.py) is the second channel
        # meant to widen it — but its weight MUST be calibrated offline on
        # house bags (pnn_sim/tools/bag_vis_replay.py) before nonzero: a
        # vis_d jitter above the lidar room-room gap (0.15-0.25) would
        # phantom-fire places, the exact chase/merge failure already solved
        # on the lidar side. See MEMORY of this project: validate OFFLINE.
        # create_drift_gate: only CREATE a new place when the (post-EMA)
        # fingerprint is settled — per-tick drift below this value. Walking
        # carries the view through doorway/corridor transients; creating a
        # place mid-crossing stores a doorway reference that then ATTRACTS
        # the rooms on both sides (they arrive matching the smeared transient)
        # and the tour collapses to 1-2 places (house-tour bag analysis,
        # pnn_sim/tools/tour_analyze.py). Settled-only creation makes every
        # place a dwell center. 0 = off.
        # slot_blend: consolidation rate of a matched slot toward the live
        # fingerprint (the historical 0.98/0.02 tracking). At 0.02/tick the
        # stored reference CHASES a slowly drifting query — equilibrium
        # dmin = fp drift rate / blend, comfortably under match_thresh —
        # so a continuously walking rover matches ONE place forever and
        # new places only appear on teleport-size jumps (the places-stuck-
        # at-1 collapse; see pnn_sim/tools/t4_place_probe.py + rules).
        # Per-scan jitter is already handled query-side by fp_ema_tau_s;
        # 0.0 = frozen references. Validated in batched place.py replay.
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
        self.slot_blend = float(slot_blend)
        self.create_drift_gate = float(create_drift_gate)
        self.vis_weight = float(vis_weight)
        self._prev_fp: np.ndarray | None = None
        self._fp_ema: np.ndarray | None = None
        self._vis_fps: list[np.ndarray | None] = []  # per-slot visual refs
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

    def update(self, scan, vis_fp: np.ndarray | None = None) -> float:
        """Fold the current scan in; return place novelty in [0, 1].

        1.0 = nothing remembered looks like this (a new room);
        0.0 = dead match for a recently-occupied place.

        vis_fp: query-side EMA'd visual fingerprint of the freshest camera
        frame (see vis_fingerprint.VisFingerprintEMA), or None when the
        camera is off/unavailable. Combined distance is
        lidar_d + vis_weight * vis_d; with vis_weight 0.0 (or vis_fp None)
        this is exactly the lidar-only rule.
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
                self._vis_fps = [self._vis_fps[i] for i in keep]
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
        # drift since last tick (settledness test for create gate)
        drift = 0.0 if self._prev_fp is None else float(
            np.linalg.norm(fp - self._prev_fp))
        self._prev_fp = fp.copy()
        vf: np.ndarray | None = (None if vis_fp is None
                                 else np.asarray(vis_fp, dtype=np.float32))
        use_vis = self.vis_weight > 0.0 and vf is not None
        if not self._fps:
            self._fps.append(fp)
            self._vis_fps.append(vf.copy() if use_vis and vf is not None else None)
            self._weights.append(max(dt, 0.1))
            self.novelty = 1.0
            return 1.0

        # Euclidean distance to every remembered place (lidar channel;
        # visual channel added per-slot when both sides have one).
        vw = self.vis_weight if (use_vis and vf is not None) else 0.0
        d = np.asarray([
            float(np.linalg.norm(fp - p))
            + (vw * float(np.linalg.norm(vf - v))
               if vf is not None and v is not None else 0.0)
            for p, v in zip(self._fps, self._vis_fps)])
        i = int(np.argmin(d))
        dmin = float(d[i])
        self.novelty = float(np.clip(dmin / self.match_thresh, 0.0, 1.0))

        if dmin < self.match_thresh:
            # Recognized: reinforce; optionally let the stored fingerprint
            # track slow appearance changes (doors opening) — see slot_blend
            # doc in __init__ for why 0.0 is the validated default.
            self._weights[i] += dt
            if self.slot_blend > 0.0:
                self._fps[i] = ((1 - self.slot_blend) * self._fps[i]
                                + self.slot_blend * fp)
                if use_vis and self._vis_fps[i] is not None and vf is not None:
                    self._vis_fps[i] = ((1 - self.slot_blend) * self._vis_fps[i]
                                        + self.slot_blend * vf)
            # Sustained novelty: a place is still "new" until enough presence
            # has accumulated there. A freshly-created room decays from ~1 to
            # 0 over fam_scale_s; a place not visited lately (weight decayed)
            # reads novel again — exactly "rooms I've been in lately".
            fresh = float(np.clip(1.0 - self._weights[i] / self.fam_scale_s,
                                  0.0, 1.0))
            self.novelty = max(self.novelty, fresh)
        elif (self.create_drift_gate > 0.0
              and drift > self.create_drift_gate):
            # Mid-walk transient: novel but unsettled. High novelty (keeps
            # the rover moving) but NO stored reference — see ctor doc.
            pass
        else:
            self._fps.append(fp)
            self._vis_fps.append(vf.copy() if use_vis and vf is not None else None)
            self._weights.append(max(dt, 0.1))
            if len(self._fps) > self.max_places:
                j = int(np.argmin(self._weights))
                self._fps.pop(j)
                self._vis_fps.pop(j)
                self._weights.pop(j)
        return self.novelty

    def n_places(self) -> int:
        return len(self._fps)

    def clear(self) -> None:
        """Forget everything (rover picked up / moved to a new building)."""
        self._fps.clear()
        self._vis_fps.clear()
        self._weights.clear()
        self._fp_ema = None
        self._prev_fp = None
        self.novelty = 1.0
