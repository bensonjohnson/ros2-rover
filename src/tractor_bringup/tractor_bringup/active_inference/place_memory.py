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
    def __init__(self, n_freq: int = 10, match_thresh: float = 0.25,
                 tau_s: float = 900.0, max_places: int = 64):
        self.n_freq = int(n_freq)
        self.match_thresh = float(match_thresh)
        self.tau_s = float(tau_s)
        self.max_places = int(max_places)
        self._fps: list[np.ndarray] = []     # unit-norm fingerprints
        self._weights: list[float] = []      # seconds of presence, decaying
        self._last = time.monotonic()
        self.novelty = 1.0                   # of the most recent update

    def fingerprint(self, scan) -> np.ndarray:
        """Rotation-invariant room descriptor from the binned scan.

        The raw DC term (mean openness x num_bins) dwarfs the shape
        harmonics and makes every room look alike after normalization, so
        it is removed before the FFT and re-added as a single half-weight
        "room size" channel; harmonics are scaled to amplitude units so
        size and shape carry comparable votes.
        """
        s = np.asarray(scan, dtype=np.float64)
        m = float(s.mean())
        harm = np.abs(np.fft.rfft(s - m))[1:self.n_freq] / (s.size / 2.0)
        v = np.concatenate([[0.5 * m], harm])
        return v / (np.linalg.norm(v) + 1e-9)

    def update(self, scan) -> float:
        """Fold the current scan in; return place novelty in [0, 1].

        1.0 = nothing remembered looks like this (a new room);
        0.0 = dead match for a recently-occupied place.
        """
        now = time.monotonic()
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
        if not self._fps:
            self._fps.append(fp)
            self._weights.append(max(dt, 0.1))
            self.novelty = 1.0
            return 1.0

        # Cosine distance to every remembered place (fingerprints are unit).
        d = 1.0 - np.asarray([float(fp @ p) for p in self._fps])
        i = int(np.argmin(d))
        dmin = float(d[i])
        self.novelty = float(np.clip(dmin / self.match_thresh, 0.0, 1.0))

        if dmin < self.match_thresh:
            # Recognized: reinforce, and let the stored fingerprint track
            # slow appearance changes (doors opening, furniture moved).
            self._weights[i] += dt
            blended = 0.98 * self._fps[i] + 0.02 * fp
            self._fps[i] = blended / (np.linalg.norm(blended) + 1e-9)
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
        self.novelty = 1.0
