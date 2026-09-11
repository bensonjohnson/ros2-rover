"""LaserScan -> (ranges, bearings) for the PC spatial map. No ROS imports.

The map's soft-raycast sensor model consumes metric ranges plus an explicit
per-beam bearing, so a beam can simply be DROPPED — which is what makes honest
handling of real lidar returns possible:

  - no return (inf, or >= range_max): the beam genuinely saw nothing out to
    max range, so it is free the whole way. Clamp it to the map's max range;
    the map reads "no hit" and marks free space. This is what the sim's
    dropout beams already do.
  - error (NaN, or below range_min): the sensor failed, which is NOT evidence
    of free space. Drop the beam. Folding these into max range would carve
    phantom corridors through walls — the failure mode that matters, because
    the policy steers toward free-space-adjacent unknown.

Kept ROS-free so it can be unit-tested off the rover, where rclpy is absent.
"""

from __future__ import annotations

import numpy as np


def scan_to_beams(ranges, angle_min, angle_increment, range_min, range_max,
                  map_max_range, min_beams=8):
    """Return (ranges[K] float32, bearings[K] float32) or (None, None).

    `ranges` is the raw LaserScan.ranges sequence. Returns None when too few
    beams survive to say anything about the world."""
    r = np.asarray(ranges, dtype=np.float32)
    n = r.shape[0]
    if n == 0:
        return None, None
    bearings = angle_min + np.arange(n, dtype=np.float32) * angle_increment

    rmin = max(float(range_min), 1e-3)
    finite = np.isfinite(r)
    no_return = np.isinf(r) | (finite & (r >= float(range_max)))
    valid_hit = finite & (r >= rmin) & (r < float(range_max))
    keep = valid_hit | no_return
    if int(keep.sum()) < min_beams:
        return None, None

    out = np.where(no_return, np.float32(map_max_range), r)[keep]
    out = np.clip(out, rmin, map_max_range).astype(np.float32)
    return np.ascontiguousarray(out), np.ascontiguousarray(bearings[keep])
