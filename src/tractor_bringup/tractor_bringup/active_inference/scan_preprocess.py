"""LaserScan -> fixed-length normalized observation vector.

The world model needs a constant-width, finite, normalized input regardless of
the lidar's native sample count, dropouts, or inf/NaN returns. We bin the full
360 deg into `num_bins` angular sectors and take the *nearest* return in each
sector (min-pooling), because for obstacle avoidance the closest thing in a
direction is what matters.

Output convention (per bin), in [0, 1]:
    1.0 -> nothing closer than max_range (open space)
    0.0 -> obstacle right at the sensor
So the vector is a coarse "openness" map around the robot. Bin 0 is the sensor's
angle_min and bins increase with angle.
"""

from __future__ import annotations

import numpy as np


def preprocess_scan(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    num_bins: int = 72,
    max_range: float = 5.0,
    min_range: float = 0.05,
) -> np.ndarray:
    """Reduce a raw LaserScan ranges array to `num_bins` floats in [0, 1].

    inf/NaN/zero/out-of-band returns are treated as "no obstacle" (max_range).
    Empty sectors (no beams land in them) also read as open space.
    """
    ranges = np.asarray(ranges, dtype=np.float32)
    n = ranges.shape[0]

    # Sanitize: invalid or too-far -> max_range (open); too-close clamps to min.
    clean = ranges.copy()
    invalid = ~np.isfinite(clean) | (clean <= 0.0) | (clean < min_range)
    clean[invalid] = max_range
    np.clip(clean, min_range, max_range, out=clean)

    # Angle of each beam, wrapped to [0, 2pi), mapped to a bin index.
    idx = np.arange(n, dtype=np.float32)
    angles = angle_min + idx * angle_increment
    two_pi = 2.0 * np.pi
    frac = np.mod(angles, two_pi) / two_pi          # [0, 1)
    bins = np.minimum((frac * num_bins).astype(np.int64), num_bins - 1)

    # Min-pool: nearest return per sector. Start at max_range (open).
    out = np.full(num_bins, max_range, dtype=np.float32)
    np.minimum.at(out, bins, clean)

    # Normalize so 1.0 == open, 0.0 == at-sensor.
    out /= max_range
    return out
