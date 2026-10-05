"""Port of lidar_safety_monitor's track-command path, ROS-free.

The brain treats the gate's clamp as an interoceptive observation (the
"hold" channel) and learns collision avoidance FROM its firings — so the sim
must reproduce the gate's behavior, not just collision physics: the arc-
shifted front corridor, phantom suppression (min points across consecutive
scans), hysteresis, the minimum hold time, rear blocking, and side-turn
equalization. Defaults mirror pc_active_inference.launch.py.

Sim time is injected (time_fn) so hold durations follow the accelerated
clock, not wall time.

Named presets (stage 4a) — GATE_PRESETS below pins the two configurations
the platform must be able to name explicitly:

  * ``sim_evo`` — what the sim + the ``evo_hw`` deploy launch use today
    (stop 0.15 / rear 0.30 m; the rest are GateConfig defaults). It is
    field-by-field equal to ``GateConfig()``.
  * ``monitor_default`` — the ROS ``lidar_safety_monitor`` node's own
    parameter defaults (stop 0.20 / rear 0.40 m, slow 0.40, hysteresis
    0.05, max_eval 4.0). Same source as sim_evo, different threshold set.

CONFLICT / discipline: sim-side and deploy-side gates must stay PAIRED —
the brain learns avoidance against whatever gate it trains under, so
switching the sim to ``monitor_default`` while the rover keeps the
deploy ``sim_evo`` overrides (or vice versa) would train against a
different safety envelope than the hardware enforces. Pick ONE preset and
use it on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sector boundaries (radians) — same constants as the rover monitor.
_SIDE_MIN = 0.52
_SIDE_MAX = 1.57
_REAR_MIN = 2.62


# Named gate presets (additive; gate_config() in fast.py is unchanged).
# Values read from the sources cited per entry; only non-default fields
# are listed, so GateConfig(**preset) yields the full config.
GATE_PRESETS = {
    # evo/deploy/evo_hw.launch.py lidar_safety_monitor params (0.15/0.30,
    # slow 0.15, hysteresis 0.10, max_eval 5.0, offset 0.06, half 0.12).
    # Identical to GateConfig() defaults.
    "sim_evo": dict(
        stop_distance=0.15,
        stop_distance_rear=0.30,
        robot_front_offset=0.06,
        robot_half_width=0.12,
    ),
    # src/tractor_bringup/tractor_bringup/lidar_safety_monitor.py
    # declare_parameter defaults: stop 0.20 (L54), rear 0.40 (L55),
    # slow 0.40 (L56), hysteresis 0.05 (L57), max_eval 4.0 (L58).
    "monitor_default": dict(
        stop_distance=0.20,
        stop_distance_rear=0.40,
        slow_distance=0.40,
        hysteresis=0.05,
        max_eval_distance=4.0,
        # offset 0.06 (L70) and half_width 0.12 (L71) match GateConfig
        # defaults; min_block_points/block_scans/min_valid_range/
        # min_block_duration/track_width (0.154) also match. The monitor
        # has no sim-side equivalent of side_stop_distance (it gates sides
        # on stop_dist), so that field keeps the sim default (None).
    ),
}


@dataclass
class GateConfig:
    stop_distance: float = 0.15
    stop_distance_rear: float = 0.30
    slow_distance: float = 0.15        # == stop -> slow zone disabled
    hysteresis: float = 0.10
    min_block_points: int = 3
    block_scans: int = 2
    min_valid_range: float = 0.05
    max_eval_distance: float = 5.0
    robot_front_offset: float = 0.06
    robot_half_width: float = 0.12
    min_block_duration: float = 0.3
    track_width: float = 0.154
    # Side-equalisation distance; None = stop_distance (the real rover's
    # monitor). Sim-only knob for the evo 'doorway' gate prototype.
    side_stop_distance: float | None = None


class SimSafetyGate:
    def __init__(self, cfg: GateConfig, time_fn):
        self.cfg = cfg
        self._now = time_fn
        self.front_blocked = False        # published as /emergency_stop
        self._rear_blocked = False
        self._front_blocked_time = None
        self._front_streak = 0
        self._rear_streak = 0
        self._front_path_dist = cfg.max_eval_distance
        self._sector = {"left": cfg.max_eval_distance,
                        "right": cfg.max_eval_distance,
                        "rear": cfg.max_eval_distance}
        self._latest_cmd = (0.0, 0.0)     # (linear.x, angular.z) synth twist
        self.stops = 0

    # ---- scan-driven state update (the rover does this per lidar rev) ----

    def process_scan(self, ranges: np.ndarray, angle_min: float,
                     angle_increment: float):
        c = self.cfg
        ranges = np.asarray(ranges)
        angles = np.arange(len(ranges)) * angle_increment + angle_min
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        valid = (np.isfinite(ranges)
                 & (ranges > c.min_valid_range)
                 & (ranges <= c.max_eval_distance))
        if not np.any(valid):
            self._front_path_dist = c.max_eval_distance
            for k in self._sector:
                self._sector[k] = c.max_eval_distance
            self._front_streak = self._rear_streak = 0
            self._update_front_blocked()
            return

        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        x_bumper = x - c.robot_front_offset

        # Front corridor, arc-shifted to follow the commanded turn.
        lin, ang = self._latest_cmd
        if lin > 0.02 and abs(ang) > 0.1:
            v = max(lin, 0.05)
            x_fwd = np.maximum(x_bumper, 0.0)
            y_shift = np.clip(ang * x_fwd * x_fwd / (2.0 * v), -0.30, 0.30)
            in_path = (np.abs(y - y_shift) <= c.robot_half_width) & (x_bumper > 0)
        else:
            in_path = (np.abs(y) <= c.robot_half_width) & (x_bumper > 0)
        self._front_path_dist = (max(float(np.min(x_bumper[in_path])), 0.0)
                                 if np.any(in_path) else c.max_eval_distance)

        close_pts = int(np.count_nonzero(in_path & (x_bumper < c.stop_distance)))
        self._front_streak = (self._front_streak + 1
                              if close_pts >= c.min_block_points else 0)
        self._update_front_blocked()

        abs_a = np.abs(a)
        left_m = (a > _SIDE_MIN) & (a < _SIDE_MAX)
        right_m = (a < -_SIDE_MIN) & (a > -_SIDE_MAX)
        rear_m = abs_a >= _REAR_MIN
        md = c.max_eval_distance
        self._sector["left"] = float(np.min(r[left_m])) if np.any(left_m) else md
        self._sector["right"] = float(np.min(r[right_m])) if np.any(right_m) else md
        self._sector["rear"] = float(np.min(r[rear_m])) if np.any(rear_m) else md

        rear_close = int(np.count_nonzero(rear_m & (r < c.stop_distance_rear)))
        self._rear_streak = (self._rear_streak + 1
                             if rear_close >= c.min_block_points else 0)

    def _update_front_blocked(self):
        c = self.cfg
        now = self._now()
        if self.front_blocked:
            if self._front_path_dist > c.stop_distance + c.hysteresis:
                if self._front_blocked_time is None:
                    self.front_blocked = False
                elif now - self._front_blocked_time >= c.min_block_duration:
                    self.front_blocked = False
                    self._front_blocked_time = None
        else:
            if (self._front_path_dist < c.stop_distance
                    and self._front_streak >= c.block_scans):
                self.front_blocked = True
                self._front_blocked_time = now
                self.stops += 1

    # ---- per-command gating (the rover's track_cmd_callback) -------------

    def gate(self, left: float, right: float) -> tuple[float, float]:
        c = self.cfg
        # Synth twist for the next scan's arc corridor — same formula the
        # rover monitor uses on /track_cmd_ai.
        self._latest_cmd = ((left + right) / 2.0 * 0.154,
                            (right - left) / c.track_width)

        # FRONT: blocked -> forward components clamped to <= 0.
        if self.front_blocked:
            left = min(left, 0.0)
            right = min(right, 0.0)
        elif (c.slow_distance > c.stop_distance + 1e-3
              and self._front_path_dist < c.slow_distance
              and max(left, right) > 0.01):
            scale = (self._front_path_dist - c.stop_distance) \
                / (c.slow_distance - c.stop_distance)
            scale = max(0.0, min(1.0, scale))
            if left > 0:
                left *= scale
            if right > 0:
                right *= scale

        # REAR: block backward motion (latched with hysteresis).
        if min(left, right) < -0.01:
            rear = self._sector["rear"]
            if self._rear_blocked:
                if rear > c.stop_distance_rear + c.hysteresis:
                    self._rear_blocked = False
                else:
                    left = max(left, 0.0)
                    right = max(right, 0.0)
            elif (rear < c.stop_distance_rear
                  and self._rear_streak >= c.block_scans):
                self._rear_blocked = True
                left = max(left, 0.0)
                right = max(right, 0.0)

        # SIDES: forward+turn toward a close obstacle -> equalize tracks.
        avg_fwd = (left + right) / 2.0
        is_zero_turn = abs(avg_fwd) < 0.05 and abs(left - right) > 0.1
        if not is_zero_turn and avg_fwd > 0.01:
            turn = right - left
            if turn > 0.1 and self._sector["left"] < c.stop_distance:
                left = right
            elif turn < -0.1 and self._sector["right"] < c.stop_distance:
                right = left

        return left, right
