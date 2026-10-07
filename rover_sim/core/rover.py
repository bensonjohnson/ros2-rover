"""Simulated rover body: tracked diff-drive dynamics + the sensor suite the
brain consumes (2D lidar, wheel velocities, gyro, accelerometer).

The point is not physical perfection but matching the EMBODIMENT the brain
meets on the real rover: the same command envelope (cmd in [-1,1] -> 0.2 m/s
per track), the driver's deadband, per-track trim asymmetry, first-order
motor lag, multiplicative track slip, and sensor noise. The world model
learns action->proprio dynamics, so these imperfections are the curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import World
from .geom import BodyConfig, rect_perimeter_offsets, rect_max_sample_spacing


@dataclass
class RoverConfig:
    # Command envelope — mirrors the real stack (kin_v_max, hiwonder params).
    v_max: float = 0.2            # m/s of one track at cmd = +-1
    track_width: float = 0.154    # m between track centers
    wheel_radius: float = 0.025   # m, for wheel rad/s proprio
    deadband: float = 0.05        # driver zeroes |cmd| below this
    left_trim: float = 0.8        # real rover's left track is trimmed down
    right_trim: float = 1.0
    motor_tau: float = 0.15       # s, first-order lag toward commanded speed
    slip_std: float = 0.05        # multiplicative per-step track slip noise
    robot_radius: float = 0.14    # collision circle (legacy default)

    # Hardware-true geometry (stage 4a) — ALL OPT-IN. body=None keeps the
    # legacy circle; lidar_mount_x=0.0 keeps the raycast origin at the
    # robot center (docs/SIM_PLATFORM.md §4). See rover_sim/core/geom.py.
    body: "BodyConfig | None" = None       # rect footprint when set
    lidar_mount_x: float = 0.0             # LD19 mount, fwd of center (0.07635)
    camera_mount_x: float = 0.0            # D435i mount, fwd of center (0.12085)

    # ---- proprio truth (fidelity item 2) — OPT-IN ----------------------
    # What the wheel channels of the observation actually contain. The legacy
    # sim feeds the lagged track speeds plus encoder noise; the DEPLOY runner
    # cannot (the rover's right encoder reads 0 while the track turns — live
    # runs 5-8), so it substitutes the motor model. That is a silent
    # train/deploy obs skew of exactly the class REVIVAL_PLAN lists as
    # failure #4. Modes:
    #   "true"       legacy: lagged track speed + N(0, 0.05) rad/s, both sides
    #   "model"      what evo/deploy/evo_runner.py --wheel-source model feeds:
    #                the same first-order motor model, NO encoder noise
    #   "dead_right" the raw hardware signature: right channel pinned to 0,
    #                left = lagged track speed + noise
    wheel_source: str = "true"

    # ---- track stall (fidelity item 3) — OPT-IN ------------------------
    # m/s. Static friction break-away for a LOADED track: a commanded track
    # speed below this value cannot move the track at all. This is the
    # mechanism behind the worst sim->real failure in the whole lineage (live
    # runs 5-7): champ20m_idx115 was sim-best (cross 0.833, 61% pivot ticks)
    # and sat PHYSICALLY STILL 56% of its ticks, because pivot commands of
    # +-0.15 -> +-0.024 m/s of track speed never broke static friction. The
    # diff-drive kinematics then produce the observed behaviour for free: both
    # tracks stalled = dead still, one stalled = an orbit about the stalled
    # track (ICC at that track, radius = track_width). 0.0 = disabled.
    track_stall_speed: float = 0.0

    # ---- mechanical ground-speed bias (fidelity item 5) — OPT-IN --------
    # Per-track factor from WHEEL surface speed to GROUND speed: the tracks do
    # not convert equally. The driver's trim exists to CANCEL this: the rover's
    # left_track_trim is 0.8 precisely because the left track is mechanically
    # ~1.25x faster at equal wheel speed (tuned by driving full throttle until
    # it ran straight). The legacy sim folds the WHOLE asymmetry into the trim,
    # so a sim-trained policy learns that cmd (1, 1) curves left — a false
    # dynamic that does not exist on a correctly-trimmed rover. Use
    # track_bias=(1.25, 1.0) with the nominal (0.8, 1.0) trims for the
    # hardware-true pair; then trim randomization models a MIS-TUNED trim
    # (~0.94-1.25 residual) instead of implying a 44%-asymmetric machine.
    # Wheel proprio is unaffected: an encoder measures wheel rotation, not
    # ground speed (this is also why evo_runner's wheel model stays trimmed).
    track_bias: tuple = (1.0, 1.0)

    # Lidar — STL19P-ish: 360 beams, 10-12 Hz on the rover. The real LD19 is
    # 482 beams / 25 m (measured on the rover); n_beams is a plain knob and
    # every path (numpy, fp32, fp16, fused Triton grid) is beam-count generic,
    # but the 360 default stays for byte-comparable legacy runs.
    n_beams: int = 360
    lidar_max_range: float = 12.0
    lidar_noise_std: float = 0.01
    lidar_dropout_p: float = 0.02

    # IMU noise
    gyro_noise_std: float = 0.02      # rad/s
    accel_noise_std: float = 0.15     # m/s^2
    gravity: float = 9.81

    seed: int = 0


class SimRover:
    def __init__(self, world: World, cfg: RoverConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or RoverConfig()
        self.rng = rng or np.random.default_rng(self.cfg.seed)
        # Stage 4a: opt-in rect footprint collision (None = legacy circle).
        self._rect_off = None
        self._rect_margin = 0.0
        if self.cfg.body is not None and self.cfg.body.shape == "rect":
            b = self.cfg.body
            self._rect_off = rect_perimeter_offsets(b.length, b.width,
                                                    b.collision_samples)
            self._rect_margin = 0.5 * rect_max_sample_spacing(
                b.length, b.width, b.collision_samples)
        self.set_world(world)

    def set_world(self, world: World):
        self.world = world
        self.x, self.y, self.theta = world.start_pose
        self.v_left = 0.0             # actual track speeds (m/s)
        self.v_right = 0.0
        self._prev_v = 0.0            # body speed last step, for accel_x
        self.collided = False
        # Latest proprio readings (set by step()).
        self.wheel_l = 0.0            # rad/s
        self.wheel_r = 0.0
        self.yaw_rate = 0.0           # rad/s
        self.accel = np.array([0.0, 0.0, self.cfg.gravity])

    # ------------------------------------------------------------------

    def step(self, cmd_left: float, cmd_right: float, dt: float):
        """Advance one control tick under track commands in [-1, 1]."""
        c = self.cfg

        def target(cmd, trim):
            if abs(cmd) < c.deadband:
                return 0.0
            return float(np.clip(cmd, -1.0, 1.0)) * c.v_max * trim

        tl = target(cmd_left, c.left_trim)
        tr = target(cmd_right, c.right_trim)

        # First-order motor lag toward the commanded speed.
        k = 1.0 - np.exp(-dt / c.motor_tau)
        self.v_left += (tl - self.v_left) * k
        self.v_right += (tr - self.v_right) * k
        if c.track_stall_speed > 0.0:
            # Static friction (fidelity item 3): a loaded track whose commanded
            # speed is below the break-away threshold does not move at all.
            if abs(tl) < c.track_stall_speed:
                self.v_left = 0.0
            if abs(tr) < c.track_stall_speed:
                self.v_right = 0.0

        # Track slip: the ground sees a noisy fraction of the track speed.
        # wheel->ground conversion also carries the mechanical bias (item 5).
        slip_l = 1.0 + self.rng.normal(0.0, c.slip_std)
        slip_r = 1.0 + self.rng.normal(0.0, c.slip_std)
        vl = self.v_left * slip_l * c.track_bias[0]
        vr = self.v_right * slip_r * c.track_bias[1]

        v = 0.5 * (vl + vr)
        w = (vr - vl) / c.track_width

        # Integrate pose; on contact, rotation still works but translation
        # stops (tracks stall against the obstacle).
        nx = self.x + v * np.cos(self.theta) * dt
        ny = self.y + v * np.sin(self.theta) * dt
        if self._rect_off is None:
            self.collided = self.world.clearance(nx, ny) < c.robot_radius
        else:
            # Rect footprint: min over perimeter samples (rotated at the
            # current heading) vs the conservative margin. See geom.py.
            ct, st = np.cos(self.theta), np.sin(self.theta)
            ox = nx + ct * self._rect_off[:, 0] - st * self._rect_off[:, 1]
            oy = ny + st * self._rect_off[:, 0] + ct * self._rect_off[:, 1]
            dmin = min(self.world.clearance(float(ax), float(ay))
                       for ax, ay in zip(ox, oy))
            self.collided = dmin < self._rect_margin
        if not self.collided:
            self.x, self.y = nx, ny
        else:
            v = 0.0
        self.theta = float((self.theta + w * dt + np.pi) % (2 * np.pi) - np.pi)

        # Proprio. Wheel rad/s reflect the TRACK speeds (encoders sit before
        # the slip, like on the rover), gyro/accel reflect the body motion.
        # wheel_source selects WHICH encoder reality the policy sees (item 2).
        if c.wheel_source == "model":
            # the deploy runner's substitution (no encoder noise)
            self.wheel_l = self.v_left / c.wheel_radius
            self.wheel_r = self.v_right / c.wheel_radius
        elif c.wheel_source == "dead_right":
            # the rover's actual defect: the right encoder reads 0
            self.wheel_l = self.v_left / c.wheel_radius \
                + self.rng.normal(0.0, 0.05)
            self.wheel_r = 0.0
        else:
            self.wheel_l = self.v_left / c.wheel_radius \
                + self.rng.normal(0.0, 0.05)
            self.wheel_r = self.v_right / c.wheel_radius \
                + self.rng.normal(0.0, 0.05)
        self.yaw_rate = w + self.rng.normal(0.0, c.gyro_noise_std)
        ax = (v - self._prev_v) / dt + self.rng.normal(0.0, c.accel_noise_std)
        ay = v * w + self.rng.normal(0.0, c.accel_noise_std)
        az = c.gravity + self.rng.normal(0.0, c.accel_noise_std)
        self.accel = np.array([ax, ay, az])
        self._prev_v = v

    # ------------------------------------------------------------------

    def scan(self) -> tuple[np.ndarray, float, float]:
        """One lidar revolution: (ranges[n_beams], angle_min, angle_increment).

        Beam 0 points along the robot's +x (forward), CCW — the convention
        both the safety gate and the brain's preprocessing assume.
        """
        c = self.cfg
        inc = 2.0 * np.pi / c.n_beams
        beam_angles = self.theta + np.arange(c.n_beams) * inc
        if c.lidar_mount_x != 0.0:
            # Origin at the true lidar mount; fan angles stay theta+beam.
            ox = self.x + c.lidar_mount_x * np.cos(self.theta)
            oy = self.y + c.lidar_mount_x * np.sin(self.theta)
        else:
            ox, oy = self.x, self.y
        r = self.world.raycast(ox, oy, beam_angles, c.lidar_max_range)
        r = r + self.rng.normal(0.0, c.lidar_noise_std, size=r.shape)
        drop = self.rng.random(r.shape) < c.lidar_dropout_p
        r = np.where(drop, np.inf, np.maximum(r, 0.02))
        return r.astype(np.float32), 0.0, inc
