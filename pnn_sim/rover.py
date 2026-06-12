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
    robot_radius: float = 0.14    # collision circle

    # Lidar — STL19P-ish: 360 beams, 10-12 Hz on the rover.
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

        # Track slip: the ground sees a noisy fraction of the track speed.
        slip_l = 1.0 + self.rng.normal(0.0, c.slip_std)
        slip_r = 1.0 + self.rng.normal(0.0, c.slip_std)
        vl = self.v_left * slip_l
        vr = self.v_right * slip_r

        v = 0.5 * (vl + vr)
        w = (vr - vl) / c.track_width

        # Integrate pose; on contact, rotation still works but translation
        # stops (tracks stall against the obstacle).
        nx = self.x + v * np.cos(self.theta) * dt
        ny = self.y + v * np.sin(self.theta) * dt
        self.collided = self.world.clearance(nx, ny) < c.robot_radius
        if not self.collided:
            self.x, self.y = nx, ny
        else:
            v = 0.0
        self.theta = float((self.theta + w * dt + np.pi) % (2 * np.pi) - np.pi)

        # Proprio. Wheel rad/s reflect the TRACK speeds (encoders sit before
        # the slip, like on the rover), gyro/accel reflect the body motion.
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
        r = self.world.raycast(self.x, self.y, beam_angles, c.lidar_max_range)
        r = r + self.rng.normal(0.0, c.lidar_noise_std, size=r.shape)
        drop = self.rng.random(r.shape) < c.lidar_dropout_p
        r = np.where(drop, np.inf, np.maximum(r, 0.02))
        return r.astype(np.float32), 0.0, inc
