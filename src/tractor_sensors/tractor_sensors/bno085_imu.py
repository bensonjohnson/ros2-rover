#!/usr/bin/env python3
"""BNO085 IMU publisher — fused orientation from the on-chip SH-2 stack.

Unlike the LSM9DS1 (raw registers, software fusion downstream), the BNO085
runs its own sensor fusion with continuous gyro auto-calibration. We read:

  - game rotation vector: magnetometer-free fused orientation. Drift is
    ~0.5 deg/min spec, far better than integrating a consumer gyro, and it
    never jumps to magnetic north (no magnetometer in the mix).
  - calibrated gyroscope (rad/s) and accelerometer (m/s^2, gravity included
    — the brain's lift detection compares |accel| against 9.81).

Publishes sensor_msgs/Imu on imu/data, same contract as the LSM9DS1 node so
the EKF and the PC brain can consume either interchangeably.

The BNO085 is known to hiccup on I2C (it clock-stretches; some controllers
dislike that): reads are wrapped with retry, and repeated failures trigger a
full chip re-init rather than killing the node.

Axis remap: `upside_down` (180 deg roll mount) negates y/z on the vector
channels and conjugates the quaternion accordingly; `yaw_sign` flips gyro z
and yaw if the chip is rotated. Verify against reality before trusting the
orientation in the EKF: spin the rover CCW (viewed from above) and check
that /imu/data angular_velocity.z is positive and yaw increases.
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    from adafruit_extended_bus import ExtendedI2C
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_GAME_ROTATION_VECTOR,
    )
    BNO_AVAILABLE = True
except ImportError:
    BNO_AVAILABLE = False


class BNO085Publisher(Node):
    def __init__(self):
        super().__init__("bno085_imu_publisher")

        self.declare_parameter("i2c_bus", 5)
        self.declare_parameter("address", 0x4A)       # 0x4B if ADR jumpered
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("update_rate", 50.0)   # Hz
        self.declare_parameter("upside_down", False)  # 180 deg roll mount
        self.declare_parameter("yaw_sign", 1.0)       # -1.0 if chip rotated
        self.declare_parameter("publish_orientation", True)

        self.i2c_bus = int(self.get_parameter("i2c_bus").value)
        self.address = int(self.get_parameter("address").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.update_rate = float(self.get_parameter("update_rate").value)
        self.upside_down = bool(self.get_parameter("upside_down").value)
        self.yaw_sign = float(self.get_parameter("yaw_sign").value)
        self.publish_orientation = bool(
            self.get_parameter("publish_orientation").value)

        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)

        self._i2c = None
        self._bno = None
        self._consecutive_errors = 0

        if not BNO_AVAILABLE:
            self.get_logger().fatal(
                "adafruit_bno08x not installed: pip3 install --user "
                "--break-system-packages adafruit-circuitpython-bno08x "
                "adafruit-extended-bus")
            return

        if not self._init_chip():
            self.get_logger().error(
                "BNO085 init failed; will keep retrying from the read loop")

        self.create_timer(1.0 / max(self.update_rate, 1.0), self._tick)
        self.get_logger().info(
            f"BNO085 on i2c-{self.i2c_bus} addr 0x{self.address:02X}, "
            f"{self.update_rate:.0f} Hz, upside_down={self.upside_down}, "
            f"yaw_sign={self.yaw_sign:+.0f}")

    # ---- hardware ----------------------------------------------------------

    def _init_chip(self) -> bool:
        try:
            if self._i2c is not None:
                try:
                    self._i2c.deinit()
                except Exception:
                    pass
            self._i2c = ExtendedI2C(self.i2c_bus)
            self._bno = BNO08X_I2C(self._i2c, address=self.address)
            self._bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self._bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self._bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
            self._consecutive_errors = 0
            self.get_logger().info("BNO085 initialized")
            return True
        except Exception as e:
            self._bno = None
            self.get_logger().warn(f"BNO085 init failed: {e}")
            return False

    # ---- publishing --------------------------------------------------------

    def _tick(self):
        if self._bno is None:
            # Periodic re-init attempt, throttled to ~1 Hz by error count.
            self._consecutive_errors += 1
            if self._consecutive_errors % max(int(self.update_rate), 1) == 0:
                self._init_chip()
            return
        try:
            gx, gy, gz = self._bno.gyro                  # rad/s
            ax, ay, az = self._bno.acceleration          # m/s^2, with gravity
            qx, qy, qz, qw = self._bno.game_quaternion
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors in (1, 10):
                self.get_logger().warn(f"BNO085 read failed: {e}")
            if self._consecutive_errors >= 25:
                self.get_logger().warn("too many read failures, re-initializing")
                self._init_chip()
            return

        if self.upside_down:
            # 180 deg roll: body x = sensor x, body y/z = -sensor y/z.
            gy, gz, ay, az = -gy, -gz, -ay, -az
            qy, qz = -qy, -qz   # conjugation by R_x(pi): x,w keep sign
        if self.yaw_sign < 0:
            gz = -gz
            qz = -qz            # mirror yaw component

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)
        msg.angular_velocity_covariance = [
            1e-4, 0.0, 0.0,
            0.0, 1e-4, 0.0,
            0.0, 0.0, 1e-4]

        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)
        msg.linear_acceleration_covariance = [
            1e-2, 0.0, 0.0,
            0.0, 1e-2, 0.0,
            0.0, 0.0, 1e-2]

        if self.publish_orientation:
            n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if n > 1e-6:
                msg.orientation.x = float(qx / n)
                msg.orientation.y = float(qy / n)
                msg.orientation.z = float(qz / n)
                msg.orientation.w = float(qw / n)
                msg.orientation_covariance = [
                    1e-3, 0.0, 0.0,
                    0.0, 1e-3, 0.0,
                    0.0, 0.0, 1e-3]
            else:
                msg.orientation_covariance[0] = -1.0
        else:
            msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BNO085Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
