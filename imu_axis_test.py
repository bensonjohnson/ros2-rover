#!/usr/bin/env python3
"""Interactive LSM9DS1 axis tester for the proprio yaw-rate channel.

Determines which published /imu/data component is the rover's YAW (vertical)
axis and its sign, so you can set `imu_yaw_axis` / `imu_yaw_sign` correctly for
the cognitive / active-inference brain — important because the IMU is mounted in
a non-standard orientation and the driver only does sign flips, no axis swaps.

It operates on the PUBLISHED values, so whatever it reports is exactly what to
configure, regardless of the driver's internal assumptions.

Usage (on the rover):
    # make sure the IMU driver is publishing /imu/data, e.g. in another shell:
    #   ros2 launch tractor_sensors lsm9ds1_imu.launch.py
    source /opt/ros/jazzy/setup.bash
    python3 imu_axis_test.py
"""

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

AXES = ("x", "y", "z")
G = 9.81


class IMUTester(Node):
    def __init__(self):
        super().__init__("imu_axis_test")
        self.accel = None
        self.gyro = None
        self.count = 0
        self.create_subscription(Imu, "/imu/data", self._cb, qos_profile_sensor_data)

    def _cb(self, msg: Imu):
        a, w = msg.linear_acceleration, msg.angular_velocity
        self.accel = np.array([a.x, a.y, a.z])
        self.gyro = np.array([w.x, w.y, w.z])
        self.count += 1


def collect(node, seconds, live=True):
    """Spin for `seconds`, returning stacked accel and gyro samples."""
    accel, gyro = [], []
    t_end = time.time() + seconds
    last_print = 0.0
    while time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.accel is not None:
            accel.append(node.accel.copy())
            gyro.append(node.gyro.copy())
            if live and time.time() - last_print > 0.25:
                last_print = time.time()
                a, w = node.accel, node.gyro
                sys.stdout.write(
                    f"\r  accel[{a[0]:+5.1f} {a[1]:+5.1f} {a[2]:+5.1f}] m/s²   "
                    f"gyro[{w[0]:+5.2f} {w[1]:+5.2f} {w[2]:+5.2f}] rad/s   ")
                sys.stdout.flush()
    if live:
        print()
    return np.array(accel), np.array(gyro)


def wait_for_data(node):
    print("Waiting for /imu/data ...")
    t_end = time.time() + 5.0
    while time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.count > 5:
            print(f"  receiving IMU data ({node.count} msgs).\n")
            return True
    print("\n  No /imu/data. Start the IMU driver first:")
    print("    ros2 launch tractor_sensors lsm9ds1_imu.launch.py\n")
    return False


def main():
    rclpy.init()
    node = IMUTester()
    try:
        if not wait_for_data(node):
            return

        print("=" * 60)
        print("STEP 1 — GRAVITY (find the vertical / yaw axis)")
        print("=" * 60)
        input("Place the rover STATIONARY and LEVEL on the floor, then press Enter...")
        accel, _ = collect(node, 3.0)
        mean_a = accel.mean(axis=0)
        vert = int(np.argmax(np.abs(mean_a)))
        print(f"\n  mean accel = [{mean_a[0]:+.2f} {mean_a[1]:+.2f} {mean_a[2]:+.2f}] m/s²")
        print(f"  -> gravity is on the {AXES[vert].upper()} axis "
              f"({mean_a[vert]:+.2f}, {'normal' if mean_a[vert] > 0 else 'inverted'})")
        off = np.abs(mean_a[[i for i in range(3) if i != vert]])
        if np.max(off) > 3.0:
            print("  ⚠ large gravity component on a second axis — rover may not be "
                  "level, or the mount is tilted. Re-level and retry for a clean read.")
        print(f"  => the YAW (vertical) axis should be: {AXES[vert]}\n")

        print("=" * 60)
        print("STEP 2 — SPIN (confirm yaw axis + sign)")
        print("=" * 60)
        print("You'll rotate the rover in place to the LEFT (counter-clockwise,")
        print("viewed from above) — the ROS-positive yaw direction.")
        input("Start spinning LEFT and keep spinning, then press Enter...")
        _, gyro = collect(node, 4.0)
        mean_w = gyro.mean(axis=0)
        peak = np.abs(gyro).max(axis=0)
        spin = int(np.argmax(np.abs(mean_w)))
        print(f"\n  mean gyro = [{mean_w[0]:+.2f} {mean_w[1]:+.2f} {mean_w[2]:+.2f}] rad/s")
        print(f"  peak |gyro| = [{peak[0]:.2f} {peak[1]:.2f} {peak[2]:.2f}] rad/s")
        print(f"  -> rotation is strongest on the {AXES[spin].upper()} axis")

        # Recommend sign so that a LEFT turn reads POSITIVE after sign applied.
        sign = 1.0 if mean_w[spin] > 0 else -1.0

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        if spin == vert:
            print(f"  ✓ gravity axis and spin axis AGREE: {AXES[spin]}")
        else:
            print(f"  ⚠ gravity axis ({AXES[vert]}) and spin axis ({AXES[spin]}) DISAGREE.")
            print("    Trust the SPIN result for yaw; re-check that the rover was")
            print("    level in step 1. (Using the spin axis below.)")
        suggested_max = max(0.5, round(float(peak[spin]) * 1.3, 1))
        print(f"\n  imu_yaw_axis := {AXES[spin]}")
        print(f"  imu_yaw_sign := {sign:+.0f}")
        print(f"  (suggest max_yaw_rate ≈ {suggested_max} rad/s so spins don't clip)\n")
        print("  Launch with:")
        print(f"    ros2 launch tractor_bringup pc_active_inference.launch.py \\")
        print(f"        imu_yaw_axis:={AXES[spin]} imu_yaw_sign:={sign:+.0f}\n")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
