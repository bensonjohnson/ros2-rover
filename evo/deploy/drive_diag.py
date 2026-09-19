#!/usr/bin/env python3
"""Fixed-command drivetrain diagnostic: does each track produce the thrust
its command asks for? Publishes constant (L,R) on /track_cmd_ai (safety
monitor in path) for 4 s per phase; records wheel velocities (joint_states),
gyro yaw, gate-passed values and estop. Phases: forward, reverse, pivot-L,
pivot-R, hard forward."""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool


class Diag(Node):
    def __init__(self):
        super().__init__("drive_diag")
        self.wl = self.wr = 0.0
        self.gyro = 0.0
        self.gated = (0.0, 0.0)
        self.estop = False
        self.create_subscription(JointState, "/joint_states",
                                 self.on_js, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu/imu", self.on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/track_cmd",
                                 self.on_g, 10)
        self.create_subscription(Bool, "/emergency_stop", self.on_e, 10)
        self.pub = self.create_publisher(Float32MultiArray,
                                         "/track_cmd_ai", 10)

    def on_js(self, m):
        try:
            li = m.name.index("left_viz_wheel_joint")
            ri = m.name.index("right_viz_wheel_joint")
        except ValueError:
            return
        if len(m.velocity) > max(li, ri):
            self.wl, self.wr = (float(m.velocity[li]),
                                float(m.velocity[ri]))

    def on_imu(self, m):
        self.gyro = float(m.angular_velocity.z)

    def on_g(self, m):
        if len(m.data) >= 2:
            self.gated = (float(m.data[0]), float(m.data[1]))

    def on_e(self, m):
        self.estop = bool(m.data)

    def phase(self, l, r, dur=4.0):
        rows = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < dur:
            msg = Float32MultiArray()
            msg.data = [float(l), float(r)]
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            rows.append((time.monotonic() - t0, self.wl, self.wr,
                         self.gyro, *self.gated, float(self.estop)))
            time.sleep(0.05)
        self.pub.publish(Float32MultiArray(data=[0.0, 0.0]))
        return np.array(rows)


def main():
    rclpy.init()
    n = Diag()
    out = {}
    for name, l, r in [("fwd_05", 0.5, 0.5), ("rev_05", -0.5, -0.5),
                       ("pivL", -0.7, 0.7), ("pivR", 0.7, -0.7),
                       ("fwd_09", 0.9, 0.9)]:
        print(f"--- phase {name}: L{l:+.1f} R{r:+.1f} (4 s) ---",
              flush=True)
        out[name] = n.phase(l, r)
    n.pub.publish(Float32MultiArray(data=[0.0, 0.0]))
    time.sleep(0.3)
    rclpy.shutdown()
    np.savez("/tmp/drive_diag.npz", **out)
    for k, a in out.items():
        wl, wr, gy = a[:, 1], a[:, 2], a[:, 3]
        gL, gR, es = a[:, 4], a[:, 5], a[:, 7]
        mid = slice(20, None)     # skip first 1 s spin-up
        print(f"{k:8s} gated L{gL[mid].mean():+.2f} R{gR[mid].mean():+.2f}"
              f"  actual wl {wl[mid].mean():+7.1f} wr {wr[mid].mean():+7.1f}"
              f" rad/s  yaw {gy[mid].mean():+.2f} rad/s  estop {es.mean():.0%}")
    # expected: fwd cmd 0.5 -> wheel ~ 0.5*0.2/0.025 = 4 rad/s (trim 0.8/1.0)


if __name__ == "__main__":
    main()
