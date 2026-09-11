#!/usr/bin/env python3
"""HARDSTOP hold — latches the rover motionless until explicitly released.

The rover's existing safety signals are all *level-triggered*:
  - /emergency_stop (Bool)  -> hiwonder_motor_driver clamps the tracks only
    while the flag is currently true (``_front_blocked = msg.data``)
  - the motor driver's cmd_vel watchdog zeroes the motors 1 s after the last
    command, but anything still publishing track_cmd/cmd_vel keeps it fed.

So "stop the rover now, and keep it stopped" needs a publisher that *holds*
those zeroes at a rate faster than anything else on the network, and that
keeps running even if the brain process is wedged. That is this node. The
brain supervisor spawns it (own process group) on HARDSTOP and SIGINTs it on
RELEASE; because it is a separate process it survives the brain being
killed, and because the supervisor re-asserts it on every state poll, a
manual kill of this node is repaired within one poll.

Publishes at ``--rate`` Hz:
    /emergency_stop   std_msgs/Bool        = True
    /track_cmd        std_msgs/Float32MultiArray = [0, 0]
    /cmd_vel          geometry_msgs/Twist  = 0

Standalone use (walk up to the rover and kill it):
    ros2 run tractor_bringup hardstop_hold
    # Ctrl-C to release
"""

from __future__ import annotations

import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool


class HardstopHold(Node):
    def __init__(self, rate_hz: float, num_tracks: int):
        super().__init__("hardstop_hold")
        self.pub_estop = self.create_publisher(Bool, "/emergency_stop", 10)
        self.pub_track = self.create_publisher(Float32MultiArray, "/track_cmd", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self._n = num_tracks
        self._ticks = 0
        self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)
        self.get_logger().warn(
            f"HARDSTOP ENGAGED — holding /emergency_stop=true and all motion "
            f"command topics at zero ({rate_hz:.0f} Hz). Release by stopping "
            f"this node.")

    def _tick(self):
        self.pub_estop.publish(Bool(data=True))
        self.pub_track.publish(Float32MultiArray(data=[0.0] * self._n))
        self.pub_cmd.publish(Twist())
        self._ticks += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latch the rover to a full stop")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="Hold publish rate (Hz); must out-publish the brain "
                         "(15 Hz control loop)")
    ap.add_argument("--num-tracks", dest="num_tracks", type=int, default=2)
    args, _unknown = ap.parse_known_args(argv)

    rclpy.init(args=None)
    node = HardstopHold(rate_hz=args.rate, num_tracks=args.num_tracks)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().warn("HARDSTOP released — motion commands live again")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
