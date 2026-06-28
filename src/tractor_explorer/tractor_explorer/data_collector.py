"""Data Collector — records sensor + action experience for remote GPU training.

Shipped via ZMQ to the training server. Collects at the control rate and
batches into fixed-length chunks for efficient network transport.
"""

import os
import sys
import json
import time
import struct
import threading
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan, JointState, Imu, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, Bool

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class DataCollector(Node):
    """Collects experience chunks and ships them to the remote training server.

    Chunk format (chunk_len consecutive timesteps):
      - lidar:    [T, 72]    float32 normalized [0,1]
      - occ:      [T, 64,64] float32 occupancy crop
      - proprio:  [T, 5]     float32 [vx, vyaw, imu_yaw, novelty, hold]
      - action:   [T, 2]     float32 raw action [-1, 1]
      - reward:   [T, 5]     float32 [progress, smoothness, turn_penalty,
                                       collision, coverage_gain]
      - done:     [T]        bool episode done?
      - timestamp:[T]        float64 wall time

    Sends via ZMQ PUSH to the remote server.
    """
    def __init__(self):
        super().__init__("data_collector")

        p = self.declare_parameter
        p("chunk_len", 64)
        p("max_chunks_pending", 100)
        p("server_addr", "tcp://192.168.1.100:5557")
        p("use_zmq", True)
        p("save_local", True)
        p("local_path", os.path.expanduser("~/.ros/explorer_chunks"))
        p("rate_hz", 15.0)

        g = self.get_parameter
        self.chunk_len = int(g("chunk_len").value)
        self.max_pending = int(g("max_chunks_pending").value)
        self.server_addr = g("server_addr").value
        self.use_zmq = bool(g("use_zmq").value) and HAS_ZMQ
        self.save_local = bool(g("save_local").value)
        self.local_path = g("local_path").value

        # Buffer
        self.buffer = deque(maxlen=self.chunk_len)
        self.pending = deque(maxlen=self.max_pending)
        self._chunk_id = 0

        # State for reward calculation
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._safety_hold = False
        self._prev_coverage = 0.0

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self._imu_cb, qos_profile_sensor_data)
        self.track_sub = self.create_subscription(
            Float32MultiArray, "/track_cmd_ai", self._track_cb, 10)
        self.estop_sub = self.create_subscription(
            Bool, "/emergency_stop", self._estop_cb, 10)

        # ZMQ
        self.zmq_ctx = None
        self.zmq_sock = None
        if self.use_zmq:
            try:
                self.zmq_ctx = zmq.Context()
                self.zmq_sock = self.zmq_ctx.socket(zmq.PUSH)
                self.zmq_sock.set_hwm(50)
                self.zmq_sock.connect(self.server_addr)
                self.get_logger().info(f"ZMQ connected to {self.server_addr}")
            except Exception as e:
                self.get_logger().warn(f"ZMQ init failed: {e}")
                self.use_zmq = False

        # Local storage
        if self.save_local:
            os.makedirs(self.local_path, exist_ok=True)

        self._step = 0
        self._chunks_sent = 0

        self.get_logger().info(
            f"Data collector ready: chunk={self.chunk_len} "
            f"zmq={self.use_zmq} local={self.save_local}")

    # ---- Message callbacks (store state for buffer) ----

    def _scan_cb(self, msg: LaserScan):
        pass  # handled by explorer_runner; we read latest from topic via service

    def _joint_cb(self, msg: JointState):
        pass

    def _imu_cb(self, msg: Imu):
        pass

    def _track_cb(self, msg: Float32MultiArray):
        self._prev_action = np.array(msg.data[:2], dtype=np.float32)

    def _estop_cb(self, msg: Bool):
        self._safety_hold = bool(msg.data)

    # ---- Buffer management ----

    def add_step(self, obs: dict, action: np.ndarray, done: bool = False):
        """Called by explorer_runner each control tick.

        obs contains: lidar (72,), occ (64,64), proprio (5,)
        """
        if not (self.use_zmq or self.save_local):
            return

        # Compute per-step reward
        # [progress, smoothness, turn_penalty, collision, coverage_gain]
        vx = obs["proprio"][0]           # normalized wheel velocity
        vyaw = obs["proprio"][1]         # normalized yaw rate

        progress = max(0.0, float(vx))
        smoothness = -float(np.linalg.norm(
            self._prev_action - action))
        turn_penalty = -abs(float(vyaw))
        collision = -1.0 if self._safety_hold else 0.0
        coverage_gain = 0.0  # updated externally by explore_manager

        reward = np.array([
            progress, smoothness, turn_penalty, collision, coverage_gain
        ], dtype=np.float32)

        step = {
            "lidar": obs["lidar"].astype(np.float32),
            "occ": obs["occ"].astype(np.float32),
            "proprio": obs["proprio"].astype(np.float32),
            "action": action.astype(np.float32),
            "reward": reward,
            "done": bool(done),
            "t": time.time(),
        }
        self.buffer.append(step)
        self._prev_action = action.copy()

        # When buffer fills, emit a chunk
        if len(self.buffer) >= self.chunk_len:
            self._emit_chunk()
            self.buffer.clear()

    def _emit_chunk(self):
        """Convert buffer to chunk and send."""
        steps = list(self.buffer)
        T = len(steps)
        if T < 2:
            return

        chunk = {
            "lidar": np.stack([s["lidar"] for s in steps]),
            "occ": np.stack([s["occ"] for s in steps]),
            "proprio": np.stack([s["proprio"] for s in steps]),
            "action": np.stack([s["action"] for s in steps]),
            "reward": np.stack([s["reward"] for s in steps]),
            "done": np.array([s["done"] for s in steps], dtype=bool),
            "is_first": np.zeros(T, dtype=bool),
            "timestamp": np.array([s["t"] for s in steps], dtype=np.float64),
        }
        chunk["is_first"][0] = True

        self._chunk_id += 1
        self._chunks_sent += 1

        if self.use_zmq and self.zmq_sock is not None:
            try:
                self._send_zmq(chunk)
            except Exception as e:
                self.get_logger().error(f"ZMQ send failed: {e}")

        if self.save_local:
            self._save_local(chunk)

    def _send_zmq(self, chunk: dict):
        """Serialize chunk and send via ZMQ.

        Uses msgpack for compact binary encoding of numpy arrays.
        """
        try:
            import msgpack
            import msgpack_numpy as mpn
            mpn.patch()

            payload = msgpack.dumps(chunk, default=mpn.encode)
            self.zmq_sock.send(payload, flags=zmq.NOBLOCK)
            if self._chunks_sent % 10 == 0:
                self.get_logger().info(
                    f"Sent chunk #{self._chunks_sent} ({len(self.buffer)} steps)")
        except zmq.Again:
            self.get_logger().warning("ZMQ send would block — dropping chunk")
        except Exception as e:
            self.get_logger().error(f"ZMQ serialize: {e}")

    def _save_local(self, chunk: dict):
        """Save chunk to local disk as numpy .npz."""
        path = os.path.join(
            self.local_path,
            f"chunk_{self._chunk_id:06d}.npz")
        np.savez_compressed(path, **chunk)

    def destroy_node(self):
        # Flush remaining buffer
        if len(self.buffer) > 0:
            self._emit_chunk()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
