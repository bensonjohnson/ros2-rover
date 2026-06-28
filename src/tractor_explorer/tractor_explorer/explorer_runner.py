"""Deep Exploration Network — ROS2 runner node.

Wires the DeepExplorerNetwork into the existing rover sensor stack and sends
track commands through the safety gate. Supports:
  - RKNN NPU inference (primary) with PyTorch CPU fallback
  - All sensor fusion: LiDAR + occupancy map + proprioception (IMU + wheels)
  - Online experience logging for remote GPU training
  - Metric SLAM integration via RTAB-Map /map topic
  - Shadow teleop (Xbox RB deadman) for safe data collection
"""

import os
import math
import time
import queue
import threading
from typing import Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy

from sensor_msgs.msg import LaserScan, JointState, Imu, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, Float32, Bool, String
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from tractor_explorer.deep_explorer_network import (
    DeepExplorerNetwork, ExplorerConfig,
    normalize_lidar, normalize_proprio,
)

# Try RKNN runtime
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

# Try ZMQ for streaming chunks to training server
try:
    import zmq
    import msgpack
    import msgpack_numpy as mpn
    mpn.patch()
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class ExplorerRunner(Node):
    """ROS2 node that fuses all sensors and runs the Deep Explorer Network.

    Sensor subscribers:
      /scan               — LiDAR (72 bins after preprocessing)
      /joint_states       — wheel velocities (left/right)
      /imu/data           — IMU (yaw rate)
      /map                — RTAB-Map occupancy grid (local crop)
      /camera/camera/aligned_depth_to_color/image_raw  — depth (optional)
      /cmd_vel_teleop     — Xbox shadow teleop
      /emergency_stop     — safety state
      /battery_voltage    — battery monitoring
    """
    def __init__(self):
        super().__init__("explorer_runner")

        # ---- Declare parameters ----
        p = self.declare_parameter
        p("control_rate_hz", 15.0)
        p("lidar_bins", 72)
        p("max_lidar_range", 5.0)
        p("occ_crop_size", 64)
        p("occ_crop_meters", 4.0)     # crop half-width in meters
        p("use_depth", False)
        p("depth_size", 0)
        p("proprio_dim", 5)
        p("action_scale", 0.6)
        p("action_smoothing", 0.35)
        p("exploration_noise", 0.05)
        p("learn", False)              # online learning (requires torch, not RKNN)
        p("model_path", os.path.expanduser("~/.ros/explorer_brain.pt"))
        p("rknn_model_path", os.path.expanduser("~/.ros/explorer_brain.rknn"))
        p("save_interval_s", 60.0)
        p("dashboard_port", 8083)
        p("log_experience", True)
        p("experience_log_path", os.path.expanduser("~/.ros/explorer_experience.jsonl"))
        p("experience_log_max_mb", 256.0)
        p("target_map_frame", "map")
        p("target_odom_frame", "odom")
        p("target_base_frame", "base_link")
        # Mapping mode: auto = full autonomous, explore = frontier-driven,
        # collect = human teleop data collection
        p("mode", "auto")
        # ZMQ chunk streaming to remote training server
        p("server_addr", "")
        p("chunk_len", 64)

        g = self.get_parameter
        self.control_rate = float(g("control_rate_hz").value)
        self.lidar_bins = int(g("lidar_bins").value)
        self.max_lidar_range = float(g("max_lidar_range").value)
        self.occ_crop_size = int(g("occ_crop_size").value)
        self.occ_crop_meters = float(g("occ_crop_meters").value)
        self.use_depth = bool(g("use_depth").value)
        self.action_scale = float(g("action_scale").value)
        self.action_smoothing = float(g("action_smoothing").value)
        self.exploration_noise = float(g("exploration_noise").value)
        self.do_learn = bool(g("learn").value)
        self.model_path = g("model_path").value
        self.rknn_model_path = g("rknn_model_path").value
        self.save_interval_s = float(g("save_interval_s").value)
        self.log_experience = bool(g("log_experience").value)
        self.experience_log_path = g("experience_log_path").value
        self.experience_log_max_bytes = int(
            float(g("experience_log_max_mb").value) * 1024 * 1024)
        self.mode = g("mode").value
        self.server_addr = g("server_addr").value
        self.chunk_len = int(g("chunk_len").value)

        # ZMQ chunk buffer for streaming to training server
        self._chunk_buffer = []
        self._chunks_sent = 0
        self.zmq_ctx = None
        self.zmq_pub = None
        if self.server_addr and HAS_ZMQ:
            try:
                self.zmq_ctx = zmq.Context()
                self.zmq_pub = self.zmq_ctx.socket(zmq.PUSH)
                self.zmq_pub.set_hwm(10)
                self.zmq_pub.connect(self.server_addr)
                self.get_logger().info(f"ZMQ connected: {self.server_addr}")
            except Exception as e:
                self.get_logger().warn(f"ZMQ init failed: {e}")
                self.zmq_pub = None
        elif self.server_addr and not HAS_ZMQ:
            self.get_logger().warn(
                "ZMQ not available — install pyzmq and msgpack for chunk streaming")

        # ---- Build network config ----
        self.net_cfg = ExplorerConfig(
            lidar_bins=self.lidar_bins,
            occ_grid_size=self.occ_crop_size,
            use_depth=self.use_depth,
            depth_size=0 if not self.use_depth else 32,
            proprio_dim=int(g("proprio_dim").value),
        )

        # ---- Load model (RKNN or PyTorch) ----
        self.rknn = None
        self.model: Optional[DeepExplorerNetwork] = None
        self._load_model()

        # ---- State ----
        self.latest_scan: Optional[np.ndarray] = None
        self.latest_occ: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self._wheel_l = 0.0
        self._wheel_r = 0.0
        self._yaw_rate = 0.0
        self._novelty = 0.0
        self._safety_hold = False
        self._teleop_action: Optional[np.ndarray] = None
        self._teleop_stamp = 0.0
        self.teleop_timeout_s = 0.5
        self.exec_action = np.zeros(2, dtype=np.float32)
        self._step = 0
        self._last_save = time.time()
        self._map_available = False
        self._map_warned = False

        # Robot kinematics (from existing stack)
        self.kin_v_max = 0.2
        self.kin_track_width = 0.154

        # ---- ROS I/O ----
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self._imu_cb, qos_profile_sensor_data)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, 10)
        self.estop_sub = self.create_subscription(
            Bool, "/emergency_stop", self._estop_cb, 10)
        self.teleop_sub = self.create_subscription(
            Twist, "/cmd_vel_teleop", self._teleop_cb, 10)

        if self.use_depth:
            self.depth_sub = self.create_subscription(
                Image, "/camera/camera/aligned_depth_to_color/image_raw",
                self._depth_cb, qos_profile_sensor_data)

        # Publishers
        self.track_pub = self.create_publisher(
            Float32MultiArray, "/track_cmd_ai", 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, "/explorer/diagnostics", 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, "/explorer/goal", 10)
        self.viz_pub = self.create_publisher(
            MarkerArray, "/explorer/viz", 10)

        # Experience logging (background thread)
        self.log_queue: Optional[queue.Queue] = None
        self.logger_active = False
        if self.log_experience:
            self.log_queue = queue.Queue()
            self.logger_active = True
            self.logger_thread = threading.Thread(
                target=self._experience_logger_loop, daemon=True)
            self.logger_thread.start()

        # Control timer
        self.timer = self.create_timer(1.0 / self.control_rate, self._control_step)

        self.get_logger().info(
            f"Deep Explorer Network up: "
            f"lidar={self.lidar_bins} occ={self.occ_crop_size}x{self.occ_crop_size} "
            f"proprio={self.net_cfg.proprio_dim} "
            f"mode={self.mode} "
            f"backend={'RKNN' if self.rknn else 'PyTorch'}"
        )

    # ---- Model loading ----

    def _load_model(self):
        """Load RKNN first, fall back to PyTorch."""
        if HAS_RKNN and os.path.exists(self.rknn_model_path):
            try:
                self.rknn = RKNNLite()
                ret = self.rknn.load_rknn(self.rknn_model_path)
                if ret != 0:
                    raise RuntimeError(f"RKNN load returned {ret}")
                # NPU core 0: balanced perf/power; core 1/2 for throughput
                ret = self.rknn.init_runtime(core_mask=0b0011)
                if ret != 0:
                    raise RuntimeError(f"RKNN init returned {ret}")
                self.get_logger().info(
                    f"Loaded RKNN model: {self.rknn_model_path}")
                return
            except Exception as e:
                self.get_logger().warn(f"RKNN load failed: {e}, falling back to PyTorch")
                self.rknn = None

        # PyTorch fallback
        self.model = DeepExplorerNetwork(self.net_cfg)
        self.model.eval()
        if os.path.exists(self.model_path):
            try:
                sd = torch.load(self.model_path, map_location="cpu",
                                weights_only=False)
                # Handle dimension changes
                own = self.model.state_dict()
                for k, v in list(sd.items()):
                    if k in own and v.shape != own[k].shape:
                        self.get_logger().warning(
                            f"Shape mismatch {k}: saved {v.shape} vs "
                            f"current {own[k].shape} — skipping")
                        del sd[k]
                self.model.load_state_dict(sd, strict=False)
                self.get_logger().info(f"Loaded model: {self.model_path}")
            except Exception as e:
                self.get_logger().warn(f"Could not load model: {e}, starting fresh")

    def _reload_rknn(self, path: str) -> bool:
        """Hot-reload RKNN model (e.g. after remote training push)."""
        if not HAS_RKNN:
            return False
        try:
            if self.rknn is not None:
                self.rknn.release()
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(path)
            if ret != 0:
                return False
            ret = self.rknn.init_runtime(core_mask=0b0011)
            self.rknn_model_path = path
            self.get_logger().info(f"Hot-reloaded RKNN model: {path}")
            return ret == 0
        except Exception as e:
            self.get_logger().error(f"RKNN reload failed: {e}")
            return False

    # ---- Callbacks ----

    def _scan_cb(self, msg: LaserScan):
        """Preprocess raw scan into 72 bins, same as pc_active_inference."""
        raw = np.asarray(msg.ranges, dtype=np.float32)
        # Bin by angle: 360° / 72 = 5° per bin
        angle_start = msg.angle_min
        angle_inc = msg.angle_increment
        n_raw = len(raw)
        bin_edges = np.linspace(0, n_raw, self.lidar_bins + 1, dtype=int)
        binned = np.zeros(self.lidar_bins, dtype=np.float32)
        for i in range(self.lidar_bins):
            segment = raw[bin_edges[i]:bin_edges[i + 1]]
            valid = segment[(segment >= 0.05) & (segment <= self.max_lidar_range)]
            if len(valid) > 0:
                binned[i] = float(valid.min())
            else:
                binned[i] = self.max_lidar_range
        # Normalize to [0, 1]: 1 = open, 0 = obstacle
        self.latest_scan = np.clip(
            binned / self.max_lidar_range, 0.0, 1.0).astype(np.float32)

    def _joint_cb(self, msg: JointState):
        try:
            li = msg.name.index("left_viz_wheel_joint")
            ri = msg.name.index("right_viz_wheel_joint")
        except ValueError:
            return
        if len(msg.velocity) > max(li, ri):
            self._wheel_l = float(msg.velocity[li])
            self._wheel_r = float(msg.velocity[ri])

    def _imu_cb(self, msg: Imu):
        self._yaw_rate = float(msg.angular_velocity.z)

    def _map_cb(self, msg: OccupancyGrid):
        """Store map and extract local crop around robot pose.

        The occupancy grid from RTAB-Map uses:
          -1 = unknown, 0 = free, 100 = occupied
        We normalize to [0, 1] for the network:
          0 = free (certain), 1 = occupied (certain), 0.5 = unknown
        """
        self._map_available = True
        self._map_info = msg.info
        self._map_data = np.array(msg.data, dtype=np.float32).reshape(
            msg.info.height, msg.info.width)

    def _depth_cb(self, msg: Image):
        """Store depth image, downsampled to 32×32."""
        import cv2
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width)
        elif msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            return
        # Downsample to 32×32
        small = cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST)
        # Normalize [0, max_range] -> [0, 1] (1 = far)
        self.latest_depth = np.clip(
            small / self.max_lidar_range, 0.0, 1.0).astype(np.float32)

    def _estop_cb(self, msg: Bool):
        self._safety_hold = bool(msg.data)

    def _teleop_cb(self, msg: Twist):
        """Convert twist to per-track command, same scale as actor output."""
        v, w = float(msg.linear.x), float(msg.angular.z)
        vl = v - 0.5 * w * self.kin_track_width
        vr = v + 0.5 * w * self.kin_track_width
        post = np.array([vl, vr], dtype=np.float32) / max(self.kin_v_max, 1e-6)
        self._teleop_action = np.clip(
            post / max(self.action_scale, 1e-6), -1.0, 1.0).astype(np.float32)
        self._teleop_stamp = time.monotonic()

    # ---- Occupancy crop ----

    def _get_occ_crop(self) -> np.ndarray:
        """Crop local occupancy grid around approximate robot position.

        Falls back to a zero array (all-unknown) if no map yet.
        """
        if not self._map_available:
            if not self._map_warned:
                self.get_logger().warning(
                    "No /map topic yet (SLAM still starting) — using unknown crop. "
                    "Rover will explore blind until slam_toolbox publishes a map.",
                    throttle_duration_sec=30.0)
                self._map_warned = True
            return np.full((self.occ_crop_size, self.occ_crop_size), 0.5,
                           dtype=np.float32)

        info = self._map_info
        res = info.resolution                 # m/pixel
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        # Simple odometry-based pose estimate (no lookupTransform needed)
        # In practice, use tf_buffer for the real robot pose
        rx = 0.0   # would use tf lookup
        ry = 0.0

        # Robot position in grid coords
        gx = int((rx - origin_x) / res)
        gy = int((ry - origin_y) / res)
        half = self.occ_crop_size // 2

        # Extract crop
        h, w = self._map_data.shape
        crop = np.full((self.occ_crop_size, self.occ_crop_size), 0.5,
                       dtype=np.float32)
        x_start = max(0, gx - half)
        y_start = max(0, gy - half)
        x_end = min(w, gx + half)
        y_end = min(h, gy + half)
        cx_s = half - (gx - x_start)
        cy_s = half - (gy - y_start)
        cx_e = cx_s + (x_end - x_start)
        cy_e = cy_s + (y_end - y_start)
        patch = self._map_data[y_start:y_end, x_start:x_end]
        # Normalize: -1=unknown→0.5, 0=free→0.0, 100=occupied→1.0
        norm = np.where(patch < 0, 0.5, patch / 100.0)
        crop[cy_s:cy_e, cx_s:cx_e] = norm
        return crop.astype(np.float32)

    # ---- Random exploration (data collection without trained model) ----

    def _random_explore_action(self):
        """Structured random exploration for data collection.

        Cycles through forward, turning, and recovery behaviors so the rover
        actively covers space instead of sitting still. Uses the lidar safety
        gate to detect obstacles and reverse out.

        Returns (action_np, value_np) in track space [-1, 1].
        """
        if not hasattr(self, '_rand_mode'):
            self._rand_mode = 'forward'
            self._rand_ticks = 0
            self._rand_duration = 30   # ~2 seconds at 15Hz
            self._rand_rng = np.random.default_rng()

        self._rand_ticks += 1

        # Stuck detection: safety gate firing while trying to go forward
        if self._safety_hold and self._rand_mode != 'reverse':
            self._rand_mode = 'reverse'
            self._rand_ticks = 0
            self._rand_duration = int(15 + self._rand_rng.uniform(0, 20))  # 1-2.3s

        # Time to pick a new action
        if self._rand_ticks >= self._rand_duration:
            self._rand_ticks = 0
            # Pick a random mode
            modes = ['forward', 'forward_left', 'forward_right',
                     'turn_left', 'turn_right', 'slight_left', 'slight_right']
            # Don't stay in reverse after recovering
            if self._rand_mode == 'reverse':
                modes = ['forward', 'forward_left', 'forward_right',
                         'slight_left', 'slight_right']
            self._rand_mode = str(self._rand_rng.choice(modes))
            # Random duration: 0.5-3 seconds
            self._rand_duration = int(8 + self._rand_rng.uniform(0, 37))

        # Map mode to track commands
        strengths = {
            'forward':        (1.0, 1.0),
            'forward_left':   (0.5, 1.0),
            'forward_right':  (1.0, 0.5),
            'slight_left':    (0.7, 1.0),
            'slight_right':   (1.0, 0.7),
            'turn_left':      (-0.6, 0.6),
            'turn_right':     (0.6, -0.6),
            'reverse':        (-0.6, -0.6),
        }
        L, R = strengths.get(self._rand_mode, (0.5, 0.5))

        # Add jitter so the trajectory isn't perfectly repetitive
        L += float(self._rand_rng.normal(0, 0.08))
        R += float(self._rand_rng.normal(0, 0.08))
        action = np.clip([L, R], -1.0, 1.0).astype(np.float32)

        # Log mode changes occasionally
        if self._rand_ticks < 3:
            self.get_logger().info(f"RandExplore: {self._rand_mode} ({self._rand_duration} ticks)")

        return action, 0.0

    # ---- Control step ----

    def _control_step(self):
        if self.latest_scan is None:
            self._publish_track(0.0, 0.0)
            return

        tick_start = time.monotonic()

        # Build inputs
        lidar_t = torch.from_numpy(self.latest_scan).unsqueeze(0)  # [1, 72]
        occ = self._get_occ_crop()
        occ_t = torch.from_numpy(occ).unsqueeze(0)                 # [1, 64, 64]
        proprio = normalize_proprio(
            torch.tensor([self._wheel_l]), torch.tensor([self._wheel_r]),
            torch.tensor([self._yaw_rate]),
            max_wv=8.0, max_yr=2.5,
            novelty=torch.tensor([self._novelty]),
            safety_hold=torch.tensor([1.0 if self._safety_hold else 0.0]),
        )                                                          # [1, 5]

        depth_t = None
        if self.use_depth and self.latest_depth is not None:
            depth_t = torch.from_numpy(self.latest_depth).unsqueeze(0)

        # Log experience before inference (for off-policy training)
        if self.logger_active and self.log_queue is not None:
            self.log_queue.put({
                "obs": {
                    "lidar": self.latest_scan.tolist(),
                    "occ": occ.tolist(),
                    "proprio": proprio[0].tolist(),
                },
                "action_prev": self.exec_action.tolist(),
                "timestamp": time.time(),
            })

        # Inference — use trained model, or random exploration to collect data
        if self.rknn is not None:
            action_np, value_np = self._rknn_infer(
                self.latest_scan, occ, proprio[0].numpy(), depth_t)
        elif self.model is not None:
            # Check if model has meaningful weights (not fresh random)
            w = next(self.model.parameters())
            if w.abs().mean().item() < 0.01 and self._step < 500:
                # Model is fresh/random — use structured random exploration instead
                action_np, value_np = self._random_explore_action()
            else:
                with torch.no_grad():
                    action_t, value_t = self.model.step(
                        lidar_t, occ_t, proprio,
                        depth=depth_t,
                        deterministic=(self.exploration_noise <= 0.0))
                action_np = action_t[0].numpy()
                value_np = float(value_t[0, 0])
        else:
            # No model file at all — structured random exploration
            action_np, value_np = self._random_explore_action()

        # Shadow teleop override
        teleop = (self._teleop_action is not None
                  and time.monotonic() - self._teleop_stamp < self.teleop_timeout_s)
        if teleop:
            raw = self._teleop_action
        else:
            raw = action_np

        # Smooth + scale
        self.exec_action = (
            self.action_smoothing * raw
            + (1.0 - self.action_smoothing) * self.exec_action)
        out = np.clip(self.exec_action * self.action_scale, -1.0, 1.0)
        self._publish_track(float(out[0]), float(out[1]))

        # Buffer step for ZMQ chunk streaming to training server
        if self.zmq_pub is not None:
            self._chunk_buffer.append({
                "lidar": self.latest_scan.copy(),
                "occ": occ.copy(),
                "proprio": proprio[0].numpy().copy(),
                "action": np.array(out, dtype=np.float32),
                "reward": np.zeros(5, dtype=np.float32),
                "done": False,
                "is_first": len(self._chunk_buffer) == 0,
            })
            if len(self._chunk_buffer) >= self.chunk_len:
                self._send_chunk()
                self._chunk_buffer = []

        self._step += 1
        tick_time = time.monotonic() - tick_start
        if tick_time > 1.0 / self.control_rate:
            self.get_logger().warning(
                f"Tick overran: {tick_time*1000:.0f}ms > "
                f"{1000/self.control_rate:.0f}ms budget",
                throttle_duration_sec=5.0)

        # Diagnostics
        diag = Float32MultiArray()
        diag.data = [float(value_np), float(out[0]), float(out[1]),
                     float(tick_time), float(1.0 if teleop else 0.0),
                     float(self._safety_hold), float(self._novelty)]
        self.diag_pub.publish(diag)

        if self._step % 50 == 0:
            self.get_logger().info(
                f"step={self._step} value={value_np:.3f} "
                f"L={out[0]:+.2f} R={out[1]:+.2f} "
                f"{'[TELEOP]' if teleop else '[AUTO]'}")

        self._maybe_save()

    def _send_chunk(self):
        """Serialize and send a chunk of steps to the training server via ZMQ."""
        if self.zmq_pub is None or not self._chunk_buffer:
            return
        try:
            T = len(self._chunk_buffer)
            chunk = {
                "lidar": np.stack([s["lidar"] for s in self._chunk_buffer]),
                "occ": np.stack([s["occ"] for s in self._chunk_buffer]),
                "proprio": np.stack([s["proprio"] for s in self._chunk_buffer]),
                "action": np.stack([s["action"] for s in self._chunk_buffer]),
                "reward": np.stack([s["reward"] for s in self._chunk_buffer]),
                "done": np.array([s["done"] for s in self._chunk_buffer], dtype=bool),
                "is_first": np.array([s["is_first"] for s in self._chunk_buffer], dtype=bool),
            }
            payload = msgpack.dumps(chunk, default=mpn.encode)
            self.zmq_pub.send(payload, flags=zmq.NOBLOCK)
            self._chunks_sent += 1
            self.get_logger().info(
                f"Chunk #{self._chunks_sent} sent ({T} steps) via ZMQ",
                throttle_duration_sec=5.0)
        except zmq.Again:
            self.get_logger().warning("ZMQ send would block — dropping chunk")
        except Exception as e:
            self.get_logger().error(f"ZMQ send failed: {e}")

    def _rknn_infer(self, lidar_np, occ_np, proprio_np, depth_np=None):
        """Run RKNN inference on the NPU.

        Inputs are already normalized. The RKNN model expects NHWC format
        for 2D inputs and flat vectors for 1D.
        """
        # Build input dict: match ONNX export names
        inputs = {
            "lidar": lidar_np.reshape(1, -1).astype(np.float32),
            "occ_grid": occ_np.reshape(1, self.occ_crop_size,
                                       self.occ_crop_size).astype(np.float32),
            "proprio": proprio_np.reshape(1, -1).astype(np.float32),
        }
        if depth_np is not None and self.use_depth:
            inputs["depth"] = depth_np.reshape(1, 32, 32).astype(np.float32)

        outputs = self.rknn.inference(inputs=list(inputs.values()))
        # Outputs: [action_0, action_1, value] or as exported
        action = np.array([outputs[0][0, 0], outputs[0][0, 1]], dtype=np.float32)
        value = float(outputs[1][0, 0])
        return action, value

    def _publish_track(self, left: float, right: float):
        msg = Float32MultiArray()
        msg.data = [left, right]
        self.track_pub.publish(msg)

    # ---- Persistence ----

    def _maybe_save(self, force: bool = False):
        if not force and time.time() - self._last_save < self.save_interval_s:
            return
        self._last_save = time.time()
        if self.model is None:
            return
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            tmp = self.model_path + ".tmp"
            torch.save(self.model.state_dict(), tmp)
            os.replace(tmp, self.model_path)
            self.get_logger().info(f"Saved model to {self.model_path}")
        except Exception as e:
            self.get_logger().warn(f"Save failed: {e}")

    def destroy_node(self):
        if self.logger_active:
            self.logger_active = False
            if self.log_queue is not None:
                self.log_queue.put(None)
            if hasattr(self, "logger_thread") and self.logger_thread is not None:
                self.logger_thread.join(timeout=2.0)
        self._maybe_save(force=True)
        # Flush remaining ZMQ chunk
        if self.zmq_pub is not None and len(self._chunk_buffer) > 0:
            self._send_chunk()
            self._chunk_buffer = []
        if self.zmq_ctx is not None:
            self.zmq_ctx.destroy(linger=0.5)
        if self.rknn is not None:
            self.rknn.release()
        super().destroy_node()

    # ---- Experience logger ----

    def _experience_logger_loop(self):
        import json
        os.makedirs(os.path.dirname(self.experience_log_path), exist_ok=True)
        self.get_logger().info(
            f"Experience logger -> {self.experience_log_path}")

        f = None
        bytes_written = 0
        try:
            f = open(self.experience_log_path, "a")
            bytes_written = f.tell()
            while self.logger_active or (
                    self.log_queue is not None
                    and not self.log_queue.empty()):
                try:
                    item = self.log_queue.get(timeout=0.5)
                    if item is None:
                        break
                    line = json.dumps(item) + "\n"
                    bytes_written += f.write(line)
                    if bytes_written >= self.experience_log_max_bytes:
                        f.close()
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        part_path = self.experience_log_path.replace(
                            ".jsonl", f"_part_{ts}.jsonl")
                        os.rename(self.experience_log_path, part_path)
                        self.get_logger().info(f"Rotated log -> {part_path}")
                        f = open(self.experience_log_path, "a")
                        bytes_written = 0
                    self.log_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.get_logger().error(f"Log error: {e}")
                    time.sleep(1.0)
        finally:
            if f is not None:
                f.close()


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
