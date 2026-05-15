#!/usr/bin/env python3
"""NoMaD inference node — pretrained visual-navigation policy on the RK3588 NPU.

Loads two RKNN submodels (`vision_encoder.rknn`, `noise_pred_net.rknn`) produced
by remote_training_server/nomad_export, runs NoMaD's 10-step diffusion loop in
Python (numpy scheduler in tractor_bringup.nomad_scheduler), converts the
predicted 8-step waypoint trajectory into a differential-drive command via
pure pursuit, and publishes `[left_track, right_track]` on /track_cmd_ai —
the same contract the RLPD runner uses, so lidar_safety_monitor and the motor
driver need no changes.

Modes (launch arg `goal_mode`):
    exploration  — input_goal_mask=1, goal_img zeros. NoMaD's diffusion samples
                   exploratory waypoints. Closest analog to RLPD autonomy.
    image_goal   — input_goal_mask=0, goal_img is the user-provided target.
                   Loaded from a file (param `goal_image_path`) at startup, or
                   pushed via /nomad/goal_image (sensor_msgs/Image).

HIL-SERL pattern: joy_node + teleop_twist_joy stay live. Holding RB on the
Xbox controller flips _intervention_active; we override the policy command
with zeros (teleop_twist already drives /cmd_vel_teleop -> safety monitor),
so the operator stays in control while autonomy is suppressed.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Joy
import tf2_ros

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

from tractor_bringup.frame_stacker import FrameStacker
from tractor_bringup.nomad_scheduler import NoMaDDDPMScheduler
from tractor_bringup.nomad_dashboard import DashboardState, start_dashboard_server


# NoMaD config: context_size=3 PAST frames. The vision encoder input stacks
# those plus the current observation -> context_size + 1 = 4 frames, i.e. a
# 12-channel obs_img tensor.
CONTEXT_SIZE = 3
NUM_OBS_FRAMES = CONTEXT_SIZE + 1
IMAGE_SIZE = 96
PRED_HORIZON = 8
ACTION_DIM = 2
ENCODING_SIZE = 256
NUM_DIFFUSION_ITERS = 10

# ImageNet normalization — NoMaD's vision encoder was trained with these
# stats. Applied in Python (not baked into the RKNN config) so the same
# .rknn files work regardless of how the rover preprocesses upstream.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

JOY_RB_BUTTON_IDX = 5  # matches RLPD runner — Xbox RB is the intervention switch

# NoMaD diffusion output post-processing (the reference repo's `get_action`).
# The model emits NORMALIZED per-step DELTAS in [-1, 1] — NOT absolute
# waypoints. Recover a metric trajectory by:
#   1. unnormalize each delta with the training action stats below,
#   2. cumulative-sum the deltas into a trajectory,
#   3. scale to meters (waypoint_scale param; reference uses max_v / frame_rate).
# Skipping this leaves the "waypoints" as tiny scattered [-1,1] values, which
# makes pure pursuit pivot erratically (spin) and projects nothing on-screen.
ACTION_DELTA_MIN = np.array([-2.5, -4.0], dtype=np.float32)  # [min_dx, min_dy]
ACTION_DELTA_MAX = np.array([5.0, 4.0], dtype=np.float32)    # [max_dx, max_dy]

# Fixed seed for deterministic diffusion sampling (see _tick).
DIFFUSION_SEED = 12345

# Number of candidate trajectories sampled per inference (the fan of paths in
# the NoMaD demo). MUST match --num_samples passed to export_nomad_onnx.py —
# noise_pred_net.rknn is compiled with this as a fixed batch size.
NUM_SAMPLES = 8


def _diffusion_to_waypoints(naction: np.ndarray, scale: float) -> np.ndarray:
    """(N, 8, 2) normalized diffusion output -> (N, 8, 2) metric waypoints.

    Mirrors get_action() from visualnav-transformer: unnormalize the deltas,
    cumulative-sum into a trajectory, scale to meters. Waypoints are in the
    robot frame (+x forward, +y left)."""
    ndeltas = naction.astype(np.float32)  # (N, 8, 2) in [-1, 1]
    deltas = (ndeltas + 1.0) * 0.5 * (ACTION_DELTA_MAX - ACTION_DELTA_MIN) + ACTION_DELTA_MIN
    return np.cumsum(deltas, axis=1) * float(scale)  # cumsum along the 8-step axis


def _quat_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _project_ground_points(
    points_xy: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    rotation: np.ndarray, translation: np.ndarray,
) -> np.ndarray:
    """Project base-frame ground points to image pixels via a pinhole camera.

    `rotation` (3x3) and `translation` (3,) map a point from the base frame
    into the camera optical frame (x=right, y=down, z=forward), i.e.
    p_cam = rotation @ p_base + translation. These come straight from the
    base_footprint -> camera_color_optical_frame TF, so camera height, pitch,
    roll and lateral offset are all handled — no manual extrinsics.

    Args:
        points_xy: (N, 2) array of (x, y) waypoints in meters, ground plane.
    Returns:
        (N, 3) array of (u, v, visible); visible is 0 when the point is
        behind the image plane.
    """
    n = points_xy.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p_base = np.array([points_xy[i, 0], points_xy[i, 1], 0.0], dtype=np.float64)
        p_cam = rotation @ p_base + translation
        z_c = p_cam[2]
        if z_c <= 1e-3:
            continue
        u = cx + fx * p_cam[0] / z_c
        v = cy + fy * p_cam[1] / z_c
        out[i] = (u, v, 1.0)
    return out


def _preprocess_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 H×W×3 -> normalized float32 (3, 96, 96), ImageNet stats."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.astype(np.float32, copy=False)


class NomadRknnRunner(Node):
    def __init__(self):
        super().__init__('nomad_rknn_runner')

        self.declare_parameter('vision_encoder_rknn', '')
        self.declare_parameter('noise_pred_net_rknn', '')
        self.declare_parameter('goal_mode', 'exploration')
        self.declare_parameter('goal_image_path', '')
        # Upstream NoMaD trains waypoints at frame_rate=4 Hz, so the PD
        # controller's DT = 1/rate must match that. Running faster than 4 Hz
        # misinterprets the predicted waypoint spacing and over-drives v/omega.
        self.declare_parameter('inference_rate_hz', 4.0)
        self.declare_parameter('nominal_speed', 0.20)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('lookahead_dist', 0.50)
        # Which predicted waypoint the PD controller chases. Matches upstream
        # NoMaD navigate.py default (index 2). Later indices amplify cumsum
        # error in the diffusion output.
        self.declare_parameter('waypoint_index', 2)
        # Metres per normalized waypoint unit. Upstream NoMaD uses
        # max_v / frame_rate = 0.20 / 4 = 0.05. Any larger value saturates v
        # to nominal_speed every tick, leaving only omega to drive behavior.
        self.declare_parameter('waypoint_scale', 0.05)
        # Upstream NoMaD does NOT reseed the RNG; per-tick noise variation is
        # part of the diffusion sampling. Reseeding collapses the multi-modal
        # output to one repeatable trajectory and pairs badly with mean-of-N
        # selection. Leave False to match upstream; waypoint_smoothing below
        # handles frame-to-frame jitter via an EMA on the chosen trajectory.
        self.declare_parameter('deterministic_sampling', False)
        self.declare_parameter('waypoint_smoothing', 0.5)
        # How many of the 8 predicted waypoints to actually use. Default keeps
        # the full trajectory; lower it only if the cumsum tail gets too noisy.
        self.declare_parameter('num_active_waypoints', 8)
        self.declare_parameter('track_width', 0.30)
        self.declare_parameter('max_track_speed', 0.45)
        # Static-friction floor: nonzero track commands are lifted to at least
        # this magnitude. Matches MIN_TRACK in the RLPD runner.
        self.declare_parameter('min_track', 0.25)
        # Flip turn direction if the rover steers opposite the overlay path
        # (mirrored track polarity on this hardware).
        self.declare_parameter('invert_steering', False)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('goal_image_topic', '/nomad/goal_image')
        # Dashboard + camera projection. Intrinsics (fx/fy/cx/cy) come from
        # CameraInfo; extrinsics come from the base_frame -> camera_optical_frame
        # TF. The fx/fy/cx/cy values here are only fallback defaults (D435i
        # color at 640x480) used until the first CameraInfo arrives.
        self.declare_parameter('dashboard_port', 8081)
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('cam_fx', 460.0)
        self.declare_parameter('cam_fy', 460.0)
        self.declare_parameter('cam_cx', 320.0)
        self.declare_parameter('cam_cy', 240.0)

        self.vision_encoder_path = str(self.get_parameter('vision_encoder_rknn').value)
        self.noise_pred_net_path = str(self.get_parameter('noise_pred_net_rknn').value)
        self.goal_mode = str(self.get_parameter('goal_mode').value).lower()
        self.goal_image_path = str(self.get_parameter('goal_image_path').value)
        self.inference_rate_hz = float(self.get_parameter('inference_rate_hz').value)
        self.nominal_speed = float(self.get_parameter('nominal_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.lookahead_dist = float(self.get_parameter('lookahead_dist').value)
        self.waypoint_index = max(0, int(self.get_parameter('waypoint_index').value))
        self.waypoint_scale = float(self.get_parameter('waypoint_scale').value)
        self.deterministic_sampling = bool(self.get_parameter('deterministic_sampling').value)
        self.waypoint_smoothing = float(self.get_parameter('waypoint_smoothing').value)
        self.num_active_waypoints = max(
            1, min(PRED_HORIZON, int(self.get_parameter('num_active_waypoints').value)))
        # EMA state — previous tick's smoothed trajectory (robot frame).
        self._prev_waypoints: Optional[np.ndarray] = None
        self.track_width = float(self.get_parameter('track_width').value)
        self.max_track_speed = float(self.get_parameter('max_track_speed').value)
        self.min_track = float(self.get_parameter('min_track').value)
        self.invert_steering = bool(self.get_parameter('invert_steering').value)
        rgb_topic = str(self.get_parameter('rgb_topic').value)
        goal_topic = str(self.get_parameter('goal_image_topic').value)
        self.dashboard_port = int(self.get_parameter('dashboard_port').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_optical_frame = str(self.get_parameter('camera_optical_frame').value)
        self.cam_fx = float(self.get_parameter('cam_fx').value)
        self.cam_fy = float(self.get_parameter('cam_fy').value)
        self.cam_cx = float(self.get_parameter('cam_cx').value)
        self.cam_cy = float(self.get_parameter('cam_cy').value)
        # True once intrinsics have been adopted from a CameraInfo message.
        self._intrinsics_from_camera_info = False
        # Cached base_frame -> camera_optical_frame extrinsics (rotation 3x3,
        # translation 3,). Populated from TF; the camera mount is fixed so a
        # single successful lookup is reused for the life of the node.
        self._extrinsics: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._extrinsics_warned = False

        if self.goal_mode not in ('exploration', 'image_goal'):
            self.get_logger().warn(
                f"unknown goal_mode '{self.goal_mode}', falling back to exploration"
            )
            self.goal_mode = 'exploration'

        # RKNN runtimes (loaded lazily — node still comes up if files missing,
        # but no commands are published).
        self._vision_encoder: Optional[RKNNLite] = None
        self._noise_pred_net: Optional[RKNNLite] = None
        self._models_ready = False
        if HAS_RKNN:
            self._models_ready = self._load_models()
        else:
            self.get_logger().error('RKNNLite not available — node will idle')

        # Context buffer of preprocessed RGB frames (CHW float32).
        self._rgb_context = FrameStacker(
            k=NUM_OBS_FRAMES, shape=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )
        self._have_first_frame = False
        self._frame_lock = threading.Lock()
        # Latest raw BGR frame at native resolution — kept for the dashboard
        # overlay (the policy itself only sees the 96x96 preprocessed stack).
        self._latest_bgr: Optional[np.ndarray] = None

        # Goal image. Either loaded from disk once, or replaced via topic.
        # Stored already-preprocessed.
        self._goal_image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        if self.goal_mode == 'image_goal' and self.goal_image_path:
            self._load_goal_from_file(self.goal_image_path)

        self._intervention_active = False
        self._emergency_stop = False

        # Scheduler is reused across inferences — it's stateless across calls
        # except for the noise the caller injects per-step.
        self._scheduler = NoMaDDDPMScheduler(num_train_timesteps=NUM_DIFFUSION_ITERS)

        # Publishers / subscribers.
        self.track_cmd_pub = self.create_publisher(Float32MultiArray, '/track_cmd_ai', 10)
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Image, rgb_topic, self._rgb_cb, 10)
        self.create_subscription(Image, goal_topic, self._goal_cb, latched_qos)
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self.create_subscription(Bool, '/emergency_stop', self._safety_cb, 10)
        self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_cb, 10
        )

        # TF — supplies the camera extrinsics for the dashboard path overlay.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Dashboard: rover-hosted live view with the predicted path overlaid.
        self._dashboard = DashboardState()
        self._dashboard_server = start_dashboard_server(
            self._dashboard, self.dashboard_port
        )

        period = 1.0 / max(self.inference_rate_hz, 0.1)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f'NoMaD runner up: mode={self.goal_mode}, rate={self.inference_rate_hz}Hz, '
            f'nominal_speed={self.nominal_speed} m/s, '
            f'dashboard=http://0.0.0.0:{self.dashboard_port}'
        )

    # ------------------------------------------------------------------ models
    def _load_models(self) -> bool:
        if not self.vision_encoder_path or not self.noise_pred_net_path:
            self.get_logger().error('RKNN paths not provided via params')
            return False
        if not os.path.exists(self.vision_encoder_path):
            self.get_logger().error(f'missing vision encoder: {self.vision_encoder_path}')
            return False
        if not os.path.exists(self.noise_pred_net_path):
            self.get_logger().error(f'missing noise pred net: {self.noise_pred_net_path}')
            return False
        try:
            ve = RKNNLite()
            if ve.load_rknn(self.vision_encoder_path) != 0:
                self.get_logger().error('vision_encoder load_rknn failed')
                return False
            if ve.init_runtime() != 0:
                self.get_logger().error('vision_encoder init_runtime failed')
                return False

            npn = RKNNLite()
            if npn.load_rknn(self.noise_pred_net_path) != 0:
                self.get_logger().error('noise_pred_net load_rknn failed')
                return False
            if npn.init_runtime() != 0:
                self.get_logger().error('noise_pred_net init_runtime failed')
                return False

            self._vision_encoder = ve
            self._noise_pred_net = npn
            self.get_logger().info('RKNN submodels loaded')
            return True
        except Exception as exc:
            self.get_logger().error(f'RKNN load error: {exc}')
            return False

    # ------------------------------------------------------------------ goals
    def _load_goal_from_file(self, path: str) -> None:
        if not os.path.exists(path):
            self.get_logger().warn(f'goal image not found at {path}; exploration mode')
            self.goal_mode = 'exploration'
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f'failed to read goal image {path}; exploration mode')
            self.goal_mode = 'exploration'
            return
        self._goal_image = _preprocess_rgb(bgr)
        self.get_logger().info(f'goal image loaded from {path}')

    def _goal_cb(self, msg: Image) -> None:
        try:
            bgr = self._image_msg_to_bgr(msg)
            self._goal_image = _preprocess_rgb(bgr)
            self.goal_mode = 'image_goal'
            self.get_logger().info('goal image updated via topic')
        except Exception as exc:
            self.get_logger().warn(f'failed to parse goal image: {exc}')

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _image_msg_to_bgr(msg: Image) -> np.ndarray:
        """Hand-rolled sensor_msgs/Image -> BGR uint8 H×W×3, no cv_bridge dep."""
        h, w = msg.height, msg.width
        encoding = msg.encoding
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding == 'bgr8':
            return buf.reshape(h, w, 3)
        if encoding == 'rgb8':
            return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if encoding in ('bgra8', 'rgba8'):
            arr = buf.reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR if encoding == 'bgra8' else cv2.COLOR_RGBA2BGR)
        raise ValueError(f'unsupported image encoding: {encoding}')

    def _rgb_cb(self, msg: Image) -> None:
        try:
            bgr = self._image_msg_to_bgr(msg)
            frame = _preprocess_rgb(bgr)
        except Exception as exc:
            self.get_logger().warn(f'RGB preprocess error: {exc}')
            return
        with self._frame_lock:
            self._latest_bgr = bgr
            if not self._have_first_frame:
                # Bootstrap: fill the context buffer with the first frame so
                # the policy never sees a zero-padded warmup.
                for _ in range(NUM_OBS_FRAMES):
                    self._rgb_context.push(frame)
                self._have_first_frame = True
            else:
                self._rgb_context.push(frame)

    def _joy_cb(self, msg: Joy) -> None:
        try:
            if len(msg.buttons) > JOY_RB_BUTTON_IDX:
                self._intervention_active = bool(msg.buttons[JOY_RB_BUTTON_IDX])
        except Exception:
            pass

    def _safety_cb(self, msg: Bool) -> None:
        self._emergency_stop = bool(msg.data)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Adopt intrinsics from the camera driver. CameraInfo.k is the
        row-major 3x3 matrix [fx 0 cx; 0 fy cy; 0 0 1]."""
        k = msg.k
        if len(k) < 6 or k[0] <= 0.0 or k[4] <= 0.0:
            return
        fx, cx, fy, cy = float(k[0]), float(k[2]), float(k[4]), float(k[5])
        if not self._intrinsics_from_camera_info:
            self.get_logger().info(
                f'adopted camera intrinsics from CameraInfo: '
                f'fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}'
            )
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = fx, fy, cx, cy
        self._intrinsics_from_camera_info = True

    def _push_raw_frame(self) -> None:
        """Stream the live frame with no overlay (used when the policy is
        idle: e-stop, intervention, or models not yet loaded)."""
        with self._frame_lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr
            if bgr is None:
                return
            ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._dashboard.set_frame(buf.tobytes())

    # ------------------------------------------------------------------ inference
    def _tick(self) -> None:
        if self._emergency_stop or self._intervention_active:
            self._publish_track(0.0, 0.0)
            self._push_raw_frame()
            return
        if not self._models_ready or not self._have_first_frame:
            self._publish_track(0.0, 0.0)
            self._push_raw_frame()
            return

        with self._frame_lock:
            context = self._rgb_context.get_stacked()  # (12, 96, 96) float32
            latest_bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
        obs_img = context[None, ...]
        goal_img = self._goal_image[None, ...]
        mask = np.zeros(1, dtype=np.int64) if self.goal_mode == 'image_goal' else np.ones(1, dtype=np.int64)

        try:
            t0 = time.monotonic()
            # Deterministic sampling: reseed the RNG identically every tick so
            # the diffusion output is a (near) deterministic function of the
            # input image. Without this, the random init noise + per-step noise
            # make consecutive inferences on near-identical frames diverge,
            # which is the dominant source of command jitter.
            if self.deterministic_sampling:
                np.random.seed(DIFFUSION_SEED)
            cond_out = self._vision_encoder.inference(inputs=[obs_img, goal_img, mask])
            # vision_encoder is batch-1; tile its 256-d output to the diffusion
            # batch so the rover samples NUM_SAMPLES candidate trajectories.
            obs_cond = np.repeat(
                cond_out[0].astype(np.float32), NUM_SAMPLES, axis=0)  # (N, 256)
            naction = np.random.randn(
                NUM_SAMPLES, PRED_HORIZON, ACTION_DIM).astype(np.float32)
            for k in self._scheduler.timesteps:
                timestep = np.full(NUM_SAMPLES, int(k), dtype=np.int64)
                noise_pred = self._noise_pred_net.inference(
                    inputs=[naction, timestep, obs_cond]
                )[0]
                naction = self._scheduler.step(
                    model_output=noise_pred, timestep=int(k), sample=naction
                )
            dt_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            self.get_logger().error(f'inference failed: {exc}')
            self._publish_track(0.0, 0.0)
            return

        # (N, 8, 2) metric trajectories — the fan of candidate paths.
        all_waypoints = _diffusion_to_waypoints(naction, self.waypoint_scale)
        # Drive the FIRST sample (upstream navigate.py: `naction = naction[0]`).
        # NoMaD's diffusion output is intentionally multi-modal; averaging
        # across samples collapses bimodal choices (e.g. "go left around" vs
        # "go right around") into the mean, which can point straight at the
        # obstacle. Per-tick EMA below handles jitter instead.
        control_path = all_waypoints[0]  # (8, 2)

        # EMA smoothing across ticks on the control path. The robot frame
        # shifts slightly between ticks, but at this rate/speed the shift is
        # small enough that blending is a good jitter damper.
        a = self.waypoint_smoothing
        if (self._prev_waypoints is not None
                and self._prev_waypoints.shape == control_path.shape and a < 1.0):
            control_path = a * control_path + (1.0 - a) * self._prev_waypoints
        self._prev_waypoints = control_path

        # Trim the noisy cumsum tail for control (overlay still draws the full
        # fan so the operator sees the raw distribution).
        control_path = control_path[:self.num_active_waypoints]

        v, omega, lookahead_idx = self._waypoints_to_vw(control_path)
        left, right = self._vw_to_tracks(v, omega)
        self._publish_track(left, right)

        if latest_bgr is not None:
            self._update_dashboard(
                latest_bgr, all_waypoints, control_path, lookahead_idx)

        # Steering diagnostic. Correlate against the dashboard: when the drawn
        # path curves LEFT, dy should be POSITIVE and w positive; tracks should
        # then be L < R. If any of those disagree with what the rover does,
        # that pins down where the sign breaks.
        cw = control_path[lookahead_idx]
        self.get_logger().info(
            f'inference {dt_ms:.0f}ms  wp[{lookahead_idx}] '
            f'dx={cw[0]:+.2f} dy={cw[1]:+.2f}  ->  v={v:+.2f} w={omega:+.2f}  '
            f'tracks L={left:+.2f} R={right:+.2f}',
            throttle_duration_sec=1.0,
        )

    # ------------------------------------------------------------------ control
    def _waypoints_to_vw(self, waypoints: np.ndarray) -> tuple[float, float, int]:
        """NoMaD reference PD controller — steers toward one chosen waypoint.

        Ported from visualnav-transformer deployment/src/pd_controller.py:
            |dx| < EPS:  v = 0;  w = sign(dy) * pi / (2 * DT)
            else:        v = dx / DT;  w = atan2(dy, dx) / DT
        Waypoints are in the robot frame (+x forward, +y left), so dy > 0 ->
        w > 0 -> left turn (ROS convention). `waypoint_index` selects which
        of the predicted points to chase (reference default: 2).

        Returns (v, omega, chosen_idx).
        """
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or waypoints.shape[0] < 1:
            return 0.0, 0.0, 0
        idx = min(self.waypoint_index, waypoints.shape[0] - 1)
        dx, dy = float(waypoints[idx, 0]), float(waypoints[idx, 1])

        dt = 1.0 / max(self.inference_rate_hz, 0.1)
        if abs(dx) < 1e-6:
            v = 0.0
            w = math.copysign(math.pi / (2.0 * dt), dy if dy != 0.0 else 1.0)
        else:
            v = dx / dt
            # atan2 (vs the reference's atan(dy/dx)) keeps the correct sign
            # and quadrant when the chosen waypoint is off to the side.
            w = math.atan2(dy, dx) / dt
        v = float(np.clip(v, 0.0, self.nominal_speed))
        w = float(np.clip(w, -self.max_angular_speed, self.max_angular_speed))
        return v, w, idx

    def _get_extrinsics(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """base_frame -> camera_optical_frame transform as (rotation, translation).

        The camera mount is rigid, so the first successful lookup is cached
        and reused. Returns None until TF can resolve the chain (the RealSense
        driver publishes the optical frame, so this needs the camera up).
        """
        if self._extrinsics is not None:
            return self._extrinsics
        try:
            tf = self._tf_buffer.lookup_transform(
                self.camera_optical_frame, self.base_frame, Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            if not self._extrinsics_warned:
                self.get_logger().warn(
                    f'camera extrinsics TF not available yet '
                    f'({self.base_frame} -> {self.camera_optical_frame}): {exc}'
                )
                self._extrinsics_warned = True
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        rotation = _quat_to_rotation(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        self._extrinsics = (rotation, translation)
        self.get_logger().info(
            f'camera extrinsics adopted from TF '
            f'({self.base_frame} -> {self.camera_optical_frame})'
        )
        return self._extrinsics

    def _draw_path(self, canvas, pix, color, thickness, lookahead_idx=-1):
        """Draw one projected path on `canvas`. Returns (n_visible, n_in_frame).

        cv2.line clips to the image, so segments are drawn even when an
        endpoint is off-frame — that keeps the polyline continuous.
        """
        h, w = canvas.shape[:2]
        prev = None
        n_visible = 0
        n_in_frame = 0
        for i in range(pix.shape[0]):
            if pix[i, 2] < 0.5:
                prev = None  # behind camera — break the polyline
                continue
            n_visible += 1
            u, v = int(round(pix[i, 0])), int(round(pix[i, 1]))
            in_frame = (0 <= u < w and 0 <= v < h)
            if in_frame:
                n_in_frame += 1
            if prev is not None:
                cv2.line(canvas, prev, (u, v), color, thickness, cv2.LINE_AA)
            if in_frame and lookahead_idx >= 0:
                radius = 7 if i == lookahead_idx else 4
                dot = (59, 59, 255) if i == lookahead_idx else color
                cv2.circle(canvas, (u, v), radius, dot, -1, cv2.LINE_AA)
            prev = (u, v)
        return n_visible, n_in_frame

    def _update_dashboard(
        self, bgr: np.ndarray, all_waypoints: np.ndarray,
        control_path: np.ndarray, lookahead_idx: int
    ) -> None:
        """Draw the fan of N sampled paths plus the bold control path onto the
        live frame and push it to the dashboard MJPEG stream."""
        try:
            extrinsics = self._get_extrinsics()
            if extrinsics is None:
                # No TF yet — stream the raw frame so the page is still live.
                ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self._dashboard.set_frame(buf.tobytes())
                return
            rotation, translation = extrinsics
            fx, fy, cx, cy = self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy
            canvas = bgr.copy()

            # Faint fan: every sampled candidate trajectory (thin, dim).
            for s in range(all_waypoints.shape[0]):
                pix_s = _project_ground_points(
                    all_waypoints[s], fx, fy, cx, cy, rotation, translation)
                self._draw_path(canvas, pix_s, (120, 200, 90), 1)

            # Bold control path (the executed mean) + lookahead marker.
            pix_c = _project_ground_points(
                control_path, fx, fy, cx, cy, rotation, translation)
            n_visible, n_in_frame = self._draw_path(
                canvas, pix_c, (20, 255, 20), 3, lookahead_idx)

            # Throttled diagnostic — distinguishes "TF/projection broken"
            # (n_visible low) from "path outside camera FOV" (n_visible high,
            # n_in_frame 0).
            h, w = canvas.shape[:2]
            vs = pix_c[pix_c[:, 2] > 0.5]
            vrange = (
                f'{vs[:, 1].min():.0f}..{vs[:, 1].max():.0f}'
                if len(vs) else 'n/a'
            )
            self.get_logger().info(
                f'overlay: control {n_visible}/{pix_c.shape[0]} in front, '
                f'{n_in_frame} in frame; fan={all_waypoints.shape[0]} paths '
                f'(image {w}x{h}, v-range {vrange})',
                throttle_duration_sec=3.0,
            )
            ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._dashboard.set_frame(buf.tobytes())
        except Exception as exc:
            self.get_logger().debug(f'dashboard render skipped: {exc}')

    def _soft_deadzone(self, x: float) -> float:
        """Lift a nonzero track command into [min_track, 1.0] so it clears the
        tracks' static-friction threshold. Mirrors soft_deadzone() in the RLPD
        runner — PWM magnitudes below ~0.25 produce no motion on this rover."""
        if abs(x) < 1e-3:
            return 0.0
        m = self.min_track
        return math.copysign(m + (1.0 - m) * abs(x), x)

    def _vw_to_tracks(self, v: float, omega: float) -> tuple[float, float]:
        """Differential-drive kinematics -> normalized track speeds in [-1, 1],
        with static-friction compensation.

        `invert_steering` flips the turn direction for hardware whose track
        polarity is mirrored relative to the ROS convention (a learned policy
        like RLPD hides this; hand-coded kinematics do not)."""
        if self.invert_steering:
            omega = -omega
        half = 0.5 * self.track_width
        left_mps = v - omega * half
        right_mps = v + omega * half
        scale = max(self.max_track_speed, 1e-6)
        left = float(np.clip(left_mps / scale, -1.0, 1.0))
        right = float(np.clip(right_mps / scale, -1.0, 1.0))
        return self._soft_deadzone(left), self._soft_deadzone(right)

    def _publish_track(self, left: float, right: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.track_cmd_pub.publish(msg)

    # ------------------------------------------------------------------ lifecycle
    def destroy_node(self):
        try:
            if self._dashboard_server is not None:
                # shutdown() stops the serve loop; server_close() releases the
                # listening socket so a restart can rebind the port.
                self._dashboard_server.shutdown()
                self._dashboard_server.server_close()
        except Exception:
            pass
        try:
            if self._vision_encoder is not None:
                self._vision_encoder.release()
        except Exception:
            pass
        try:
            if self._noise_pred_net is not None:
                self._noise_pred_net.release()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = NomadRknnRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
