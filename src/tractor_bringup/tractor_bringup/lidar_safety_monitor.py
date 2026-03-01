#!/usr/bin/env python3
"""LIDAR-based Path Safety Monitor.

Uses Cartesian path detection instead of angular sectors for the forward
direction. Only obstacles the robot would actually hit (within robot width)
trigger a stop — doorframes and walls to the side are ignored.

Behavior:
- Front path blocked: blocks forward linear.x only, keeps angular.z so the
  rover can turn away. Backward movement also allowed for recovery.
- Slow zone: gradually reduces forward speed as obstacles get closer.
- Side sectors: block turns toward nearby obstacles when moving.
- Rear sector: blocks backward movement toward obstacles.
- Pure rotation (zero-turn) is always allowed for escaping tight spaces.
- Minimum hold time prevents rapid oscillation between BLOCKED/CLEAR states.

The front blocked state is updated scan-driven (not per-command) and published
continuously on /emergency_stop so upstream nodes can respect it.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

# Side/rear sector boundaries (radians) — fixed, not parameterized
_SIDE_MIN = 0.52   # ~30°  — side sectors start here
_SIDE_MAX = 2.62   # ~150° — side sectors end / rear starts


class LidarSafetyMonitor(Node):
    """Path-based safety monitor using LIDAR."""

    def __init__(self) -> None:
        super().__init__('lidar_safety_monitor')

        # Topic parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_teleop')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('input_track_topic', '')   # Empty = disabled
        self.declare_parameter('output_track_topic', '')  # Empty = disabled

        # Safety thresholds
        self.declare_parameter('stop_distance', 0.20)
        self.declare_parameter('slow_distance', 0.40)
        self.declare_parameter('hysteresis', 0.05)
        self.declare_parameter('max_eval_distance', 4.0)
        self.declare_parameter('min_valid_range', 0.05)
        self.declare_parameter('stale_timeout', 0.2)
        self.declare_parameter('min_block_duration', 0.3)  # Minimum hold time (seconds)

        # Robot geometry
        self.declare_parameter('robot_front_offset', 0.06)  # LiDAR to front bumper (m)
        self.declare_parameter('robot_half_width', 0.12)     # Half robot width + margin (m)

        # Load
        self._load_parameters()
        self.add_on_set_parameters_callback(self._parameters_callback)

        # Distances
        self._front_path_dist = self.max_distance
        self._sector_dists = {
            'front': self.max_distance,
            'left': self.max_distance,
            'right': self.max_distance,
            'rear': self.max_distance,
        }
        self._min_overall_dist = self.max_distance

        # State
        self._latest_cmd = None
        self._commands_received = 0
        self._commands_blocked = 0
        self._emergency_stops = 0
        self._sector_stopped = {
            'front': False, 'left': False, 'right': False, 'rear': False
        }
        self._front_blocked_time = None   # Timestamp when front block started
        self._last_scan_time = self.get_clock().now()

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.input_cmd_topic, self.cmd_callback, 10
        )

        # Track command path (direct track control, disabled by default)
        if self.input_track_topic and self.output_track_topic:
            self.track_cmd_sub = self.create_subscription(
                Float32MultiArray, self.input_track_topic, self.track_cmd_callback, 10
            )
            self.track_cmd_pub = self.create_publisher(
                Float32MultiArray, self.output_track_topic, 10
            )
            self.get_logger().info(
                f'Track path enabled: {self.input_track_topic} -> {self.output_track_topic}')
        else:
            self.track_cmd_sub = None
            self.track_cmd_pub = None

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.status_pub = self.create_publisher(String, 'safety_monitor_status', 5)
        self.min_distance_pub = self.create_publisher(Float32, 'min_forward_distance', 5)
        self.sector_dist_pub = self.create_publisher(Float32MultiArray, 'sector_distances', 5)

        # Status timer (10Hz)
        self.create_timer(0.1, self._publish_status)

        self.get_logger().info('Path-based LIDAR safety monitor initialized')
        self._log_params()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _load_parameters(self):
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.input_track_topic = str(self.get_parameter('input_track_topic').value)
        self.output_track_topic = str(self.get_parameter('output_track_topic').value)

        self.stop_dist = float(self.get_parameter('stop_distance').value)
        self.slow_dist = float(self.get_parameter('slow_distance').value)
        self.hysteresis = float(self.get_parameter('hysteresis').value)
        self.max_distance = float(self.get_parameter('max_eval_distance').value)
        self.min_valid_range = float(self.get_parameter('min_valid_range').value)
        self.stale_timeout = float(self.get_parameter('stale_timeout').value)
        self.min_block_duration = float(self.get_parameter('min_block_duration').value)
        self.robot_front_offset = float(self.get_parameter('robot_front_offset').value)
        self.robot_half_width = float(self.get_parameter('robot_half_width').value)

        self.resume_dist = self.stop_dist + self.hysteresis

    def _log_params(self):
        self.get_logger().info(
            f'Params: Stop={self.stop_dist:.2f}m  Slow={self.slow_dist:.2f}m  '
            f'Resume={self.resume_dist:.2f}m  '
            f'MinHold={self.min_block_duration:.2f}s  '
            f'HalfWidth={self.robot_half_width:.3f}m  '
            f'FrontOffset={self.robot_front_offset:.3f}m'
        )

    def _parameters_callback(self, params):
        for param in params:
            if param.name == 'stop_distance':
                self.stop_dist = param.value
                self.resume_dist = self.stop_dist + self.hysteresis
            elif param.name == 'slow_distance':
                self.slow_dist = param.value
            elif param.name == 'hysteresis':
                self.hysteresis = param.value
                self.resume_dist = self.stop_dist + self.hysteresis
            elif param.name == 'robot_front_offset':
                self.robot_front_offset = param.value
            elif param.name == 'robot_half_width':
                self.robot_half_width = param.value
            elif param.name == 'stale_timeout':
                self.stale_timeout = param.value
            elif param.name == 'min_block_duration':
                self.min_block_duration = param.value
        self._log_params()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Scan processing — updates distances AND front blocked state
    # ------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan with path-based front detection.

        The front blocked state is updated here (scan-driven) with hysteresis
        and a minimum hold time to prevent oscillation.
        """
        self._last_scan_time = self.get_clock().now()

        try:
            ranges = np.array(msg.ranges)
            angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
            angles = (angles + np.pi) % (2 * np.pi) - np.pi

            valid = (
                np.isfinite(ranges)
                & (ranges > self.min_valid_range)
                & (ranges <= self.max_distance)
            )

            if not np.any(valid):
                self._front_path_dist = self.max_distance
                for k in self._sector_dists:
                    self._sector_dists[k] = self.max_distance
                self._min_overall_dist = self.max_distance
                self._update_front_blocked_state()
                self.estop_pub.publish(Bool(data=self._sector_stopped['front']))
                return

            r = ranges[valid]
            a = angles[valid]

            # --- Cartesian conversion (x=forward, y=left) ---
            x = r * np.cos(a)
            y = r * np.sin(a)

            # Forward distance from front bumper
            x_bumper = x - self.robot_front_offset

            # === FRONT: path-based — points within the robot's swept path ===
            # When turning, shift the detection corridor laterally to follow
            # the arc the robot will actually drive.  This closes the blind
            # spot between the straight-ahead corridor and the side sectors.
            hw = self.robot_half_width
            cmd = self._latest_cmd
            if (cmd is not None
                    and cmd.linear.x > 0.02
                    and abs(cmd.angular.z) > 0.1):
                v = max(cmd.linear.x, 0.05)
                w = cmd.angular.z  # positive = turning left
                x_fwd = np.maximum(x_bumper, 0.0)
                # Arc lateral offset: R*(1-cos(x/R)) ≈ w*x²/(2v)
                y_shift = np.clip(w * x_fwd * x_fwd / (2.0 * v), -0.30, 0.30)
                in_path = (np.abs(y - y_shift) <= hw) & (x_bumper > 0)
            else:
                in_path = (np.abs(y) <= hw) & (x_bumper > 0)
            if np.any(in_path):
                self._front_path_dist = max(float(np.min(x_bumper[in_path])), 0.0)
            else:
                self._front_path_dist = self.max_distance

            # === Update front blocked state (scan-driven) ===
            self._update_front_blocked_state()

            # === SIDES & REAR: sector-based using absolute angle ===
            abs_a = np.abs(a)
            left_mask = (a > _SIDE_MIN) & (a < _SIDE_MAX)
            right_mask = (a < -_SIDE_MIN) & (a > -_SIDE_MAX)
            rear_mask = abs_a >= _SIDE_MAX

            max_d = self.max_distance
            self._sector_dists['front'] = self._front_path_dist
            self._sector_dists['left'] = float(np.min(r[left_mask])) if np.any(left_mask) else max_d
            self._sector_dists['right'] = float(np.min(r[right_mask])) if np.any(right_mask) else max_d
            self._sector_dists['rear'] = float(np.min(r[rear_mask])) if np.any(rear_mask) else max_d
            self._min_overall_dist = min(self._sector_dists.values())

            # Publish estop on every scan so subscribers always have latest state
            self.estop_pub.publish(Bool(data=self._sector_stopped['front']))

        except Exception as exc:
            self.get_logger().warn(f'LIDAR processing error: {exc}')

    def _update_front_blocked_state(self) -> None:
        """Update front blocked state with hysteresis and minimum hold time.

        - Block when front_path_dist < stop_dist
        - Release when front_path_dist > resume_dist AND min_block_duration has elapsed
        - Prevents rapid oscillation that confuses the motor controller
        """
        now = self.get_clock().now()

        if self._sector_stopped['front']:
            # Currently blocked — release only if BOTH conditions met:
            # 1. Distance exceeds resume threshold (hysteresis)
            # 2. Minimum hold time has elapsed
            if self._front_path_dist > self.resume_dist:
                if self._front_blocked_time is not None:
                    elapsed = (now - self._front_blocked_time).nanoseconds / 1e9
                    if elapsed >= self.min_block_duration:
                        self._sector_stopped['front'] = False
                        self._front_blocked_time = None
                        self.get_logger().info(
                            f'CLEAR (held {elapsed:.1f}s, dist={self._front_path_dist:.2f}m)')
                else:
                    # Edge case: no blocked_time recorded
                    self._sector_stopped['front'] = False
        else:
            # Currently clear — block if too close
            if self._front_path_dist < self.stop_dist:
                self._sector_stopped['front'] = True
                self._front_blocked_time = now
                self._emergency_stops += 1
                self.get_logger().warn(
                    f'BLOCKED front={self._front_path_dist:.2f}m '
                    f'(stop={self.stop_dist:.2f}m)')

    # ------------------------------------------------------------------
    # Command gating
    # ------------------------------------------------------------------

    def track_cmd_callback(self, msg: Float32MultiArray) -> None:
        """Gate direct track commands based on obstacle proximity.

        Receives Float32MultiArray with data[0]=left, data[1]=right in [-1, 1].
        Applies the same safety logic as cmd_callback but in track space,
        preserving the track ratio where possible.
        """
        if len(msg.data) < 2:
            return

        left = float(msg.data[0])
        right = float(msg.data[1])

        # Synthesize Twist for arc-based corridor detection in scan_callback
        # linear.x = average track speed * max_linear_speed_approx
        # angular.z = differential / wheel_separation
        wheel_sep = 0.154
        synth_cmd = Twist()
        synth_cmd.linear.x = (left + right) / 2.0 * 0.154
        synth_cmd.angular.z = (right - left) / wheel_sep
        self._latest_cmd = synth_cmd

        time_since_scan = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9

        if time_since_scan > self.stale_timeout:
            # No recent scan — stop
            left = 0.0
            right = 0.0
        else:
            front_dist = self._front_path_dist

            # --- FRONT: block all forward motion per-track ---
            if self._sector_stopped['front']:
                left = min(left, 0.0)
                right = min(right, 0.0)
            elif front_dist < self.slow_dist:
                # Slow zone: scale both tracks uniformly (preserves turn ratio)
                if max(left, right) > 0.01:
                    scale = (front_dist - self.stop_dist) / (self.slow_dist - self.stop_dist)
                    scale = max(0.0, min(1.0, scale))
                    if left > 0:
                        left *= scale
                    if right > 0:
                        right *= scale

            # --- REAR: block backward motion ---
            if min(left, right) < -0.01:
                rear_dist = self._sector_dists['rear']
                if self._sector_stopped['rear']:
                    if rear_dist > self.resume_dist:
                        self._sector_stopped['rear'] = False
                    else:
                        left = max(left, 0.0)
                        right = max(right, 0.0)
                else:
                    if rear_dist < self.stop_dist:
                        self._sector_stopped['rear'] = True
                        left = max(left, 0.0)
                        right = max(right, 0.0)

            # --- SIDES: during forward+turn, equalize tracks to go straight ---
            avg_forward = (left + right) / 2.0
            is_zero_turn = abs(avg_forward) < 0.05 and abs(left - right) > 0.1

            if not is_zero_turn and avg_forward > 0.01:
                turn_amount = right - left  # positive = turning left
                if turn_amount > 0.1:
                    # Turning left — check left side
                    side_dist = self._sector_dists['left']
                    if side_dist < self.stop_dist:
                        left = right  # Equalize to go straight
                elif turn_amount < -0.1:
                    # Turning right — check right side
                    side_dist = self._sector_dists['right']
                    if side_dist < self.stop_dist:
                        right = left  # Equalize to go straight

        out_msg = Float32MultiArray()
        out_msg.data = [left, right]
        self.track_cmd_pub.publish(out_msg)

    def cmd_callback(self, msg: Twist) -> None:
        """Gate velocity commands based on obstacle proximity.

        Front blocked state is already managed by scan_callback — this only
        applies the gating to individual commands without updating state.
        """
        self._latest_cmd = msg
        self._commands_received += 1

        out_cmd = Twist()
        out_cmd.linear.x = msg.linear.x
        out_cmd.linear.y = msg.linear.y
        out_cmd.linear.z = msg.linear.z
        out_cmd.angular.x = msg.angular.x
        out_cmd.angular.y = msg.angular.y
        out_cmd.angular.z = msg.angular.z

        time_since_scan = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9

        if time_since_scan > self.stale_timeout:
            # No recent scan — stop everything
            out_cmd.linear.x = 0.0
            out_cmd.angular.z = 0.0
            if self._commands_received % 30 == 0:
                self.get_logger().warn(f'STALE DATA ({time_since_scan:.2f}s) — stopping')
        else:
            front_dist = self._front_path_dist

            # --- FRONT: block all forward motion ---
            # When blocked, hard-zero any forward command. Allow backward commands
            # to pass through so the rover can reverse away from obstacles.
            if self._sector_stopped['front']:
                out_cmd.linear.x = min(msg.linear.x, 0.0)
                if msg.linear.x > 0:
                    self._commands_blocked += 1
            elif msg.linear.x > 0.01 and front_dist < self.slow_dist:
                # Gradual slowdown in the slow zone
                scale = (front_dist - self.stop_dist) / (self.slow_dist - self.stop_dist)
                scale = max(0.0, min(1.0, scale))
                out_cmd.linear.x = msg.linear.x * scale

            # --- REAR: only block backward linear.x ---
            if out_cmd.linear.x < -0.01:
                rear_dist = self._sector_dists['rear']
                if self._sector_stopped['rear']:
                    if rear_dist > self.resume_dist:
                        self._sector_stopped['rear'] = False
                    else:
                        out_cmd.linear.x = 0.0
                else:
                    if rear_dist < self.stop_dist:
                        self._sector_stopped['rear'] = True
                        out_cmd.linear.x = 0.0

            # --- FORWARD+TURN: slow/stop forward speed toward turn-side obstacles ---
            # When curving, the rover sweeps into the turn side. Scale down
            # forward speed proportionally to the turn-side obstacle distance.
            if out_cmd.linear.x > 0.01 and abs(msg.angular.z) > 0.1:
                turn_side = 'left' if msg.angular.z > 0 else 'right'
                side_dist = self._sector_dists[turn_side]
                if side_dist < self.slow_dist:
                    side_scale = max(0.0, min(1.0,
                        (side_dist - self.stop_dist) / (self.slow_dist - self.stop_dist)))
                    out_cmd.linear.x *= side_scale

            # --- SIDES: block turns toward obstacles when moving ---
            is_zero_turn = abs(msg.linear.x) < 0.05

            if not is_zero_turn:
                if msg.angular.z > 0.05:
                    left_dist = self._sector_dists['left']
                    if self._sector_stopped['left']:
                        if left_dist > self.resume_dist:
                            self._sector_stopped['left'] = False
                        else:
                            out_cmd.angular.z = 0.0
                    else:
                        if left_dist < self.stop_dist:
                            self._sector_stopped['left'] = True
                            out_cmd.angular.z = 0.0

                if msg.angular.z < -0.05:
                    right_dist = self._sector_dists['right']
                    if self._sector_stopped['right']:
                        if right_dist > self.resume_dist:
                            self._sector_stopped['right'] = False
                        else:
                            out_cmd.angular.z = 0.0
                    else:
                        if right_dist < self.stop_dist:
                            self._sector_stopped['right'] = True
                            out_cmd.angular.z = 0.0
            else:
                # Pure rotation — always allow, clear side stops
                self._sector_stopped['left'] = False
                self._sector_stopped['right'] = False

        self.cmd_pub.publish(out_cmd)

    # ------------------------------------------------------------------
    # Status publishing
    # ------------------------------------------------------------------

    def _publish_status(self) -> None:
        self.min_distance_pub.publish(Float32(data=float(self._front_path_dist)))

        sector_msg = Float32MultiArray()
        sector_msg.data = [
            float(self._sector_dists['front']),
            float(self._sector_dists['left']),
            float(self._sector_dists['right']),
            float(self._sector_dists['rear']),
        ]
        self.sector_dist_pub.publish(sector_msg)

        # Publish estop continuously so late-joining subscribers get current state
        self.estop_pub.publish(Bool(data=self._sector_stopped['front']))

        blocked = [s for s, v in self._sector_stopped.items() if v]
        f = self._sector_dists
        dist_str = f'F:{f["front"]:.2f} L:{f["left"]:.2f} R:{f["right"]:.2f} B:{f["rear"]:.2f}'
        if blocked:
            status = f'BLOCKED: {", ".join(blocked)} | {dist_str}'
        elif self._front_path_dist < self.slow_dist:
            status = f'SLOW | {dist_str}'
        else:
            status = f'CLEAR | {dist_str}'
        self.status_pub.publish(String(data=status))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
