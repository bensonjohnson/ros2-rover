#!/usr/bin/env python3
"""
PPO Manager Node (Live Training - Option B)
- Subscribes to /bev/image, /cmd_vel_ai, /odom, /emergency_stop, /min_forward_distance
- Accumulates short on-policy rollouts and runs bounded PPO updates in background
- Exports actor to RKNN upon improvement or min interval and triggers reload in NPU node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

import numpy as np
import time
from cv_bridge import CvBridge
import threading
import os
import glob
import torch

from .ppo_trainer_bev import PPOTrainerBEV
from .rknn_trainer_bev import RKNNTrainerBEV


class PPOManagerNode(Node):
    def __init__(self):
        super().__init__('ppo_manager')

        # Parameters
        self.declare_parameter('bev_image_topic', '/bev/image')
        self.declare_parameter('update_interval_sec', 15.0)
        self.declare_parameter('min_export_interval_sec', 120.0)
        self.declare_parameter('rollout_capacity', 4096)
        self.declare_parameter('minibatch_size', 128)
        self.declare_parameter('update_epochs', 2)
        self.declare_parameter('ppo_clip', 0.2)
        self.declare_parameter('entropy_coef', 0.01)
        self.declare_parameter('value_coef', 0.5)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('gae_lambda', 0.95)
        self.declare_parameter('bev_channels', 4)
        self.declare_parameter('bev_size', [200, 200])
        self.declare_parameter('proprio_dim', 3)  # keep minimal sensor for now
        self.declare_parameter('reward_forward_scale', 5.0)
        self.declare_parameter('reward_block_penalty', -1.0)
        self.declare_parameter('reward_emergency_penalty', -5.0)
        self.declare_parameter('cpu_guard_skip_updates', True)

        self.bev_topic = self.get_parameter('bev_image_topic').value
        self.update_interval = float(self.get_parameter('update_interval_sec').value)
        self.min_export_interval = float(self.get_parameter('min_export_interval_sec').value)
        self.reward_forward_scale = float(self.get_parameter('reward_forward_scale').value)
        self.reward_block_penalty = float(self.get_parameter('reward_block_penalty').value)
        self.reward_emergency_penalty = float(self.get_parameter('reward_emergency_penalty').value)

        C = int(self.get_parameter('bev_channels').value)
        bev_size = self.get_parameter('bev_size').value
        H = int(bev_size[0])
        W = int(bev_size[1])
        proprio_dim = int(self.get_parameter('proprio_dim').value)

        self.trainer = PPOTrainerBEV(
            bev_channels=C, bev_size=(H, W), proprio_dim=proprio_dim,
            rollout_capacity=int(self.get_parameter('rollout_capacity').value),
            minibatch_size=int(self.get_parameter('minibatch_size').value),
            update_epochs=int(self.get_parameter('update_epochs').value),
            ppo_clip=float(self.get_parameter('ppo_clip').value),
            entropy_coef=float(self.get_parameter('entropy_coef').value),
            value_coef=float(self.get_parameter('value_coef').value),
            gamma=float(self.get_parameter('gamma').value),
            gae_lambda=float(self.get_parameter('gae_lambda').value),
        )

        # Export helper reuses existing RKNN pipeline
        self.export_helper = RKNNTrainerBEV(bev_channels=C, enable_debug=False, extra_proprio=proprio_dim - 3)
        self.last_export_time = 0.0

        # State
        self.bridge = CvBridge()
        self.latest_bev = None
        self.last_bev_ts = 0.0
        self.last_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.have_odom = False
        self.emergency = False
        self.min_forward = 10.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_step_time = time.time()

        # Subscriptions
        self.create_subscription(Image, self.bev_topic, self.bev_cb, qos_profile_sensor_data)
        self.create_subscription(Twist, 'cmd_vel_ai', self.act_cb, 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Bool, 'emergency_stop', self.emerg_cb, 10)
        self.create_subscription(Float32, 'min_forward_distance', self.minf_cb, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/ppo_status', 10)

        # Timers
        self.create_timer(self.update_interval, self.update_timer)
        self.create_timer(1.0, self.status_timer)

        # Service client to reload RKNN in NPU node
        self.reload_cli = self.create_client(Trigger, '/reload_rknn')

        self.get_logger().info("PPO Manager initialized (live training, bounded updates)")
        # Try to load latest PPO checkpoint to resume training
        self.load_ppo_checkpoint()

    def bev_cb(self, msg: Image):
        try:
            bev = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if isinstance(bev, np.ndarray) and bev.ndim == 3:
                self.latest_bev = bev.astype(np.float32, copy=False)
                self.last_bev_ts = time.time()
                # On BEV tick, try to add transition if odom/action available
                self.maybe_add_transition()
        except Exception as e:
            self.get_logger().debug(f"BEV conversion failed: {e}")

    def act_cb(self, msg: Twist):
        self.last_action = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def odom_cb(self, msg: Odometry):
        self.have_odom = True
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)

    def emerg_cb(self, msg: Bool):
        self.emergency = bool(msg.data)

    def minf_cb(self, msg: Float32):
        self.min_forward = float(msg.data)

    def compute_reward(self, dt: float) -> float:
        # Progress-based reward with safety penalties
        progress = 0.0
        if self.have_odom and hasattr(self, 'last_pos'):
            progress = float(np.linalg.norm(self.position - self.last_pos))
        r = self.reward_forward_scale * progress
        if self.emergency:
            r += self.reward_emergency_penalty
        # Penalty if trying to go forward while blocked very near
        if self.last_action[0] > 0.0 and self.min_forward <= 0.25:
            r += self.reward_block_penalty
        return r

    def maybe_add_transition(self):
        if self.latest_bev is None or not self.have_odom:
            return
        now = time.time()
        dt = now - self.last_step_time
        if dt < 0.08:  # roughly limit to ~12.5 Hz max
            return
        self.last_step_time = now
        # Observation (sensor minimal zeros for now)
        bev = self.latest_bev
        sens = np.zeros((self.trainer.proprio_dim,), dtype=np.float32)
        # Map action to [-1, 1] policy domain based on expected scaling in runtime
        # Here we assume commands are already normalized in [-1,1] by the NPU; if not, clamp
        act = np.clip(self.last_action.copy(), -1.0, 1.0)
        # Compute reward and done flag
        reward = self.compute_reward(dt)
        done = False  # keep continuous episodes
        # Add transition
        self.trainer.add_transition(bev, sens, act, reward, done)
        # Update last_pos after storing progress
        if self.have_odom:
            self.last_pos = self.position.copy()

    def update_timer(self):
        # Background PPO update with small budget
        stats = self.trainer.update()
        if stats.get('updated'):
            self.get_logger().info(f"PPO updated: size={stats['size']} loss={stats['avg_loss']:.3f}")
            # Export on min interval
            if (time.time() - self.last_export_time) > self.min_export_interval:
                try:
                    # Load actor weights into export helper and convert
                    self.export_helper.model.load_state_dict(self.trainer.actor_state_dict(), strict=False)
                    # Save PPO checkpoint (.pth) alongside RKNN so training can resume later
                    self.save_ppo_checkpoint()
                    self.export_helper.convert_to_rknn()
                    self.last_export_time = time.time()
                    # Request NPU to reload RKNN
                    self.reload_rknn()
                except Exception as e:
                    self.get_logger().warn(f"RKNN export/reload failed: {e}")

    def reload_rknn(self):
        if not self.reload_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("/reload_rknn service not available")
            return
        req = Trigger.Request()
        future = self.reload_cli.call_async(req)
        # Allow longer time for the NPU node to reload the runtime
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.done():
            resp = future.result()
            if resp and resp.success:
                self.get_logger().info(f"RKNN reloaded via NPU node: {resp.message}")
            else:
                self.get_logger().warn(f"RKNN reload service call failed: {getattr(resp, 'message', 'no response')}" )
        else:
            self.get_logger().warn("RKNN reload service call timed out")

    def save_ppo_checkpoint(self):
        try:
            model_dir = getattr(self.export_helper, 'model_dir', 'models')
            os.makedirs(model_dir, exist_ok=True)
            ts = int(time.time())
            ckpt_path = os.path.join(model_dir, f"ppo_actor_critic_{ts}.pth")
            payload = {
                'actor_state_dict': self.trainer.actor.state_dict(),
                'critic_state_dict': self.trainer.critic.state_dict(),
                'log_std': self.trainer.log_std.data.clone().cpu(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'bev_channels': self.trainer.bev_channels,
                'bev_size': self.trainer.bev_size,
                'proprio_dim': self.trainer.proprio_dim,
                'timestamp': ts,
            }
            torch.save(payload, ckpt_path)
            # Update latest symlink
            latest = os.path.join(model_dir, "ppo_actor_critic_latest.pth")
            try:
                if os.path.islink(latest) or os.path.exists(latest):
                    os.remove(latest)
            except Exception:
                pass
            try:
                os.symlink(os.path.basename(ckpt_path), latest)
            except Exception:
                # Symlink might fail on some FS; ignore
                pass
            self.get_logger().info(f"Saved PPO checkpoint: {ckpt_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to save PPO checkpoint: {e}")

    def load_ppo_checkpoint(self):
        try:
            model_dir = getattr(self.export_helper, 'model_dir', 'models')
            latest = os.path.join(model_dir, "ppo_actor_critic_latest.pth")
            ckpt_path = None
            if os.path.exists(latest):
                ckpt_path = latest
            else:
                cand = sorted(glob.glob(os.path.join(model_dir, "ppo_actor_critic_*.pth")))
                if cand:
                    ckpt_path = cand[-1]
            if not ckpt_path:
                self.get_logger().info("No PPO checkpoint found; starting fresh")
                return
            payload = torch.load(ckpt_path, map_location='cpu')
            # Basic shape checks (best-effort)
            self.trainer.actor.load_state_dict(payload.get('actor_state_dict', {}), strict=False)
            self.trainer.critic.load_state_dict(payload.get('critic_state_dict', {}), strict=False)
            log_std = payload.get('log_std')
            if log_std is not None and hasattr(self.trainer, 'log_std'):
                with torch.no_grad():
                    self.trainer.log_std.copy_(log_std.to(self.trainer.log_std.device))
            opt_state = payload.get('optimizer_state_dict')
            if opt_state:
                try:
                    self.trainer.optimizer.load_state_dict(opt_state)
                except Exception:
                    # Optimizer shapes may change; skip silently
                    pass
            ts = payload.get('timestamp', 0)
            if ts:
                self.last_export_time = float(ts)
            self.get_logger().info(f"Loaded PPO checkpoint: {ckpt_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load PPO checkpoint: {e}")

    def status_timer(self):
        msg = String()
        msg.data = f"PPO | buf={self.trainer.buffer.size} last_export={int(time.time()-self.last_export_time)}s ago"
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PPOManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
