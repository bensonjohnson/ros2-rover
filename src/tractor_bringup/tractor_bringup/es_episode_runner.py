#!/usr/bin/env python3
"""ES-SAC Episode Runner (Thin Client).

Remote Inference Mode:
1. Collect Sensor Data.
2. Serialize & Send to Server (NATS).
3. Receive Action.
4. Execute.
"""

import os
import time
import json
import threading
import asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import nats
from tractor_bringup.serialization_utils import serialize_batch
from tractor_bringup.occupancy_processor import RawSensorProcessor

class ESEpisodeRunner(Node):
    def __init__(self):
        super().__init__('es_episode_runner')
        
        # Params
        self.declare_parameter('nats_server', 'nats://nats.gokickrocks.org:4222')
        self.declare_parameter('max_episode_steps', 1000) # ~33 seconds @ 30Hz
        
        self.nats_server_url = self.get_parameter('nats_server').value
        self.max_idx = self.get_parameter('max_episode_steps').value
        
        # ROS
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_ai', 10)
        
        # Sensors
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Bool, '/safety_monitor_status', self._safety_cb, 10)

        # State vars
        self.latest = {'scan': None, 'depth': None, 'odom': None, 'imu': None, 'wheels': None}
        self.safety_override = False
        
        # Processing
        self.processor = RawSensorProcessor(grid_size=128, max_range=4.0)
        
        # NATS
        self.nc = None
        
        # Control Loop
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        
    def _depth_cb(self, msg): self.latest['depth'] = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
    def _scan_cb(self, msg): self.latest['scan'] = msg
    def _odom_cb(self, msg): self.latest['odom'] = msg
    def _imu_cb(self, msg): self.latest['imu'] = msg
    def _joint_cb(self, msg): self.latest['wheels'] = msg
    def _safety_cb(self, msg): self.safety_override = msg.data

    async def connect_nats(self):
        try:
            self.nc = await nats.connect(self.nats_server_url, max_reconnect_attempts=-1)
            print("Connected to NATS")
        except Exception as e:
            print(f"NATS Connection Error: {e}")

    def _main_loop(self):
        asyncio.run(self._async_main_loop())
        
    async def _async_main_loop(self):
        await self.connect_nats()
        # Wait for sensors
        time.sleep(2.0)
        
        while rclpy.ok():
            try:
                # 1. Start Episode (Implicit by sending data)
                # But we need to signal 'New Episode' to separate data?
                # Actually, continuous training is fine, but for ES score we need boundaries.
                # Let's run fixed length episodes.
                
                print(f"Starting Episode (Remote Inference)...")
                data = await self.run_episode()
                
                # 2. Send Results
                if data and self.nc:
                    print(f"Episode Done. sending results (Reward: {np.sum(data['rewards']):.2f})")
                    # We need to know which model ID we used.
                    # The server assigns it. We assume server tracks assignments by rover_id.
                    # We just query "What was my model?" OR server knows.
                    # Protocol hack: Send result with rover_id, server looks up active model.
                    data['metadata'] = {'rover_id': 'rover_1', 'model_id': -1} # Server will fill model_id
                    
                    payload = serialize_batch(data)
                    await self.nc.publish("es.episode_result", payload)
                
                # Brief pause between episodes
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error in loop: {e}")
                await asyncio.sleep(1)

    async def run_episode(self):
        """Run single remote inference episode."""
        laser_buf, depth_buf, proprio_buf = [], [], []
        action_buf, reward_buf, done_buf = [], [], []
        
        steps = 0
        
        # Pre-allocate reuse buffers
        header_dict = {'rover_id': 'rover_1'}
        header_bytes = json.dumps(header_dict).encode()
        header_len = len(header_bytes).to_bytes(2, 'big')
        
        while steps < self.max_idx and rclpy.ok():
            step_start = time.perf_counter()
            
            if self.latest['scan'] is None or self.latest['depth'] is None:
                await asyncio.sleep(0.01)
                continue
                
            # 1. Process local sensors
            laser, depth = self.processor.process(self.latest['depth'], self.latest['scan'])
            # Quantize to uint8 for transport (0 or 1 for laser, 0-255 for depth)
            laser_u8 = laser.astype(np.uint8) 
            depth_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
            
            proprio = np.zeros(10, dtype=np.float32) # Fill real values ideally
            
            # 2. Transmit to Server
            if self.nc and self.nc.is_connected:
                # Payload: HeaderLen(2) + Header + Laser + Depth + Proprio
                payload = header_len + header_bytes + laser_u8.tobytes() + depth_u8.tobytes() + proprio.tobytes()
                
                try:
                    # Request-Reply pattern (RPC)
                    response = await self.nc.request("es.step_inference", payload, timeout=0.2)
                    
                    # 3. Receive Action
                    action = np.frombuffer(response.data, dtype=np.float32)
                except TimeoutError:
                    print("TIMEOUT waiting for action from server!")
                    action = np.zeros(2, dtype=np.float32) # Stop
            else:
                action = np.zeros(2, dtype=np.float32)

            # 4. Execute
            msg = Twist()
            msg.linear.x = float(action[0] * 0.2) # Scale max speed
            msg.angular.z = float(action[1] * 1.0)
            if self.safety_override: 
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                
            self.cmd_pub.publish(msg)
            
            # 5. Calculate Reward (Local)
            r = msg.linear.x 
            if self.safety_override: r -= 1.0
            
            # Store
            laser_buf.append(laser) # Store full float for training? Or the sent uint8?
            # Replay buffer expects uint8 for storage anyway.
            # But the 'add_batch' util might expect specific shapes.
            # Let's store what we have.
            laser_buf.append(laser_u8.reshape(1, 128, 128))
            depth_buf.append(depth_u8.reshape(1, 100, 848))
            proprio_buf.append(proprio)
            action_buf.append(action)
            reward_buf.append(r)
            done_buf.append(False)
            
            steps += 1
            
            # Control rate
            elapsed = time.perf_counter() - step_start
            wait = 0.033 - elapsed
            if wait > 0: await asyncio.sleep(wait)
            
        # Stop
        self.cmd_pub.publish(Twist())
            
        return {
            'laser': np.array(laser_buf), # (T, 1, 128, 128)
            'depth': np.array(depth_buf), # (T, 1, 100, 848)
            'proprio': np.array(proprio_buf),
            'actions': np.array(action_buf),
            'rewards': np.array(reward_buf),
            'dones': np.array(done_buf)
        }

def main():
    rclpy.init()
    node = ESEpisodeRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
