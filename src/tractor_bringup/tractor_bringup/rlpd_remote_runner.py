#!/usr/bin/env python3
"""RLPD + HIL-SERL Remote Runner — model-free SAC inference on rover, training on remote GPU.

v3 observation pipeline (replaces legacy RGB+BEV+ResNet18):
  - RKNN inputs (K=frame_stack=4 by default):
        depth_stack   (1, K, 72, 96) float32 in [0, 1]
        lidar_stack   (1, K, 360)    float32 in [0, 1]
        proprio_stack (1, 6*K)       float32 (normalized)
    Output is mean_logstd (1, 4).
  - Realsense D435i 640×480 depth is clipped to [0.2, 6.0]m, resized to 96×72
    with INTER_NEAREST (preserves invalid-edge geometry), stored uint8 on the
    wire (trainer divides by 255).
  - STL19P /scan ranges are resampled to 360 beams via np.interp against the
    angle grid, invalid/inf replaced by range_max, normalized to [0, 1].
  - Episode boundary / model swap resets the frame stacks (zero-pad).
  - HIL-SERL interventions: joy_node + teleop_twist_joy stay running in
    autonomous mode. Holding RB (deadman) overrides the policy action with
    the teleop-derived track command for that tick AND sets
    `is_intervention[t] = True` and `rewards[t, 4] = -1.0` — policy learns
    to avoid needing correction.
  - Chunk schema_version=3 on the wire. demos.npz from v2 is incompatible
    and gets archived by the server on first v3 start.
  - Calibration .npz captures stacked depth/lidar/proprio for RKNN INT8.
"""

import os
import json
import math
import time
import threading
import asyncio
import tempfile
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
from cv_bridge import CvBridge

import zmq
import zmq.asyncio
import msgpack

from tractor_bringup.serialization_utils import (
    serialize_batch, deserialize_status
)
from tractor_bringup.frame_stacker import FrameStacker

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

from sensor_msgs.msg import Image, Imu, JointState, LaserScan, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Reward channel order — must match REWARD_CHANNELS_RLPD on the server.
RLPD_REWARD_CHANNELS = ('smoothness', 'progress', 'turn_penalty', 'collision', 'intervention')
N_REWARD_CHANNELS = len(RLPD_REWARD_CHANNELS)

PROPRIO_DIM = 6
ACTION_DIM = 2
PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)

# Xbox RB button index — used for HIL-SERL interventions.
JOY_RB_BUTTON_IDX = 5

# v3 observation shapes — must match the server's RLPDVisionEncoderV3.
V3_SCHEMA_VERSION = 3
DEPTH_H, DEPTH_W = 72, 96            # resize target for D435i depth
DEPTH_CLIP_MIN_MM = 200              # 0.2 m
DEPTH_CLIP_MAX_MM = 6000             # 6.0 m
DEPTH_RANGE_MM = DEPTH_CLIP_MAX_MM - DEPTH_CLIP_MIN_MM
LIDAR_BEAMS = 360                    # resampled count
LIDAR_DEFAULT_MAX_RANGE = 12.0       # STL19P typical max range


def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    return np.clip(normalized, -3.0, 3.0).astype(np.float32)


# ============================================================================
# Chunk Buffer — v3: depth (uint8 96×72) + lidar (float32 360) + proprio
# ============================================================================


class ChunkBuffer:
    """Buffer of single-frame transitions for the v3 wire format.

    Reward channels (T, 5): coverage, frontier, collision, episodic, intervention.
    Server-side frame stacking happens at replay sample time so the wire
    format stays minimal (one frame per step).
    """

    def __init__(self, chunk_len: int, proprio_dim: int = PROPRIO_DIM,
                 reward_dim: int = N_REWARD_CHANNELS):
        self.chunk_len = chunk_len
        self.capacity = chunk_len + 16  # small slack
        self.reward_dim = reward_dim
        self.ptr = 0
        self.size = 0
        self.depth = np.zeros((self.capacity, 1, DEPTH_H, DEPTH_W), dtype=np.uint8)
        self.lidar = np.zeros((self.capacity, LIDAR_BEAMS), dtype=np.float32)
        self.proprio = np.zeros((self.capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, reward_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.is_first = np.zeros(self.capacity, dtype=np.bool_)
        self.is_intervention = np.zeros(self.capacity, dtype=np.bool_)
        self.is_demo = np.zeros(self.capacity, dtype=np.bool_)

    def add(self, depth_u8, lidar, proprio, action, reward, done, is_first,
            is_intervention=False, is_demo=False):
        i = self.ptr
        # depth_u8 must already be uint8 (1, H, W) in [0, 255]; lidar float32 (360,) in [0, 1]
        self.depth[i] = depth_u8
        self.lidar[i] = lidar
        self.proprio[i] = proprio
        self.actions[i] = action
        r = np.asarray(reward, dtype=np.float32)
        if r.ndim == 0:
            r = np.broadcast_to(r, (self.reward_dim,))
        self.rewards[i] = r
        self.dones[i] = done
        self.is_first[i] = is_first
        self.is_intervention[i] = bool(is_intervention)
        self.is_demo[i] = bool(is_demo)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_chunk(self):
        n = self.size
        return {
            'depth': self.depth[:n].copy(),   # uint8 (n, 1, 72, 96)
            'lidar': self.lidar[:n].copy(),   # float32 (n, 360) in [0, 1]
            'proprio': self.proprio[:n].copy(),
            'actions': self.actions[:n].copy(),
            'rewards': self.rewards[:n].copy(),
            'dones': self.dones[:n].copy(),
            'is_first': self.is_first[:n].copy(),
            'is_intervention': self.is_intervention[:n].copy(),
            'is_demo': self.is_demo[:n].copy(),
            'schema_version': V3_SCHEMA_VERSION,
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


# ============================================================================
# Stuck Detector (unchanged from Dreamer)
# ============================================================================


class StuckDetector:
    def __init__(self, window_size=60, stuck_threshold=0.15):
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold
        self.position_history = deque(maxlen=window_size)
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_steps = 0

    def update(self, odom_msg):
        pos = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
        self.position_history.append(pos)
        if len(self.position_history) < self.position_history.maxlen:
            return False
        positions = np.array(self.position_history)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        is_stuck = total_distance < self.stuck_threshold
        if is_stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 30 and not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_steps = 45
        return is_stuck


# ============================================================================
# Dashboard
# ============================================================================


class DashboardStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'total_steps': 0,
            'model_version': 0,
            'update_count': 0,
            'buffer_fill': 0,
            'buffer_capacity': 0,
            'chunks_shipped': 0,
            'inference_backend': 'cpu',
            'rknn_converting': False,
            'zmq_connected': False,
            'server_status': 'unknown',
            'uptime_s': 0,
            'linear_vel': 0.0,
            'angular_vel': 0.0,
            'left_track': 0.0,
            'right_track': 0.0,
            'min_lidar_dist': 10.0,
            'reward': 0.0,
            'safety_blocked': False,
            'is_stuck': False,
            'intervention_active': False,
            'intervention_rate_recent': 0.0,
            'mode': 'teleop',
            'episode_count': 0,
            'episode_reward_avg': 0.0,
            'episode_reward_history': [],
            'actor_loss_history': [],
            'critic_loss_history': [],
            'q_history': [],
            'reward_history': [],
            'velocity_history': [],
            'steps_per_sec': 0.0,
        }
        self._start_time = time.time()
        self._step_times = deque(maxlen=100)
        self._intervention_window = deque(maxlen=900)  # ~30s @ 30Hz

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self._data:
                    self._data[k] = v

    def record_step(self, intervention: bool):
        now = time.time()
        with self._lock:
            self._step_times.append(now)
            self._intervention_window.append(1 if intervention else 0)
            if len(self._step_times) >= 2:
                dt = self._step_times[-1] - self._step_times[0]
                if dt > 0:
                    self._data['steps_per_sec'] = (len(self._step_times) - 1) / dt
            if self._intervention_window:
                self._data['intervention_rate_recent'] = (
                    sum(self._intervention_window) / len(self._intervention_window)
                )

    def append_reward(self, reward, velocity):
        with self._lock:
            self._data['reward_history'].append(reward)
            self._data['velocity_history'].append(velocity)
            if len(self._data['reward_history']) > 600:
                self._data['reward_history'] = self._data['reward_history'][-600:]
                self._data['velocity_history'] = self._data['velocity_history'][-600:]

    def append_training(self, actor_loss, critic_loss, q_mean):
        with self._lock:
            for key, val in [('actor_loss_history', actor_loss),
                             ('critic_loss_history', critic_loss),
                             ('q_history', q_mean)]:
                self._data[key].append(val)
                if len(self._data[key]) > 200:
                    self._data[key] = self._data[key][-200:]

    def get_json(self):
        with self._lock:
            self._data['uptime_s'] = time.time() - self._start_time
            return json.dumps(self._data)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>RLPD Remote Rover</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:12px}
h1{text-align:center;font-size:1.4em;margin-bottom:10px;color:#a78bfa}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-bottom:10px}
.card{background:#1a1d27;border-radius:8px;padding:14px;border:1px solid #2a2d3a}
.card h3{font-size:0.85em;color:#888;text-transform:uppercase;margin-bottom:8px}
.stat{font-size:2em;font-weight:700;color:#a78bfa}
.stat-row{display:flex;justify-content:space-between;margin:4px 0}
.stat-row .label{color:#888}.stat-row .val{color:#e0e0e0;font-weight:600}
.chart-container{position:relative;height:180px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:700px){.row2{grid-template-columns:1fr}}
.hil-on{color:#f87171;font-weight:700}
.hil-off{color:#888}
</style></head><body>
<h1>RLPD + HIL-SERL Remote Rover</h1>
<div class="grid">
  <div class="card"><h3>Status</h3>
    <div class="stat" id="model-ver">v0</div>
    <div class="stat-row"><span class="label">Mode</span><span class="val" id="mode">-</span></div>
    <div class="stat-row"><span class="label">Uptime</span><span class="val" id="uptime">0s</span></div>
    <div class="stat-row"><span class="label">Steps/s</span><span class="val" id="tps">0</span></div>
    <div class="stat-row"><span class="label">Backend</span><span class="val" id="backend">cpu</span></div>
    <div class="stat-row"><span class="label">ZMQ</span><span class="val" id="zmq">-</span></div>
    <div class="stat-row"><span class="label">Server</span><span class="val" id="server">-</span></div>
  </div>
  <div class="card"><h3>HIL Intervention</h3>
    <div class="stat" id="hil-state">OFF</div>
    <div class="stat-row"><span class="label">Recent rate</span><span class="val" id="hil-rate">0%</span></div>
    <div class="stat-row"><span class="label">Hold RB to override</span><span class="val">deadman</span></div>
  </div>
  <div class="card"><h3>Progress</h3>
    <div class="stat-row"><span class="label">Total Steps</span><span class="val" id="total-steps">0</span></div>
    <div class="stat-row"><span class="label">Chunks Shipped</span><span class="val" id="chunks">0</span></div>
    <div class="stat-row"><span class="label">Server Updates</span><span class="val" id="updates">0</span></div>
    <div class="stat-row"><span class="label">Buffer</span><span class="val" id="buffer">0/0</span></div>
  </div>
  <div class="card"><h3>Live Control</h3>
    <div class="stat-row"><span class="label">Velocity</span><span class="val" id="vel">0.00 m/s</span></div>
    <div class="stat-row"><span class="label">Tracks L/R</span><span class="val" id="tracks">0/0</span></div>
    <div class="stat-row"><span class="label">LiDAR Min</span><span class="val" id="lidar">0</span></div>
    <div class="stat-row"><span class="label">Reward</span><span class="val" id="reward">0</span></div>
    <div class="stat-row"><span class="label">Safety</span><span class="val" id="safety">OK</span></div>
  </div>
</div>
<div class="row2">
  <div class="card"><h3>Reward (Live)</h3><div class="chart-container"><canvas id="chart-live"></canvas></div></div>
  <div class="card"><h3>Q-mean / Actor / Critic (server)</h3><div class="chart-container"><canvas id="chart-train"></canvas></div></div>
</div>
<script>
const opts=(l)=>({responsive:true,maintainAspectRatio:false,animation:{duration:0},
  scales:{x:{display:false},y:{title:{display:true,text:l,color:'#888'},ticks:{color:'#888'},grid:{color:'#2a2d3a'}}},
  plugins:{legend:{labels:{color:'#ccc'}}}});
const live=new Chart(document.getElementById('chart-live'),{type:'line',data:{labels:[],datasets:[
  {label:'Reward',data:[],borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,tension:0.3}]},options:opts('Reward')});
const train=new Chart(document.getElementById('chart-train'),{type:'line',data:{labels:[],datasets:[
  {label:'Actor',data:[],borderColor:'#4ade80',borderWidth:2,pointRadius:1,tension:0.3},
  {label:'Critic',data:[],borderColor:'#f87171',borderWidth:2,pointRadius:1,tension:0.3},
  {label:'Q',data:[],borderColor:'#facc15',borderWidth:2,pointRadius:1,tension:0.3}]},options:opts('Value')});
function f(n,d=2){return Number(n).toFixed(d)}
function ft(s){s=Math.floor(s);if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m '+s%60+'s';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';}
async function poll(){
  try{const r=await fetch('/api/stats');const d=await r.json();
    document.getElementById('model-ver').textContent='v'+d.model_version;
    document.getElementById('mode').textContent=d.mode;
    document.getElementById('uptime').textContent=ft(d.uptime_s);
    document.getElementById('tps').textContent=f(d.steps_per_sec,1);
    document.getElementById('backend').textContent=d.rknn_converting?'Converting':d.inference_backend.toUpperCase();
    document.getElementById('zmq').textContent=d.zmq_connected?'OK':'DOWN';
    document.getElementById('server').textContent=d.server_status;
    const hilEl=document.getElementById('hil-state');
    hilEl.textContent=d.intervention_active?'ACTIVE':'off';
    hilEl.className=d.intervention_active?'stat hil-on':'stat hil-off';
    document.getElementById('hil-rate').textContent=(d.intervention_rate_recent*100).toFixed(1)+'%';
    document.getElementById('total-steps').textContent=d.total_steps.toLocaleString();
    document.getElementById('chunks').textContent=d.chunks_shipped;
    document.getElementById('updates').textContent=d.update_count;
    document.getElementById('buffer').textContent=d.buffer_fill+'/'+d.buffer_capacity;
    document.getElementById('vel').textContent=f(d.linear_vel)+' m/s';
    document.getElementById('tracks').textContent=f(d.left_track)+'/'+f(d.right_track);
    document.getElementById('lidar').textContent=f(d.min_lidar_dist)+' m';
    document.getElementById('reward').textContent=f(d.reward,3);
    document.getElementById('safety').textContent=d.safety_blocked?'BLOCKED':(d.is_stuck?'STUCK':'OK');
    live.data.labels=d.reward_history.map((_,i)=>i);live.data.datasets[0].data=d.reward_history;live.update();
    if(d.actor_loss_history.length){
      train.data.labels=d.actor_loss_history.map((_,i)=>i);
      train.data.datasets[0].data=d.actor_loss_history;
      train.data.datasets[1].data=d.critic_loss_history;
      train.data.datasets[2].data=d.q_history;
      train.update();
    }
  }catch(e){}
}
setInterval(poll,1000);poll();
</script></body></html>"""


def _make_dashboard_handler(stats):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/stats':
                body = stats.get_json().encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            else:
                body = DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass
    return H


def start_dashboard_server(stats, port=8081):
    server = HTTPServer(('0.0.0.0', port), _make_dashboard_handler(stats))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ============================================================================
# Main node
# ============================================================================


class RLPDRemoteRunner(Node):
    """RLPD runner: model-free SAC on the rover NPU with HIL-SERL interventions.

    v3 RKNN inputs (per tick, K=frame_stack):
        depth_stack   (1, K, 72, 96) float32 in [0, 1]
        lidar_stack   (1, K, 360)    float32 in [0, 1]
        proprio_stack (1, 6*K)       float32 (normalized)

    Output:
        mean_logstd (1, 4) = [mean_l, mean_r, log_std_l, log_std_r]
    """

    def __init__(self):
        super().__init__('rlpd_remote_runner')

        self.declare_parameter('server_addr', '192.168.1.100')
        self.declare_parameter('server_pull_port', 5555)
        self.declare_parameter('server_pub_port', 5556)
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('chunk_len', 64)
        self.declare_parameter('frame_stack', 4)
        self.declare_parameter('invert_linear_vel', False)
        self.declare_parameter('dashboard_port', 8081)
        self.declare_parameter('reward_weights_path', '')
        self.declare_parameter('teleop_mode', False)
        self.declare_parameter('teleop_cmd_topic', '/cmd_vel_teleop')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('stochastic_exec', False)

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.server_pull_port = int(self.get_parameter('server_pull_port').value)
        self.server_pub_port = int(self.get_parameter('server_pub_port').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.chunk_len = int(self.get_parameter('chunk_len').value)
        self.frame_stack = int(self.get_parameter('frame_stack').value)
        self.invert_linear_vel = bool(self.get_parameter('invert_linear_vel').value)
        self.dashboard_port = int(self.get_parameter('dashboard_port').value)
        self._reward_weights_path = str(self.get_parameter('reward_weights_path').value)
        self._teleop_mode = bool(self.get_parameter('teleop_mode').value)
        self._teleop_cmd_topic = str(self.get_parameter('teleop_cmd_topic').value)
        self._joy_topic = str(self.get_parameter('joy_topic').value)
        self._stochastic_exec = bool(self.get_parameter('stochastic_exec').value)

        self._latest_teleop = (0.0, 0.0)  # (linear.x m/s, angular.z rad/s)
        self._intervention_active = False  # RB held in autonomous mode

        # Reward channel clip values (per-channel bounds applied before
        # weighting on the rover; the server multiplies by w_* and sums).
        self._reward_clip = {
            'smoothness':   (-1.0, 0.0),
            'progress':     (0.0, 0.5),
            'turn_penalty': (-1.0, 0.0),
        }
        self._load_reward_clip()

        # Frame stackers — must match the server's RLPDVisionEncoderV3 layout.
        # Depth stored as uint8 (rover converts mm → [0,1] float → uint8*255);
        # lidar stored as float32 (already normalized to [0,1] by /range_max).
        self.depth_stacker = FrameStacker(k=self.frame_stack, shape=(1, DEPTH_H, DEPTH_W), dtype=np.uint8)
        # Lidar declared with a leading 1-dim so axis-0 concat in FrameStacker
        # yields (K, 360) — matches the RKNN graph's 3D input[1] (B, K, 360).
        self.lidar_stacker = FrameStacker(k=self.frame_stack, shape=(1, LIDAR_BEAMS), dtype=np.float32)
        self.pro_stacker = FrameStacker(k=self.frame_stack, shape=(PROPRIO_DIM,), dtype=np.float32)

        self.buffer = ChunkBuffer(chunk_len=self.chunk_len)

        self._model_version = 0
        self._update_count = 0
        self._total_steps = 0
        self._chunks_shipped = 0
        self._boot_time = time.time()

        # RKNN state
        self._rknn_runtime = None
        self._rknn_available = HAS_RKNN
        self._calibration_dir = Path("./calibration_data_rlpd")
        self._calibration_dir.mkdir(exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix='rlpd_remote_'))
        if self._rknn_available:
            self.get_logger().info('RKNNLite available - will use NPU for inference')
        else:
            self.get_logger().info('RKNNLite not available - random exploration only')

        # No persistent recurrent state — model-free SAC.
        self._next_is_first = True  # first transition after boot

        # Sensor state
        self._latest_depth_raw = None  # uint16 mm passthrough from D435i
        self._latest_scan = None       # LaserScan msg
        self._latest_odom = None
        self._latest_rf2o_odom = None
        self._latest_imu = None
        self._latest_wheel_vels = None
        self._safety_override = False
        self._velocity_confidence = 1.0
        self._latest_fused_yaw = 0.0

        self._curriculum_max_speed = self.max_linear
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20)

        self._current_episode_length = 0
        self._current_episode_reward = 0.0
        self._episode_reward_history = deque(maxlen=50)
        self.MAX_EPISODE_STEPS = 512

        # Recovery triggers (same as Dreamer — they no longer write reward / done)
        self.stuck_detector = StuckDetector(stuck_threshold=0.05)
        self._is_stuck = False
        self._post_reset_cooldown_steps = 30
        self._post_reset_cooldown = 0
        self._fwd_cmd_no_motion_count = 0
        self._slip_detected = False
        self._slip_recovery_active = False
        self._slip_backup_origin = None
        self._slip_backup_distance = 0.0
        self.SLIP_DETECTION_FRAMES = 15
        self.SLIP_CMD_THRESHOLD = 0.2
        self.SLIP_VEL_THRESHOLD = 0.03
        self.SLIP_BACKUP_LIMIT = 0.15
        self._wall_stop_steps = 0

        # Reward computation state. Style-learning reward needs no mapping
        # state — only collision latching. The proprio frontier_angle slot is
        # kept at 0.0 so the wire format / encoder layout stay unchanged
        # (encoder will learn to ignore the dead dim).
        self._frontier_angle_pi = 0.0
        self._emergency_prev = False
        self._collision_latched = False
        self._collision_first = False

        self._sensor_warmup_complete = False
        self._sensor_warmup_countdown = 90

        # ZMQ
        self._zmq_ctx = None
        self._sub_sock = None
        self._zmq_connected = False
        self._zmq_loop = None
        self._stop_event = threading.Event()

        self._push_ctx = zmq.Context()
        self._push_sock = self._push_ctx.socket(zmq.PUSH)
        self._push_sock.setsockopt(zmq.RECONNECT_IVL, 1000)
        self._push_sock.setsockopt(zmq.LINGER, 5000)
        self._push_sock.setsockopt(zmq.SNDHWM, 8)
        push_addr = f"tcp://{self.server_addr}:{self.server_pull_port}"
        self._push_sock.connect(push_addr)
        self.get_logger().info(f'ZMQ PUSH connected to {push_addr}')

        self.dashboard_stats = DashboardStats()
        self.dashboard_stats.update(mode='teleop' if self._teleop_mode else 'autonomous')
        self._dashboard_server = start_dashboard_server(self.dashboard_stats, self.dashboard_port)
        self._episode_count = 0

        self._zmq_thread = threading.Thread(target=self._run_zmq_loop, daemon=True)
        self._zmq_thread.start()

        self.bridge = CvBridge()
        self._setup_subscribers()
        self.track_cmd_pub = self.create_publisher(Float32MultiArray, 'track_cmd_ai', 10)

        # Teleop and joy subscriptions:
        #   - teleop mode: subscribe to /cmd_vel_teleop (RKNN disabled)
        #   - autonomous: subscribe to BOTH /joy (intervention detect) and
        #     /cmd_vel_teleop (override action source)
        self.create_subscription(Twist, self._teleop_cmd_topic, self._teleop_cmd_cb, 10)
        self.create_subscription(Joy, self._joy_topic, self._joy_cb, 10)

        if self._teleop_mode:
            self.get_logger().info(
                f'TELEOP COLLECTION MODE — actions from {self._teleop_cmd_topic}; '
                f'RKNN inference disabled. All chunks marked is_demo=True.'
            )
        else:
            self.get_logger().info(
                f'AUTONOMOUS MODE with HIL-SERL — hold RB on Xbox controller to '
                f'intervene. Teleop overrides policy when deadman is pressed.'
            )

        self.create_timer(1.0 / self.inference_rate, self._control_loop)
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        self.get_logger().info(
            f'RLPD Remote Runner initialized: chunk_len={self.chunk_len}, '
            f'frame_stack={self.frame_stack}, '
            f'server={self.server_addr}:{self.server_pull_port}/{self.server_pub_port}, '
            f'dashboard=http://0.0.0.0:{self.dashboard_port}'
        )

    # ========== ROS2 Setup ==========

    def _setup_subscribers(self):
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Odometry, '/odom_rf2o', self._rf2o_odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Bool, '/emergency_stop', self._safety_cb, 10)
        self.create_subscription(Float32, '/velocity_confidence', self._vel_conf_cb, 10)

    # ========== Sensor Callbacks ==========

    def _depth_cb(self, msg):
        self._latest_depth_raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def _scan_cb(self, msg):
        self._latest_scan = msg

    def _odom_cb(self, msg):
        self._latest_odom = (
            msg.pose.pose.position.x, msg.pose.pose.position.y,
            msg.twist.twist.linear.x, msg.twist.twist.angular.z
        )
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self._latest_fused_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._is_stuck = self.stuck_detector.update(msg)

    def _rf2o_odom_cb(self, msg):
        self._latest_rf2o_odom = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _imu_cb(self, msg):
        self._latest_imu = (
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        )

    def _joint_cb(self, msg):
        if len(msg.velocity) >= 4:
            self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])

    def _load_reward_clip(self):
        if not self._reward_weights_path or not HAS_YAML:
            return
        try:
            with open(self._reward_weights_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f'Could not read reward weights yaml: {e}')
            return
        clip = cfg.get('clip', {})
        for ch in ('smoothness', 'progress', 'turn_penalty'):
            if ch in clip and isinstance(clip[ch], (list, tuple)) and len(clip[ch]) == 2:
                self._reward_clip[ch] = (float(clip[ch][0]), float(clip[ch][1]))

    def _safety_cb(self, msg):
        state = bool(msg.data)
        if state and not self._emergency_prev:
            self._collision_first = True
        self._collision_latched = state
        self._safety_override = state
        self._emergency_prev = state

    def _teleop_cmd_cb(self, msg):
        self._latest_teleop = (float(msg.linear.x), float(msg.angular.z))

    def _joy_cb(self, msg):
        # HIL-SERL: RB (button 5) acts as the intervention switch. In teleop
        # mode this is the standard deadman; in autonomous mode it triggers
        # the policy override.
        try:
            if len(msg.buttons) > JOY_RB_BUTTON_IDX:
                self._intervention_active = bool(msg.buttons[JOY_RB_BUTTON_IDX])
        except Exception:
            pass

    def _vel_conf_cb(self, msg):
        self._velocity_confidence = msg.data

    # ========== LiDAR Processing ==========

    def _lidar_min_dist(self, scan_msg) -> float:
        if not scan_msg:
            return 3.0
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)
        if not np.any(valid):
            return 3.0
        return float(ranges[valid].min())

    def _scan_to_range_vector(self, scan_msg) -> np.ndarray:
        """LaserScan → fixed-length (360,) float32 range image in [0, 1].

        Invalid readings (inf / NaN / <=0 / >range_max) get clipped to
        `range_max` so the network sees a clean sentinel for "no obstacle in
        this beam". Resampled to LIDAR_BEAMS via linear interpolation against
        the scanner's native angle grid (STL19P delivers a variable count).
        """
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        range_max = float(scan_msg.range_max) if scan_msg.range_max > 0.0 else LIDAR_DEFAULT_MAX_RANGE
        # Replace invalid first, then clip.
        invalid = ~np.isfinite(ranges) | (ranges <= 0.0)
        ranges = np.where(invalid, range_max, ranges)
        ranges = np.clip(ranges, 0.0, range_max)

        n_native = ranges.shape[0]
        if n_native == LIDAR_BEAMS:
            resampled = ranges
        else:
            # Map each native beam to its angle, then sample LIDAR_BEAMS evenly
            # over the same angular span. `np.interp` handles the count change
            # without assuming the native count.
            native_angles = scan_msg.angle_min + np.arange(n_native, dtype=np.float32) * scan_msg.angle_increment
            target_angles = np.linspace(scan_msg.angle_min,
                                        scan_msg.angle_min + (n_native - 1) * scan_msg.angle_increment,
                                        LIDAR_BEAMS, dtype=np.float32)
            resampled = np.interp(target_angles, native_angles, ranges).astype(np.float32)
        return (resampled / range_max).astype(np.float32)

    # ========== Depth Processing ==========

    def _depth_to_input(self, depth_raw) -> np.ndarray:
        """uint16 mm depth → uint8 (1, 72, 96) in [0, 255].

        Pipeline: clip to [0.2, 6.0]m, replace invalid (0 / NaN / out-of-range)
        with the max-range value (255 after normalization — "no nearby obstacle"
        sentinel), resize to 96×72 with INTER_NEAREST (nearest preserves the
        invalid-edge geometry; bilinear would blend mm values from invalid
        pixels into their neighbors).
        """
        d = np.asarray(depth_raw)
        if d.dtype != np.uint16:
            d = d.astype(np.float32)
            invalid = ~np.isfinite(d) | (d <= 0) | (d > DEPTH_CLIP_MAX_MM)
            d = np.where(invalid, DEPTH_CLIP_MAX_MM, d).astype(np.uint16)
        else:
            invalid = (d == 0) | (d > DEPTH_CLIP_MAX_MM)
            d = np.where(invalid, DEPTH_CLIP_MAX_MM, d).astype(np.uint16)
        d = np.clip(d, DEPTH_CLIP_MIN_MM, DEPTH_CLIP_MAX_MM)
        d_norm = ((d.astype(np.float32) - DEPTH_CLIP_MIN_MM) / DEPTH_RANGE_MM)  # [0, 1]
        d_u8 = (d_norm * 255.0 + 0.5).astype(np.uint8)
        d_resized = cv2.resize(d_u8, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_NEAREST)
        return d_resized[None, :, :]  # (1, H, W)

    # ========== Teleop → tracks ==========

    def _teleop_to_tracks(self) -> np.ndarray:
        """Convert latest /cmd_vel_teleop twist to differential-drive [left, right].

        Lifted byte-for-byte from dreamer_remote_runner.py so demos and
        autonomous-mode actions live in the same coordinate frame.
        """
        lin, ang = self._latest_teleop
        max_lin = self.max_linear if self.max_linear > 0 else 1.0
        max_ang = self.max_angular if self.max_angular > 0 else 1.0
        norm_lin = max(-1.0, min(1.0, lin / max_lin))
        norm_ang = max(-1.0, min(1.0, ang / max_ang))
        left = max(-1.0, min(1.0, norm_lin - norm_ang))
        right = max(-1.0, min(1.0, norm_lin + norm_ang))
        return np.array([left, right], dtype=np.float32)

    # ========== Reward (multi-channel; 5 channels for RLPD) ==========

    def _reset_reward_state_on_episode(self):
        self._collision_latched = False
        self._collision_first = False

    def _compute_reward_channels(self, action_now: np.ndarray, prev_action: np.ndarray,
                                 linear_vel: float = 0.0, angular_vel: float = 0.0,
                                 intervention_now: bool = False):
        """Style-learning reward. Returns (reward_vec[5], collision).

        Channel order matches RLPD_REWARD_CHANNELS:
          0 smoothness   : -‖a_t - a_{t-1}‖² (action jerk penalty)
          1 progress     : max(0, linear_vel)  (forward motion bonus)
          2 turn_penalty : -|angular_vel|      (discourage spin-in-place)
          3 collision    : -1.0 per tick /emergency_stop is held
          4 intervention : -1.0 per tick the human held RB (HIL-SERL)
        """
        # Channel 0: action smoothness — quadratic in the per-track delta.
        delta = action_now.astype(np.float32) - prev_action.astype(np.float32)
        r_smoothness = -float(np.dot(delta, delta))
        lo, hi = self._reward_clip['smoothness']
        r_smoothness = float(np.clip(r_smoothness, lo, hi))

        # Channel 1: forward progress.
        r_progress = float(max(0.0, linear_vel))
        lo, hi = self._reward_clip['progress']
        r_progress = float(np.clip(r_progress, lo, hi))

        # Channel 2: turn-in-place penalty.
        r_turn = -float(abs(angular_vel))
        lo, hi = self._reward_clip['turn_penalty']
        r_turn = float(np.clip(r_turn, lo, hi))

        # Channel 3: collision (sticky -1 while /emergency_stop is asserted).
        penalty_active = self._collision_latched
        collision = self._collision_first
        self._collision_first = False
        r_collision = -1.0 if penalty_active else 0.0

        # Channel 4 (HIL-SERL): intervention penalty.
        r_intervention = -1.0 if intervention_now else 0.0

        reward_vec = np.array(
            [r_smoothness, r_progress, r_turn, r_collision, r_intervention],
            dtype=np.float32,
        )
        return reward_vec, collision

    # ========== Control Loop ==========

    def _control_loop(self):
        if self._latest_depth_raw is None or self._latest_scan is None:
            return

        if not self._sensor_warmup_complete:
            self._sensor_warmup_countdown -= 1
            if self._sensor_warmup_countdown <= 0:
                self._sensor_warmup_complete = True
                self.get_logger().info('Sensor warmup complete')
            return

        # v3 visual modalities: depth tile (1, 72, 96) uint8 + lidar (360,) float32.
        depth_u8 = self._depth_to_input(self._latest_depth_raw)
        lidar_vec = self._scan_to_range_vector(self._latest_scan)
        lidar_min = self._lidar_min_dist(self._latest_scan)

        current_linear = 0.0
        current_angular = 0.0
        if self._latest_rf2o_odom:
            current_linear, current_angular = self._latest_rf2o_odom
        elif self._latest_odom:
            current_linear, current_angular = self._latest_odom[2], self._latest_odom[3]
        if self.invert_linear_vel:
            current_linear = -current_linear

        # Slip detection
        cmd_fwd_prev = (self._prev_action[0] + self._prev_action[1]) / 2.0
        if cmd_fwd_prev > self.SLIP_CMD_THRESHOLD and abs(current_linear) < self.SLIP_VEL_THRESHOLD:
            self._fwd_cmd_no_motion_count += 1
            if self._fwd_cmd_no_motion_count >= self.SLIP_DETECTION_FRAMES and not self._slip_detected:
                self._slip_detected = True
                if self._latest_odom and not self._slip_recovery_active:
                    self._slip_recovery_active = True
                    self._slip_backup_origin = (self._latest_odom[0], self._latest_odom[1])
                    self._slip_backup_distance = 0.0
        else:
            if self._fwd_cmd_no_motion_count > 0:
                self._fwd_cmd_no_motion_count = 0
            if self._slip_detected:
                self._slip_detected = False
                self._slip_recovery_active = False
                self._slip_backup_origin = None
                self._slip_backup_distance = 0.0

        if self._slip_recovery_active and self._slip_backup_origin and self._latest_odom:
            x, y = self._latest_odom[0], self._latest_odom[1]
            ox, oy = self._slip_backup_origin
            self._slip_backup_distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if self._slip_backup_distance >= self.SLIP_BACKUP_LIMIT:
                self._slip_recovery_active = False

        # Proprio (6D, same layout as Dreamer)
        proprio_raw = np.array([
            lidar_min, self._prev_action[0], self._prev_action[1],
            current_linear, current_angular, self._frontier_angle_pi,
        ], dtype=np.float32)
        proprio = normalize_proprio(proprio_raw)

        # Episode boundary / model swap → reset stacks
        is_first = self._next_is_first
        if is_first:
            self.depth_stacker.reset()
            self.lidar_stacker.reset()
            self.pro_stacker.reset()
            self._reset_reward_state_on_episode()
        self._next_is_first = False

        # Update stacks with the just-observed step
        self.depth_stacker.push(depth_u8)
        self.lidar_stacker.push(lidar_vec[None, :])  # (360,) → (1, 360)
        self.pro_stacker.push(proprio)

        # Determine whether this tick is a human intervention. Pure teleop mode
        # is never an "intervention" in the HIL-SERL sense — those are
        # bootstrap demos (is_demo=True). In autonomous mode, holding RB makes
        # this tick an intervention.
        intervention_now = (not self._teleop_mode) and self._intervention_active

        # ---- Action selection ----
        if self._teleop_mode:
            action_np = self._teleop_to_tracks()
            policy_used = False
        elif intervention_now:
            # HIL-SERL: human override beats the policy. The teleop action is
            # what gets executed AND what's stored in the chunk's `actions[t]`,
            # so the demo replay sees the corrected behavior.
            action_np = self._teleop_to_tracks()
            policy_used = False
        elif self._rknn_runtime is not None:
            # v3 RKNN inputs in [0, 1] with batch dim prepended.
            depth_stack_in = (self.depth_stacker.get_stacked().astype(np.float32) / 255.0)[None, ...]
            lidar_stack_in = self.lidar_stacker.get_stacked()[None, ...]  # already float32 [0,1]
            pro_stack_in = self.pro_stacker.get_stacked()[None, ...]
            try:
                outputs = self._rknn_runtime.inference(
                    inputs=[depth_stack_in, lidar_stack_in, pro_stack_in]
                )
                if outputs is None or len(outputs) < 1:
                    raise RuntimeError(
                        f'RKNN returned {0 if outputs is None else len(outputs)} outputs, '
                        f'expected 1 [mean_logstd]. Rebuild .rknn from latest ONNX.'
                    )
                mean_logstd = outputs[0][0]  # (4,)
                action_mean = mean_logstd[:2]
                log_std = mean_logstd[2:4]
                if np.isnan(action_mean).any() or np.isinf(action_mean).any():
                    self.get_logger().error('RKNN output NaN/Inf, using zeros')
                    action_mean = np.zeros(2, dtype=np.float32)
                if self._stochastic_exec:
                    std = np.exp(np.clip(log_std, -5.0, 2.0))
                    noise = np.random.normal(0, 1, size=2) * std
                    action_np = np.tanh(action_mean + noise).astype(np.float32)
                else:
                    action_np = np.tanh(action_mean).astype(np.float32)
                policy_used = True
            except Exception as e:
                self.get_logger().error(f'RKNN inference error: {e}')
                action_np = np.array(
                    [np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 1.0)],
                    dtype=np.float32,
                )
                policy_used = False
        else:
            # No model yet — random exploration fills the online replay.
            action_np = np.array(
                [np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 1.0)],
                dtype=np.float32,
            )
            policy_used = False

        # Apply safety + deadzone (identical to Dreamer runner)
        is_stuck = self._is_stuck
        monitor_blocking = self._safety_override
        if monitor_blocking:
            self._wall_stop_steps += 1
        else:
            self._wall_stop_steps = 0

        def soft_deadzone(v, m):
            if abs(v) < 0.001:
                return 0.0
            return math.copysign(m + (1.0 - m) * abs(v), v)

        MIN_TRACK = 0.25
        if self._slip_detected and self._slip_recovery_active and not monitor_blocking:
            if not hasattr(self, '_slip_recovery_turn_dir'):
                self._slip_recovery_turn_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            rl = -0.4 + self._slip_recovery_turn_dir * 0.15
            rr = -0.4 - self._slip_recovery_turn_dir * 0.15
            left_track = math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(rl), rl)
            right_track = math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(rr), rr)
            actual_action = np.array([left_track, right_track], dtype=np.float32)
        elif monitor_blocking:
            left_track = min(soft_deadzone(action_np[0], MIN_TRACK), 0.0)
            right_track = min(soft_deadzone(action_np[1], MIN_TRACK), 0.0)
            actual_action = np.array([left_track, right_track], dtype=np.float32)
        else:
            left_track = soft_deadzone(action_np[0], MIN_TRACK)
            right_track = soft_deadzone(action_np[1], MIN_TRACK)
            actual_action = np.array([left_track, right_track], dtype=np.float32)
            if hasattr(self, '_slip_recovery_turn_dir'):
                delattr(self, '_slip_recovery_turn_dir')

        # Publish track command — BUT NOT in teleop mode. The hiwonder motor
        # driver subscribes to BOTH /cmd_vel AND /track_cmd; in teleop mode
        # the safety_monitor → /cmd_vel → motor pipeline is already driving
        # the rover from /cmd_vel_teleop. Publishing /track_cmd_ai here would
        # create a second motor command stream at slightly different scaling
        # and timing, causing yaw drift (the two pipelines fight each other).
        # We still build the chunk with `actual_action` so demos are
        # consistent across teleop and autonomous, but only the runner's
        # autonomous path actually drives the motor.
        if not self._teleop_mode:
            track_msg = Float32MultiArray()
            track_msg.data = [float(left_track), float(right_track)]
            self.track_cmd_pub.publish(track_msg)

        # ---- Reward (5-channel, style-learning) ----
        reward_vec, collision = self._compute_reward_channels(
            action_now=actual_action, prev_action=self._prev_action,
            linear_vel=current_linear, angular_vel=current_angular,
            intervention_now=intervention_now,
        )
        if np.isnan(reward_vec).any() or np.isinf(reward_vec).any():
            return

        # Rover-side scalar is a dashboard proxy only — the server recombines
        # channels with its own w_* weights for training.
        reward_scalar = float(reward_vec.sum())

        # ---- Episode boundary (only step cap; collision/intervention do not terminate) ----
        if self._post_reset_cooldown > 0:
            self._post_reset_cooldown -= 1
        episode_done = False
        done_reason = None
        if self._current_episode_length + 1 >= self.MAX_EPISODE_STEPS:
            episode_done = True
            done_reason = 'max_steps'
        if collision:
            self.get_logger().info(
                f'collision (no episode reset) | step={self._current_episode_length} | '
                f'ep_rew_so_far={self._current_episode_reward:.2f}'
            )

        # Store transition with is_first / is_intervention / is_demo flags.
        is_demo_flag = self._teleop_mode
        self.buffer.add(
            depth_u8=depth_u8, lidar=lidar_vec, proprio=proprio,
            action=actual_action, reward=reward_vec, done=episode_done,
            is_first=is_first,
            is_intervention=intervention_now, is_demo=is_demo_flag,
        )
        self._total_steps += 1
        self._current_episode_reward += reward_scalar
        self._current_episode_length += 1

        if episode_done:
            self._episode_reward_history.append(self._current_episode_reward)
            self._episode_count += 1
            self.get_logger().info(
                f'Episode boundary ({done_reason}) | len={self._current_episode_length} | '
                f'rew={self._current_episode_reward:.2f} | ep#{self._episode_count}'
            )
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
            self._next_is_first = True
            # Clear stuck detector history
            self.stuck_detector.position_history.clear()
            self.stuck_detector.stuck_counter = 0
            self.stuck_detector.recovery_mode = False
            self.stuck_detector.recovery_steps = 0
            self._wall_stop_steps = 0
            self._post_reset_cooldown = self._post_reset_cooldown_steps

        self._prev_action = actual_action
        self._prev_linear_cmds.append(actual_action[0])

        avg_ep_rew = float(np.mean(list(self._episode_reward_history))) if self._episode_reward_history else 0.0
        backend = 'npu' if self._rknn_runtime else 'cpu'
        self.dashboard_stats.update(
            total_steps=self._total_steps,
            model_version=self._model_version,
            update_count=self._update_count,
            buffer_fill=self.buffer.size,
            buffer_capacity=self.chunk_len,
            chunks_shipped=self._chunks_shipped,
            inference_backend=backend,
            zmq_connected=self._zmq_connected,
            linear_vel=float(current_linear),
            angular_vel=float(current_angular),
            left_track=float(left_track),
            right_track=float(right_track),
            min_lidar_dist=float(lidar_min),
            reward=reward_scalar,
            safety_blocked=bool(monitor_blocking),
            is_stuck=bool(is_stuck),
            intervention_active=bool(intervention_now),
            episode_count=self._episode_count,
            episode_reward_avg=avg_ep_rew,
            episode_reward_history=list(self._episode_reward_history),
        )
        self.dashboard_stats.append_reward(reward_scalar, float(current_linear))
        self.dashboard_stats.record_step(intervention=intervention_now)

        if self._total_steps % 300 == 0:
            self.get_logger().info(
                f'Step {self._total_steps} | AvgRew: {avg_ep_rew:.3f} | '
                f'v{self._model_version} | chunks={self._chunks_shipped} | '
                f'policy={"on" if policy_used else "off"} | '
                f'intervene={int(intervention_now)}'
            )

        # Calibration: save stacked v3 frames so RKNN INT8 quantization sees
        # the actual deployment input distribution. Depth saved as uint8
        # (converter will divide by 255 to match the ONNX [0, 1] expectation);
        # lidar saved as float32 in [0, 1]; proprio float32 already normalized.
        if np.random.rand() < 0.1:
            calib_files = list(self._calibration_dir.glob('*.npz'))
            if len(calib_files) < 100:
                timestamp = int(time.time() * 1000)
                np.savez_compressed(
                    self._calibration_dir / f"calib_{timestamp}.npz",
                    depth_stack=self.depth_stacker.get_stacked(),
                    lidar_stack=self.lidar_stacker.get_stacked(),
                    proprio_stack=self.pro_stacker.get_stacked(),
                )

        # Ship chunk when full
        if self.buffer.size >= self.chunk_len:
            self._ship_chunk()

    # ========== Chunk shipping ==========

    def _ship_chunk(self):
        if not self._zmq_connected:
            self.buffer.clear()
            return

        chunk = self.buffer.get_chunk()
        chunk['metadata'] = {
            'rover_id': 'rover-rlpd',
            'model_version': self._model_version,
            'total_steps': self._total_steps,
            'algorithm': 'rlpd',
            'mode': 'teleop' if self._teleop_mode else 'autonomous',
            'timestamp': time.time(),
        }
        self.buffer.clear()

        def _send():
            try:
                msg_bytes = serialize_batch(chunk)
                self._push_sock.send(msg_bytes, flags=zmq.NOBLOCK)
                self._chunks_shipped += 1
            except zmq.Again:
                self.get_logger().warn('ZMQ send queue full, dropping chunk')
            except Exception as e:
                self.get_logger().error(f'Chunk shipping failed: {e}')

        threading.Thread(target=_send, daemon=True).start()

    # ========== ZMQ SUB loop ==========

    def _run_zmq_loop(self):
        asyncio.run(self._zmq_main())

    async def _zmq_main(self):
        try:
            self._zmq_loop = asyncio.get_running_loop()
            self._zmq_ctx = zmq.asyncio.Context()
            self._sub_sock = self._zmq_ctx.socket(zmq.SUB)
            self._sub_sock.setsockopt(zmq.RECONNECT_IVL, 1000)
            self._sub_sock.subscribe(b"model")
            self._sub_sock.subscribe(b"status")
            pub_addr = f"tcp://{self.server_addr}:{self.server_pub_port}"
            self._sub_sock.connect(pub_addr)
            self.get_logger().info(f'ZMQ SUB connected to {pub_addr}')
            self._zmq_connected = True
            self.dashboard_stats.update(zmq_connected=True)

            while not self._stop_event.is_set():
                try:
                    if await self._sub_sock.poll(timeout=1000):
                        frames = await self._sub_sock.recv_multipart()
                        if len(frames) >= 2:
                            topic, data = frames[0], frames[1]
                            if topic == b"model":
                                await self._handle_model_message(data)
                            elif topic == b"status":
                                self._handle_status_message(data)
                except Exception as e:
                    self.get_logger().error(f'ZMQ recv error: {e}')
                    await asyncio.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f'ZMQ loop error: {e}')
        finally:
            self._zmq_connected = False
            self.dashboard_stats.update(zmq_connected=False)
            if self._zmq_ctx:
                self._zmq_ctx.destroy(linger=0)

    async def _handle_model_message(self, data):
        if self._teleop_mode:
            return
        try:
            model_msg = msgpack.unpackb(data)
            model_version = model_msg["version"]
            onnx_bytes = model_msg["onnx_bytes"]
            rknn_bytes = model_msg.get("rknn_bytes")

            if model_version <= self._model_version:
                return

            self.get_logger().info(f'Received RLPD model v{model_version}')

            train_stats = model_msg.get("train_stats", {})
            if train_stats:
                self.dashboard_stats.append_training(
                    train_stats.get('actor_loss', 0),
                    train_stats.get('critic_loss', 0),
                    train_stats.get('q_mean', 0),
                )

            if rknn_bytes and HAS_RKNN:
                rknn_path = self._temp_dir / "latest_model.rknn"
                with open(rknn_path, 'wb') as f:
                    f.write(rknn_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                if self._load_rknn_model(str(rknn_path)):
                    self._model_version = model_version
                    self._update_count = model_version
                    # Force frame-stack reset on first tick after model swap
                    self._next_is_first = True
                    self.get_logger().info(f'RKNN RLPD v{model_version} loaded (server-converted)')
            elif HAS_RKNN:
                onnx_path = self._temp_dir / "latest_model.onnx"
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                self.dashboard_stats.update(rknn_converting=True)
                rknn_path = str(onnx_path).replace('.onnx', '.rknn')
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, self._convert_and_load_rknn, str(onnx_path), rknn_path, model_version
                )
                self.dashboard_stats.update(rknn_converting=False)
            else:
                self._model_version = model_version
                self._update_count = model_version
        except Exception as e:
            self.get_logger().error(f'Model handling failed: {e}')
            self.dashboard_stats.update(rknn_converting=False)

    def _handle_status_message(self, data):
        try:
            status = deserialize_status(data)
            self.dashboard_stats.update(
                server_status=status.get("status", "unknown"),
                update_count=status.get("model_version", self._update_count),
            )
        except Exception:
            pass

    # ========== RKNN management ==========

    def _convert_and_load_rknn(self, onnx_path: str, rknn_path: str, model_version: int):
        try:
            convert_script = "./convert_onnx_to_rknn.sh"
            if not os.path.exists(convert_script):
                convert_script = "/home/benson/ros2-rover/convert_onnx_to_rknn.sh"
            if not os.path.exists(convert_script):
                self.get_logger().warn('convert_onnx_to_rknn.sh not found')
                return
            result = subprocess.run(
                [convert_script, onnx_path, str(self._calibration_dir)],
                capture_output=True, text=True, timeout=300
            )
            if os.path.exists(rknn_path):
                if self._load_rknn_model(rknn_path):
                    self._model_version = model_version
                    self._update_count = model_version
                    self._next_is_first = True
                    self.get_logger().info(f'RKNN RLPD v{model_version} loaded')
            else:
                self.get_logger().error(f'RKNN conversion failed')
                self.get_logger().error(f'STDERR: {result.stderr[-500:]}')
        except Exception as e:
            self.get_logger().error(f'RKNN conversion error: {e}')

    def _load_rknn_model(self, rknn_path: str) -> bool:
        if not HAS_RKNN:
            return False
        try:
            new_runtime = RKNNLite()
            if new_runtime.load_rknn(rknn_path) != 0:
                return False
            if new_runtime.init_runtime() != 0:
                return False
            self._rknn_runtime = new_runtime
            self.get_logger().info(f'RKNN loaded from {rknn_path}')
            return True
        except Exception as e:
            self.get_logger().error(f'RKNN load error: {e}')
            return False

    def destroy_node(self):
        self._stop_event.set()
        if self._zmq_thread.is_alive():
            self._zmq_thread.join(timeout=2.0)
        if self._push_sock:
            self._push_sock.close(linger=5000)
        if self._push_ctx:
            self._push_ctx.term()
        if self._dashboard_server:
            self._dashboard_server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RLPDRemoteRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
