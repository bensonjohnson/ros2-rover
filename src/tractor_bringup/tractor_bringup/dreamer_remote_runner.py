#!/usr/bin/env python3
"""DreamerV3 Remote Runner — Collects chunks on rover, trains on remote GPU server.

Fork of ppo_remote_runner.py adapted for DreamerV3:
- Off-policy: rollouts are appended to the server's replay buffer; no wait-for-model.
- Short chunks (default 64 steps) shipped frequently instead of 2048-step rollouts.
- Persistent RSSM state (h, z) maintained between ticks; reset on episode boundaries.
- RKNN inference inputs/outputs extended to carry RSSM state through the NPU.
- Rollout schema adds `is_first` flag per step so the server can re-drive RSSM.

Architecture:
- Unified BEV (LiDAR + Depth) -> 2x128x128 grid
- 6-dim proprioception, optional RGB 3x84x84
- RKNN NPU inference (30Hz), random actions until first model arrives
- Direct track control: [left_speed, right_speed] in [-1, 1]
- ZeroMQ PUSH/PULL for chunks, PUB/SUB for model updates
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

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, Float32MultiArray
from std_srvs.srv import Trigger

from tractor_bringup.occupancy_processor import UnifiedBEVProcessor
from tractor_bringup.coverage_tracker import CoverageTracker
from tractor_bringup.episodic_novelty import EpisodicNovelty

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# RSSM dimensions — must match remote_training_server/model_architectures.py
RSSM_HIDDEN_DIM = 512
RSSM_Z_GROUPS = 32
RSSM_Z_CLASSES = 32
RSSM_Z_DIM = RSSM_Z_GROUPS * RSSM_Z_CLASSES  # 1024


def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    return np.clip(normalized, -3.0, 3.0).astype(np.float32)


# ============================================================================
# Chunk Buffer (collects a short sequence, ships to server replay)
# ============================================================================

class ChunkBuffer:
    """Short-horizon buffer for Dreamer. Ships every `chunk_len` steps.

    Rewards are multi-channel: (T, K). Channel order:
        0 = coverage, 1 = frontier, 2 = collision, 3 = episodic novelty.
    P2E intrinsic and lifetime modulator are server-side only.
    """

    def __init__(self, chunk_len: int, proprio_dim: int = 6, reward_dim: int = 4):
        self.chunk_len = chunk_len
        self.capacity = chunk_len + 16  # small slack
        self.reward_dim = reward_dim
        self.ptr = 0
        self.size = 0
        self.bev = np.zeros((self.capacity, 2, 128, 128), dtype=np.uint8)
        self.rgb = np.zeros((self.capacity, 3, 84, 84), dtype=np.uint8)
        self.proprio = np.zeros((self.capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, reward_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.is_first = np.zeros((self.capacity,), dtype=np.bool_)

    def add(self, bev, proprio, action, reward, done, is_first, rgb=None):
        i = self.ptr
        self.bev[i] = (bev * 255.0).astype(np.uint8)
        if rgb is not None:
            self.rgb[i] = rgb
        else:
            self.rgb[i] = 0
        self.proprio[i] = proprio
        self.actions[i] = action
        r = np.asarray(reward, dtype=np.float32)
        if r.ndim == 0:
            r = np.broadcast_to(r, (self.reward_dim,))
        self.rewards[i] = r
        self.dones[i] = done
        self.is_first[i] = is_first
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_chunk(self):
        n = self.size
        return {
            'bev': self.bev[:n].astype(np.float32) / 255.0,
            'rgb': self.rgb[:n].astype(np.float32) / 255.0,
            'proprio': self.proprio[:n].copy(),
            'actions': self.actions[:n].copy(),
            'rewards': self.rewards[:n].copy(),
            'dones': self.dones[:n].copy(),
            'is_first': self.is_first[:n].copy(),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


# ============================================================================
# Stuck Detector (unchanged from PPO)
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
            'episode_count': 0,
            'episode_reward_avg': 0.0,
            'episode_reward_history': [],
            'wm_loss_history': [],
            'actor_loss_history': [],
            'critic_loss_history': [],
            'reward_history': [],
            'velocity_history': [],
            'steps_per_sec': 0.0,
        }
        self._start_time = time.time()
        self._step_times = deque(maxlen=100)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self._data:
                    self._data[k] = v

    def record_step(self):
        now = time.time()
        with self._lock:
            self._step_times.append(now)
            if len(self._step_times) >= 2:
                dt = self._step_times[-1] - self._step_times[0]
                if dt > 0:
                    self._data['steps_per_sec'] = (len(self._step_times) - 1) / dt

    def append_reward(self, reward, velocity):
        with self._lock:
            self._data['reward_history'].append(reward)
            self._data['velocity_history'].append(velocity)
            if len(self._data['reward_history']) > 600:
                self._data['reward_history'] = self._data['reward_history'][-600:]
                self._data['velocity_history'] = self._data['velocity_history'][-600:]

    def append_training(self, wm_loss, actor_loss, critic_loss):
        with self._lock:
            for key, val in [('wm_loss_history', wm_loss),
                             ('actor_loss_history', actor_loss),
                             ('critic_loss_history', critic_loss)]:
                self._data[key].append(val)
                if len(self._data[key]) > 200:
                    self._data[key] = self._data[key][-200:]

    def get_json(self):
        with self._lock:
            self._data['uptime_s'] = time.time() - self._start_time
            return json.dumps(self._data)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Dreamer Remote Rover</title>
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
</style></head><body>
<h1>Dreamer Remote Rover Dashboard</h1>
<div class="grid">
  <div class="card"><h3>Status</h3>
    <div class="stat" id="model-ver">v0</div>
    <div class="stat-row"><span class="label">Uptime</span><span class="val" id="uptime">0s</span></div>
    <div class="stat-row"><span class="label">Steps/s</span><span class="val" id="tps">0</span></div>
    <div class="stat-row"><span class="label">Backend</span><span class="val" id="backend">cpu</span></div>
    <div class="stat-row"><span class="label">ZMQ</span><span class="val" id="zmq">-</span></div>
    <div class="stat-row"><span class="label">Server</span><span class="val" id="server">-</span></div>
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
  <div class="card"><h3>Episodes</h3>
    <div class="stat" id="ep-count">0</div>
    <div class="stat-row"><span class="label">Avg Reward</span><span class="val" id="ep-avg">0</span></div>
  </div>
</div>
<div class="row2">
  <div class="card"><h3>Reward (Live)</h3><div class="chart-container"><canvas id="chart-live"></canvas></div></div>
  <div class="card"><h3>World-Model Loss</h3><div class="chart-container"><canvas id="chart-wm"></canvas></div></div>
</div>
<div class="row2">
  <div class="card"><h3>Actor Loss</h3><div class="chart-container"><canvas id="chart-actor"></canvas></div></div>
  <div class="card"><h3>Critic Loss</h3><div class="chart-container"><canvas id="chart-critic"></canvas></div></div>
</div>
<script>
const opts=(l)=>({responsive:true,maintainAspectRatio:false,animation:{duration:0},
  scales:{x:{display:false},y:{title:{display:true,text:l,color:'#888'},ticks:{color:'#888'},grid:{color:'#2a2d3a'}}},
  plugins:{legend:{labels:{color:'#ccc'}}}});
const live=new Chart(document.getElementById('chart-live'),{type:'line',data:{labels:[],datasets:[
  {label:'Reward',data:[],borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,tension:0.3}]},options:opts('Reward')});
const wm=new Chart(document.getElementById('chart-wm'),{type:'line',data:{labels:[],datasets:[
  {label:'WM Loss',data:[],borderColor:'#a78bfa',borderWidth:2,pointRadius:1,tension:0.3}]},options:opts('Loss')});
const ac=new Chart(document.getElementById('chart-actor'),{type:'line',data:{labels:[],datasets:[
  {label:'Actor',data:[],borderColor:'#4ade80',borderWidth:2,pointRadius:1,tension:0.3}]},options:opts('Loss')});
const cr=new Chart(document.getElementById('chart-critic'),{type:'line',data:{labels:[],datasets:[
  {label:'Critic',data:[],borderColor:'#f87171',borderWidth:2,pointRadius:1,tension:0.3}]},options:opts('Loss')});
function f(n,d=2){return Number(n).toFixed(d)}
function ft(s){s=Math.floor(s);if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m '+s%60+'s';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';}
async function poll(){
  try{const r=await fetch('/api/stats');const d=await r.json();
    document.getElementById('model-ver').textContent='v'+d.model_version;
    document.getElementById('uptime').textContent=ft(d.uptime_s);
    document.getElementById('tps').textContent=f(d.steps_per_sec,1);
    document.getElementById('backend').textContent=d.rknn_converting?'Converting':d.inference_backend.toUpperCase();
    document.getElementById('zmq').textContent=d.zmq_connected?'OK':'DOWN';
    document.getElementById('server').textContent=d.server_status;
    document.getElementById('total-steps').textContent=d.total_steps.toLocaleString();
    document.getElementById('chunks').textContent=d.chunks_shipped;
    document.getElementById('updates').textContent=d.update_count;
    document.getElementById('buffer').textContent=d.buffer_fill+'/'+d.buffer_capacity;
    document.getElementById('vel').textContent=f(d.linear_vel)+' m/s';
    document.getElementById('tracks').textContent=f(d.left_track)+'/'+f(d.right_track);
    document.getElementById('lidar').textContent=f(d.min_lidar_dist)+' m';
    document.getElementById('reward').textContent=f(d.reward,3);
    document.getElementById('safety').textContent=d.safety_blocked?'BLOCKED':(d.is_stuck?'STUCK':'OK');
    document.getElementById('ep-count').textContent=d.episode_count;
    document.getElementById('ep-avg').textContent=f(d.episode_reward_avg,3);
    live.data.labels=d.reward_history.map((_,i)=>i);live.data.datasets[0].data=d.reward_history;live.update();
    if(d.wm_loss_history.length){wm.data.labels=d.wm_loss_history.map((_,i)=>i);wm.data.datasets[0].data=d.wm_loss_history;wm.update();}
    if(d.actor_loss_history.length){ac.data.labels=d.actor_loss_history.map((_,i)=>i);ac.data.datasets[0].data=d.actor_loss_history;ac.update();}
    if(d.critic_loss_history.length){cr.data.labels=d.critic_loss_history.map((_,i)=>i);cr.data.datasets[0].data=d.critic_loss_history;cr.update();}
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


def start_dashboard_server(stats, port=8080):
    server = HTTPServer(('0.0.0.0', port), _make_dashboard_handler(stats))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ============================================================================
# Main node
# ============================================================================

class DreamerRemoteRunner(Node):
    """DreamerV3 runner: RKNN inference on rover with persistent RSSM state,
    short chunks shipped to remote GPU server via ZMQ for off-policy training."""

    def __init__(self):
        super().__init__('dreamer_remote_runner')

        self.declare_parameter('server_addr', '192.168.1.100')
        self.declare_parameter('server_pull_port', 5555)
        self.declare_parameter('server_pub_port', 5556)
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('chunk_len', 64)
        self.declare_parameter('invert_linear_vel', False)
        self.declare_parameter('dashboard_port', 8080)
        self.declare_parameter('reward_weights_path', '')

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.server_pull_port = int(self.get_parameter('server_pull_port').value)
        self.server_pub_port = int(self.get_parameter('server_pub_port').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.chunk_len = int(self.get_parameter('chunk_len').value)
        self.invert_linear_vel = bool(self.get_parameter('invert_linear_vel').value)
        self.dashboard_port = int(self.get_parameter('dashboard_port').value)
        self._reward_weights_path = str(self.get_parameter('reward_weights_path').value)

        # Reward channel order is load-bearing: must match REWARD_CHANNELS in
        # remote_training_server/model_architectures.py.
        self.REWARD_CHANNEL_ORDER = ('coverage', 'frontier', 'collision', 'episodic')
        self._reward_clip = {
            'coverage': (0.0, 0.5),
            'frontier': (-0.1, 0.1),
            'episodic': (0.0, 1.0),
        }
        self._load_reward_clip()

        self.buffer = ChunkBuffer(
            chunk_len=self.chunk_len, proprio_dim=6,
            reward_dim=len(self.REWARD_CHANNEL_ORDER),
        )

        self._model_version = 0
        self._update_count = 0
        self._total_steps = 0
        self._chunks_shipped = 0
        self._boot_time = time.time()

        # RKNN state
        self._rknn_runtime = None
        self._rknn_available = HAS_RKNN
        self._calibration_dir = Path("./calibration_data_dreamer")
        self._calibration_dir.mkdir(exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix='dreamer_remote_'))
        if self._rknn_available:
            self.get_logger().info('RKNNLite available - will use NPU for inference')
        else:
            self.get_logger().info('RKNNLite not available - random exploration only')

        # Persistent RSSM state
        self._prev_h = np.zeros((1, RSSM_HIDDEN_DIM), dtype=np.float32)
        self._prev_z = np.zeros((1, RSSM_Z_DIM), dtype=np.float32)
        self._prev_a = np.zeros((1, 2), dtype=np.float32)
        self._next_is_first = True  # first transition after boot

        # Sensor state
        self._latest_rgb = None
        self._latest_depth_raw = None
        self._latest_scan = None
        self._latest_bev = None
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

        # Stuck/slip detectors remain alive — but only as recovery triggers.
        # They no longer write to reward or set `done`. Preserving persistent
        # RSSM state across these events lets the world model learn escapes.
        self.stuck_detector = StuckDetector(stuck_threshold=0.05)
        self._is_stuck = False
        # Post-reset cooldown: suppress termination + clear stale stuck history
        # right after an episode boundary so we don't spin-loop on persistently
        # blocked physical states.
        self._post_reset_cooldown_steps = 30
        self._post_reset_cooldown = 0
        self._recovery_steps_remaining = 0
        self._recovery_turn_dir = 0.0
        self._fwd_cmd_no_motion_count = 0
        self._slip_detected = False
        self._slip_recovery_active = False
        self._slip_backup_origin = None
        self._slip_backup_distance = 0.0
        self.SLIP_DETECTION_FRAMES = 15
        self.SLIP_CMD_THRESHOLD = 0.2
        self.SLIP_VEL_THRESHOLD = 0.03
        self.SLIP_BACKUP_LIMIT = 0.15

        # Consecutive ticks the safety stop has been asserted. Drives the
        # scripted reverse-and-turn recovery; *not* an episode-termination
        # signal. (The old code used this to set done=True after 450 ticks,
        # which wiped RSSM state mid-recovery and prevented learning escapes.)
        self._wall_stop_steps = 0

        # --- Reward system (replaces the old phase-gated shaping) ---
        # Channels:
        #   coverage  : driven by newly-known cells this tick (CoverageTracker)
        #   frontier  : potential-based shaping  Φ(s') - Φ(s)  with Φ = -frontier_distance
        #   collision : -10 on /emergency_stop rising edge (single terminal event)
        #   episodic  : k-NN pseudo-count over the RSSM feat (h ⊕ z)
        self.coverage_tracker = CoverageTracker(
            grid_size=200, resolution=0.05, max_range=4.0,
            coverage_alpha=0.001,
        )
        self.episodic_novelty = EpisodicNovelty(
            embed_dim=RSSM_HIDDEN_DIM + RSSM_Z_DIM,
            buffer_size=256, k=10,
        )
        self._prev_phi: float | None = None
        self._frontier_angle_pi = 0.0      # last frontier bearing / π ∈ [-1, 1]
        self._emergency_prev = False       # rising-edge tracker for /emergency_stop
        self._collision_latched = False    # true for one tick after rising edge

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

        self.occupancy_processor = UnifiedBEVProcessor(grid_size=128, max_range=4.0)

        self.dashboard_stats = DashboardStats()
        self._dashboard_server = start_dashboard_server(self.dashboard_stats, self.dashboard_port)
        self._episode_count = 0

        self._zmq_thread = threading.Thread(target=self._run_zmq_loop, daemon=True)
        self._zmq_thread.start()

        self.bridge = CvBridge()
        self._setup_subscribers()
        self.track_cmd_pub = self.create_publisher(Float32MultiArray, 'track_cmd_ai', 10)

        self.create_timer(1.0 / self.inference_rate, self._control_loop)
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        self.get_logger().info(
            f'Dreamer Remote Runner initialized: chunk_len={self.chunk_len}, '
            f'server={self.server_addr}:{self.server_pull_port}/{self.server_pub_port}, '
            f'dashboard=http://0.0.0.0:{self.dashboard_port}'
        )

    # ========== ROS2 Setup ==========

    def _setup_subscribers(self):
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._rgb_cb, qos_profile_sensor_data)
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

    def _rgb_cb(self, msg):
        rgb_raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb_resized = cv2.resize(rgb_raw, (84, 84), interpolation=cv2.INTER_AREA)
        self._latest_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)

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
        """Populate `self._reward_clip` from `reward_weights.yaml`, if provided."""
        if not self._reward_weights_path or not HAS_YAML:
            return
        try:
            with open(self._reward_weights_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f'Could not read reward weights yaml: {e}')
            return
        clip = cfg.get('clip', {})
        for ch in ('coverage', 'frontier', 'episodic'):
            if ch in clip and isinstance(clip[ch], (list, tuple)) and len(clip[ch]) == 2:
                self._reward_clip[ch] = (float(clip[ch][0]), float(clip[ch][1]))

    def _safety_cb(self, msg):
        # Rising-edge of the safety-stop signal is the ONLY collision
        # signal that enters the reward / episode boundary. Staying stopped
        # does not keep emitting a penalty.
        state = bool(msg.data)
        if state and not self._emergency_prev:
            self._collision_latched = True
        self._safety_override = state
        self._emergency_prev = state

    def _vel_conf_cb(self, msg):
        self._velocity_confidence = msg.data

    # ========== LiDAR Processing ==========

    def _lidar_min_dist(self, scan_msg) -> float:
        """Closest valid LiDAR return, used for the safety / recovery heuristics
        and as a proprio channel. Heading hints from gap-finding are gone — the
        world model + frontier shaping now drive direction selection."""
        if not scan_msg:
            return 3.0
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)
        if not np.any(valid):
            return 3.0
        return float(ranges[valid].min())

    # ========== Reward (multi-channel; see config/reward_weights.yaml) ==========

    def _reset_reward_state_on_episode(self):
        """Clear per-episode novelty memory + potential baseline on is_first."""
        self.episodic_novelty.reset()
        self._prev_phi = None
        self._collision_latched = False

    def _compute_reward_channels(self, robot_x, robot_y, robot_yaw, scan_msg,
                                 feat_embed: np.ndarray):
        """Assemble the 4-channel reward vector. Called once per control tick.

        Returns:
            reward_vec: np.ndarray shape (4,) — [coverage, frontier, collision, episodic]
            coverage_step: CoverageStep (used to fill proprio `frontier_angle / π`)
            collision: bool (drives episode termination)
        """
        cov_step = self.coverage_tracker.step(
            ranges=np.asarray(scan_msg.ranges, dtype=np.float32),
            angle_min=scan_msg.angle_min,
            angle_increment=scan_msg.angle_increment,
            robot_x=robot_x, robot_y=robot_y, robot_yaw=robot_yaw,
        )

        # Channel 0: coverage (already clipped inside CoverageTracker)
        lo, hi = self._reward_clip['coverage']
        r_coverage = float(np.clip(cov_step.coverage_delta, lo, hi))

        # Channel 1: frontier potential-based shaping (γ Φ(s') − Φ(s) with γ=1 here;
        # the policy's γ=0.997 is applied server-side in λ-returns already)
        if cov_step.has_frontier:
            phi = cov_step.phi
            if self._prev_phi is None:
                r_frontier = 0.0
            else:
                r_frontier = phi - self._prev_phi
            self._prev_phi = phi
        else:
            r_frontier = 0.0
            self._prev_phi = None
        lo, hi = self._reward_clip['frontier']
        r_frontier = float(np.clip(r_frontier, lo, hi))

        # Channel 2: collision (rising edge of /emergency_stop)
        collision = self._collision_latched
        r_collision = -10.0 if collision else 0.0
        self._collision_latched = False

        # Channel 3: episodic novelty over RSSM feat (h ⊕ z). Feat is what the
        # server-side SimHash also buckets on, so the two timescales are aligned.
        lo, hi = self._reward_clip['episodic']
        r_episodic = float(np.clip(self.episodic_novelty.step(feat_embed), lo, hi))

        # Remember the frontier bearing for proprio (rover-observable steering hint).
        if cov_step.has_frontier:
            self._frontier_angle_pi = float(np.clip(cov_step.frontier_angle / math.pi, -1.0, 1.0))
        else:
            self._frontier_angle_pi = 0.0

        return np.array([r_coverage, r_frontier, r_collision, r_episodic], dtype=np.float32), \
               cov_step, collision


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

        bev_grid = self.occupancy_processor.process(
            depth_img=self._latest_depth_raw, laser_scan=self._latest_scan
        )
        self._latest_bev = bev_grid

        lidar_min = self._lidar_min_dist(self._latest_scan)

        current_linear = 0.0
        current_angular = 0.0
        if self._latest_rf2o_odom:
            current_linear, current_angular = self._latest_rf2o_odom
        elif self._latest_odom:
            current_linear, current_angular = self._latest_odom[2], self._latest_odom[3]
        if self.invert_linear_vel:
            current_linear = -current_linear

        # Slip detection — kept as a *recovery* trigger only. Does not write
        # to reward and does not set `done` (those would wipe RSSM state and
        # block the world model from learning recovery dynamics).
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

        # Proprio. The 6th channel is the **frontier bearing** (in [-1, 1] =
        # angle / π) computed by CoverageTracker — a rover-observable steering
        # hint toward the closest unexplored cell. Replaces the old BEV-derived
        # gap heading; gap-finding is now subsumed by the world model + the
        # frontier potential reward.
        proprio_raw = np.array([
            lidar_min, self._prev_action[0], self._prev_action[1],
            current_linear, current_angular, self._frontier_angle_pi,
        ], dtype=np.float32)
        proprio = normalize_proprio(proprio_raw)

        if self._latest_rgb is not None:
            rgb_chw_uint8 = np.transpose(self._latest_rgb, (2, 0, 1))
            rgb_float = rgb_chw_uint8.astype(np.float32) / 255.0
            rgb_input = ((rgb_float - RGB_MEAN.reshape(3, 1, 1)) / RGB_STD.reshape(3, 1, 1))[None, ...]
        else:
            rgb_chw_uint8 = np.zeros((3, 84, 84), dtype=np.uint8)
            rgb_input = np.zeros((1, 3, 84, 84), dtype=np.float32)

        # Reset RSSM state on first tick after episode boundary
        is_first = self._next_is_first
        if is_first:
            self._prev_h = np.zeros((1, RSSM_HIDDEN_DIM), dtype=np.float32)
            self._prev_z = np.zeros((1, RSSM_Z_DIM), dtype=np.float32)
            self._prev_a = np.zeros((1, 2), dtype=np.float32)
        self._next_is_first = False

        # Inference
        if self._rknn_runtime is not None:
            bev_input = bev_grid[None, ...].astype(np.float32)
            proprio_input = proprio[None, ...]
            try:
                outputs = self._rknn_runtime.inference(inputs=[
                    bev_input, proprio_input, rgb_input,
                    self._prev_h, self._prev_z, self._prev_a,
                ])
                if outputs is None or len(outputs) < 3:
                    n = 0 if outputs is None else len(outputs)
                    shapes = [getattr(o, 'shape', None) for o in (outputs or [])]
                    raise RuntimeError(
                        f'RKNN returned {n} outputs (shapes={shapes}), expected 3 '
                        '[mean_logstd, new_h, new_z]. Rebuild .rknn from latest ONNX.'
                    )
                mean_logstd = outputs[0][0]  # (2*action_dim,)
                action_mean = mean_logstd[:2]
                log_std = mean_logstd[2:4]
                new_h = outputs[1]
                new_z = outputs[2]

                if np.isnan(action_mean).any() or np.isinf(action_mean).any():
                    self.get_logger().error('RKNN output NaN/Inf, using zeros')
                    action_mean = np.zeros(2)
                    new_h = np.zeros_like(self._prev_h)
                    new_z = np.zeros_like(self._prev_z)

                std = np.exp(np.clip(log_std, -14, 0.7))
                noise = np.random.normal(0, 1, size=2) * std
                action_np = np.tanh(action_mean + noise).astype(np.float32)

                self._prev_h = new_h.astype(np.float32)
                self._prev_z = new_z.astype(np.float32)
            except Exception as e:
                self.get_logger().error(f'RKNN inference error: {e}')
                action_np = np.array([np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 1.0)], dtype=np.float32)
        else:
            # No model yet — random exploration fills the replay buffer for the
            # server's initial world-model training.
            action_np = np.array([np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 1.0)], dtype=np.float32)

        # Apply safety + deadzone
        is_stuck = self._is_stuck
        monitor_blocking = self._safety_override

        # `_wall_stop_steps` is now a *recovery counter only* — it never sets
        # `done`. The plan removed the old "blocked for 450 ticks → episode done"
        # path because terminating wiped RSSM state mid-recovery and stopped the
        # WM from learning escapes.
        if monitor_blocking:
            self._wall_stop_steps += 1
        else:
            self._wall_stop_steps = 0

        # Scripted recovery: fires on prolonged no-progress. Action depends on
        # whether there's actually an obstacle nearby:
        #   - near wall  → reverse + turn (needs clearance before going forward)
        #   - open space → forward + turn (policy is spinning, just needs to go)
        stuck_long_enough = self.stuck_detector.stuck_counter >= 15
        wall_pinned = self._wall_stop_steps >= 15
        if (wall_pinned or stuck_long_enough) and self._recovery_steps_remaining == 0:
            self._recovery_steps_remaining = 30  # ~1s at 30Hz
            self._recovery_turn_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            self._recovery_is_reverse = wall_pinned or lidar_min < 0.4

        if self._recovery_steps_remaining > 0:
            if getattr(self, '_recovery_is_reverse', False):
                rl = -0.5 + self._recovery_turn_dir * 0.15
                rr = -0.5 - self._recovery_turn_dir * 0.15
            else:
                rl = 0.5 + self._recovery_turn_dir * 0.25
                rr = 0.5 - self._recovery_turn_dir * 0.25
            action_np = np.array([rl, rr], dtype=np.float32)
            self._recovery_steps_remaining -= 1

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
            actual_action = np.array([left_track, right_track])
        elif monitor_blocking:
            left_track = min(soft_deadzone(action_np[0], MIN_TRACK), 0.0)
            right_track = min(soft_deadzone(action_np[1], MIN_TRACK), 0.0)
            actual_action = np.array([left_track, right_track])
        else:
            left_track = soft_deadzone(action_np[0], MIN_TRACK)
            right_track = soft_deadzone(action_np[1], MIN_TRACK)
            actual_action = np.array([left_track, right_track])
            if hasattr(self, '_slip_recovery_turn_dir'):
                delattr(self, '_slip_recovery_turn_dir')

        # Publish
        track_msg = Float32MultiArray()
        track_msg.data = [float(left_track), float(right_track)]
        self.track_cmd_pub.publish(track_msg)

        # Feed action back to RSSM for next tick
        self._prev_a = actual_action.astype(np.float32)[None, :]

        # ---- Reward (multi-channel) ----
        # `feat_embed` = concat(h_t, z_t) computed from the just-observed step.
        # This is the same representation the server-side LifetimeSimHash buckets
        # on, so the episodic (rover) and lifetime (server) novelty signals stay
        # aligned. Pre-RKNN the state is zeros → episodic novelty saturates near
        # 1.0, which is fine for the random-exploration warm-up.
        feat_embed = np.concatenate([self._prev_h.ravel(), self._prev_z.ravel()])
        robot_x = self._latest_odom[0] if self._latest_odom else 0.0
        robot_y = self._latest_odom[1] if self._latest_odom else 0.0
        reward_vec, _cov_step, collision = self._compute_reward_channels(
            robot_x, robot_y, self._latest_fused_yaw,
            self._latest_scan, feat_embed,
        )
        if np.isnan(reward_vec).any() or np.isinf(reward_vec).any():
            return

        # ---- Episode boundary ----
        # Only **collision** (rising edge of /emergency_stop) and the step cap
        # terminate. Stuck / spinning / wall-pinned no longer set `done` —
        # they exist purely as recovery triggers above.
        if self._post_reset_cooldown > 0:
            self._post_reset_cooldown -= 1
        episode_done = False
        done_reason = None
        if collision:
            episode_done = True
            done_reason = 'collision'
        elif self._current_episode_length + 1 >= self.MAX_EPISODE_STEPS:
            episode_done = True
            done_reason = 'max_steps'

        # Store transition (with is_first marker). Reward is the (4,) vector.
        self.buffer.add(
            bev_grid, proprio, actual_action, reward_vec, episode_done,
            is_first=is_first, rgb=rgb_chw_uint8
        )
        reward_scalar = float(reward_vec.sum())
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
            # Next tick is the first of a new episode — RSSM state will be
            # reset; clear per-episode reward state (novelty buffer, frontier
            # potential baseline) so the new episode starts fresh.
            self._next_is_first = True
            self._reset_reward_state_on_episode()
            # Clear stuck detector history so the new episode isn't instantly
            # flagged as stuck based on pre-reset positions, and enforce a
            # cooldown window before any new termination can fire.
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
            episode_count=self._episode_count,
            episode_reward_avg=avg_ep_rew,
            episode_reward_history=list(self._episode_reward_history),
        )
        self.dashboard_stats.append_reward(reward_scalar, float(current_linear))
        self.dashboard_stats.record_step()

        if self._total_steps % 300 == 0:
            self.get_logger().info(
                f'Step {self._total_steps} | AvgRew: {avg_ep_rew:.3f} | '
                f'v{self._model_version} | chunks={self._chunks_shipped}'
            )

        if np.random.rand() < 0.1:
            calib_files = list(self._calibration_dir.glob('*.npz'))
            if len(calib_files) < 100:
                timestamp = int(time.time() * 1000)
                np.savez_compressed(
                    self._calibration_dir / f"calib_{timestamp}.npz",
                    bev=bev_grid, proprio=proprio, rgb=rgb_chw_uint8,
                    prev_h=self._prev_h, prev_z=self._prev_z, prev_a=self._prev_a,
                )

        # Ship chunk when full — unlike PPO, we don't stop the rover; Dreamer is
        # off-policy so we keep collecting immediately from the next tick.
        if self.buffer.size >= self.chunk_len:
            self._ship_chunk()

    # ========== Chunk shipping ==========

    def _ship_chunk(self):
        if not self._zmq_connected:
            # Drop chunk but keep running — server will catch up when connected
            self.buffer.clear()
            return

        chunk = self.buffer.get_chunk()
        chunk['metadata'] = {
            'rover_id': 'rover-dreamer',
            'model_version': self._model_version,
            'total_steps': self._total_steps,
            'algorithm': 'dreamer',
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
        try:
            model_msg = msgpack.unpackb(data)
            model_version = model_msg["version"]
            onnx_bytes = model_msg["onnx_bytes"]
            rknn_bytes = model_msg.get("rknn_bytes")

            if model_version <= self._model_version:
                return

            self.get_logger().info(f'Received Dreamer model v{model_version}')

            train_stats = model_msg.get("train_stats", {})
            if train_stats:
                self.dashboard_stats.append_training(
                    train_stats.get('wm_loss', 0),
                    train_stats.get('actor_loss', 0),
                    train_stats.get('critic_loss', 0),
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
                    # Force RSSM reset on first tick with new model to avoid state mismatch
                    self._next_is_first = True
                    self.get_logger().info(f'RKNN Dreamer v{model_version} loaded (server-converted)')
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
                convert_script = "/home/benson/Documents/ros2-rover/convert_onnx_to_rknn.sh"
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
                    self.get_logger().info(f'RKNN Dreamer v{model_version} loaded')
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
    node = DreamerRemoteRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
