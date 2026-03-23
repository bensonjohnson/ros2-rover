#!/usr/bin/env python3
"""PPO Remote Runner - Collects rollouts on rover, trains on remote GPU server.

Combines the sensor pipeline + RKNN inference from ppo_local_runner with
ZeroMQ-based rollout shipping. The rover collects a full PPO rollout
(2048 steps), ships it to the GPU server for training, and receives
updated ONNX models back.

On-policy correctness: The rover stops after each rollout. The server
recomputes log_probs from its PyTorch model, trains PPO, then sends
updated weights back. The rover loads the new RKNN model and resumes.

Architecture:
- Unified BEV (LiDAR + Depth) -> 2x128x128 grid
- 6-dim proprioception
- RKNN NPU inference (30Hz), PyTorch CPU fallback
- Direct track control: [left_speed, right_speed] in [-1, 1]
- ZeroMQ PUSH/PULL for rollouts, PUB/SUB for model updates
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
from typing import Tuple
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
    serialize_batch, deserialize_batch,
    serialize_status, deserialize_status
)

# RKNN Support
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

# ROS2 Messages
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, Float32MultiArray
from std_srvs.srv import Trigger

# Reuse existing BEV processor and phase manager
from tractor_bringup.occupancy_processor import UnifiedBEVProcessor
from tractor_bringup.phase_manager import PhaseManager

# Proprioception normalization (must match SAC runner / RKNN export)
PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)


def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    return np.clip(normalized, -3.0, 3.0).astype(np.float32)


# ============================================================================
# Rollout Buffer (numpy, lightweight - no PyTorch needed on rover)
# ============================================================================

class RolloutBuffer:
    """On-policy rollout buffer. Stores one PPO rollout for shipping to server."""

    def __init__(self, capacity: int, proprio_dim: int = 6):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        # uint8 BEV for memory efficiency (0-1 -> 0-255)
        self.bev = np.zeros((capacity, 2, 128, 128), dtype=np.uint8)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, bev, proprio, action, reward, done):
        i = self.ptr
        self.bev[i] = (bev * 255.0).astype(np.uint8)
        self.proprio[i] = proprio
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_rollout(self):
        """Return the current rollout as a dict of numpy arrays."""
        n = self.size
        return {
            'bev': self.bev[:n].astype(np.float32) / 255.0,
            'proprio': self.proprio[:n].copy(),
            'actions': self.actions[:n].copy(),
            'rewards': self.rewards[:n].copy(),
            'dones': self.dones[:n].copy(),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


# ============================================================================
# Stuck Detector
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

    def get_recovery_action(self):
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            return np.array([-0.5, np.random.uniform(-0.8, 0.8)])
        else:
            self.recovery_mode = False
            self.stuck_counter = 0
            return None


# ============================================================================
# Dashboard HTTP Server
# ============================================================================

class DashboardStats:
    """Thread-safe stats container for the web dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'total_steps': 0,
            'model_version': 0,
            'update_count': 0,
            'buffer_fill': 0,
            'buffer_capacity': 0,
            'phase': 'exploration',
            'training_active': False,
            'warmup_active': False,
            'inference_backend': 'cpu',
            'rknn_converting': False,
            'zmq_connected': False,
            'server_status': 'unknown',
            'uptime_s': 0,
            # Live sensor/control
            'linear_vel': 0.0,
            'angular_vel': 0.0,
            'left_track': 0.0,
            'right_track': 0.0,
            'min_lidar_dist': 10.0,
            'reward': 0.0,
            'safety_blocked': False,
            'is_stuck': False,
            # Episode stats
            'episode_count': 0,
            'episode_reward_avg': 0.0,
            'episode_reward_history': [],
            # Training loss history (from server)
            'policy_loss_history': [],
            'value_loss_history': [],
            'entropy_history': [],
            'training_time_history': [],
            # Reward time series
            'reward_history': [],
            'velocity_history': [],
            # Throughput
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

    def append_training(self, policy_loss, value_loss, entropy, train_time):
        with self._lock:
            self._data['policy_loss_history'].append(policy_loss)
            self._data['value_loss_history'].append(value_loss)
            self._data['entropy_history'].append(entropy)
            self._data['training_time_history'].append(train_time)
            for key in ['policy_loss_history', 'value_loss_history', 'entropy_history', 'training_time_history']:
                if len(self._data[key]) > 200:
                    self._data[key] = self._data[key][-200:]

    def get_json(self):
        with self._lock:
            self._data['uptime_s'] = time.time() - self._start_time
            return json.dumps(self._data)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPO Remote Rover Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:12px}
h1{text-align:center;font-size:1.4em;margin-bottom:10px;color:#7eb8ff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin-bottom:10px}
.card{background:#1a1d27;border-radius:8px;padding:14px;border:1px solid #2a2d3a}
.card h3{font-size:0.85em;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.stat{font-size:2em;font-weight:700;color:#7eb8ff}
.stat-sm{font-size:1.1em;color:#aaa;margin-top:4px}
.stat-row{display:flex;justify-content:space-between;margin:4px 0}
.stat-row .label{color:#888}.stat-row .val{color:#e0e0e0;font-weight:600}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:700}
.badge-green{background:#1a3a2a;color:#4ade80}
.badge-yellow{background:#3a3a1a;color:#facc15}
.badge-red{background:#3a1a1a;color:#f87171}
.badge-blue{background:#1a2a3a;color:#60a5fa}
.chart-container{position:relative;height:180px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:700px){.row2{grid-template-columns:1fr}}
#status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.alive{background:#4ade80}.training{background:#facc15}.disconnected{background:#f87171}
</style>
</head>
<body>
<h1><span id="status-dot" class="alive"></span>PPO Remote Rover Dashboard</h1>

<div class="grid">
  <div class="card">
    <h3>Status</h3>
    <div id="status-text" class="stat" style="font-size:1.4em">Collecting</div>
    <div class="stat-sm">Phase: <span id="phase" class="badge badge-blue">exploration</span></div>
    <div class="stat-row"><span class="label">Uptime</span><span class="val" id="uptime">0s</span></div>
    <div class="stat-row"><span class="label">Throughput</span><span class="val" id="tps">0.0 steps/s</span></div>
    <div class="stat-row"><span class="label">Backend</span><span class="val" id="backend">cpu</span></div>
    <div class="stat-row"><span class="label">ZMQ</span><span class="val" id="zmq">disconnected</span></div>
    <div class="stat-row"><span class="label">Server</span><span class="val" id="server">unknown</span></div>
  </div>
  <div class="card">
    <h3>Training Progress</h3>
    <div class="stat" id="model-ver">v0</div>
    <div class="stat-row"><span class="label">Total Steps</span><span class="val" id="total-steps">0</span></div>
    <div class="stat-row"><span class="label">Updates</span><span class="val" id="updates">0</span></div>
    <div class="stat-row"><span class="label">Buffer</span><span class="val" id="buffer">0/0</span></div>
  </div>
  <div class="card">
    <h3>Live Control</h3>
    <div class="stat-row"><span class="label">Velocity</span><span class="val" id="vel">0.00 m/s</span></div>
    <div class="stat-row"><span class="label">Tracks (L/R)</span><span class="val" id="tracks">0.00 / 0.00</span></div>
    <div class="stat-row"><span class="label">LiDAR Min</span><span class="val" id="lidar">0.00 m</span></div>
    <div class="stat-row"><span class="label">Reward</span><span class="val" id="reward">0.00</span></div>
    <div class="stat-row"><span class="label">Safety</span><span class="val" id="safety">OK</span></div>
  </div>
  <div class="card">
    <h3>Episodes</h3>
    <div class="stat" id="ep-count">0</div>
    <div class="stat-row"><span class="label">Avg Reward</span><span class="val" id="ep-avg">0.00</span></div>
  </div>
</div>

<div class="row2">
  <div class="card"><h3>Reward & Velocity (Live)</h3><div class="chart-container"><canvas id="chart-live"></canvas></div></div>
  <div class="card"><h3>Episode Rewards</h3><div class="chart-container"><canvas id="chart-ep"></canvas></div></div>
</div>
<div class="row2">
  <div class="card"><h3>Policy / Value Loss</h3><div class="chart-container"><canvas id="chart-loss"></canvas></div></div>
  <div class="card"><h3>Entropy</h3><div class="chart-container"><canvas id="chart-ent"></canvas></div></div>
</div>

<script>
const POLL_MS = 1000;
const chartOpts = (yLabel) => ({
  responsive:true, maintainAspectRatio:false,
  animation:{duration:0},
  scales:{x:{display:false},y:{title:{display:true,text:yLabel,color:'#888'},ticks:{color:'#888'},grid:{color:'#2a2d3a'}},
    ...(yLabel==='Loss'?{y1:{position:'right',title:{display:true,text:'Value Loss',color:'#888'},ticks:{color:'#888'},grid:{drawOnChartArea:false}}}:{})
  },
  plugins:{legend:{labels:{color:'#ccc',boxWidth:12,padding:8}}}
});

const liveChart = new Chart(document.getElementById('chart-live'),{type:'line',data:{labels:[],datasets:[
  {label:'Reward',data:[],borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,tension:0.3},
  {label:'Velocity',data:[],borderColor:'#4ade80',borderWidth:1.5,pointRadius:0,tension:0.3,yAxisID:'y'}
]},options:chartOpts('Value')});

const epChart = new Chart(document.getElementById('chart-ep'),{type:'line',data:{labels:[],datasets:[
  {label:'Episode Reward',data:[],borderColor:'#facc15',borderWidth:2,pointRadius:2,tension:0.3,fill:true,backgroundColor:'rgba(250,204,21,0.1)'}
]},options:chartOpts('Reward')});

const lossChart = new Chart(document.getElementById('chart-loss'),{type:'line',data:{labels:[],datasets:[
  {label:'Policy Loss',data:[],borderColor:'#f87171',borderWidth:2,pointRadius:2,tension:0.3},
  {label:'Value Loss',data:[],borderColor:'#fb923c',borderWidth:2,pointRadius:2,tension:0.3,yAxisID:'y1'}
]},options:chartOpts('Loss')});

const entChart = new Chart(document.getElementById('chart-ent'),{type:'line',data:{labels:[],datasets:[
  {label:'Entropy',data:[],borderColor:'#a78bfa',borderWidth:2,pointRadius:2,tension:0.3,fill:true,backgroundColor:'rgba(167,139,250,0.1)'}
]},options:chartOpts('Entropy')});

function fmt(n,d=2){return Number(n).toFixed(d)}
function fmtTime(s){
  s=Math.floor(s);
  if(s<60)return s+'s';
  if(s<3600)return Math.floor(s/60)+'m '+s%60+'s';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}

async function poll(){
  try{
    const r=await fetch('/api/stats');
    const d=await r.json();

    const dot=document.getElementById('status-dot');
    const stxt=document.getElementById('status-text');
    if(d.training_active){dot.className='training';stxt.textContent='Server Training...';}
    else if(d.warmup_active){dot.className='alive';stxt.textContent='Warmup (Random)';}
    else{dot.className='alive';stxt.textContent='Collecting';}

    document.getElementById('phase').textContent=d.phase;
    document.getElementById('uptime').textContent=fmtTime(d.uptime_s);
    document.getElementById('tps').textContent=fmt(d.steps_per_sec,1)+' steps/s';
    const be=document.getElementById('backend');
    be.textContent=d.rknn_converting?'Converting RKNN...':d.inference_backend.toUpperCase();
    be.style.color=d.inference_backend==='npu'?'#4ade80':d.rknn_converting?'#facc15':'#888';

    const zmqEl=document.getElementById('zmq');
    zmqEl.textContent=d.zmq_connected?'Connected':'Disconnected';
    zmqEl.style.color=d.zmq_connected?'#4ade80':'#f87171';
    document.getElementById('server').textContent=d.server_status;

    document.getElementById('model-ver').textContent='v'+d.model_version;
    document.getElementById('total-steps').textContent=d.total_steps.toLocaleString();
    document.getElementById('updates').textContent=d.update_count;
    document.getElementById('buffer').textContent=d.buffer_fill+'/'+d.buffer_capacity;

    document.getElementById('vel').textContent=fmt(d.linear_vel)+' m/s';
    document.getElementById('tracks').textContent=fmt(d.left_track)+' / '+fmt(d.right_track);
    document.getElementById('lidar').textContent=fmt(d.min_lidar_dist)+' m';
    document.getElementById('reward').textContent=fmt(d.reward,3);

    const safeEl=document.getElementById('safety');
    if(d.safety_blocked){safeEl.textContent='BLOCKED';safeEl.style.color='#f87171';}
    else if(d.is_stuck){safeEl.textContent='STUCK';safeEl.style.color='#facc15';}
    else{safeEl.textContent='OK';safeEl.style.color='#4ade80';}

    document.getElementById('ep-count').textContent=d.episode_count;
    document.getElementById('ep-avg').textContent=fmt(d.episode_reward_avg,3);

    liveChart.data.labels=d.reward_history.map((_,i)=>i);
    liveChart.data.datasets[0].data=d.reward_history;
    liveChart.data.datasets[1].data=d.velocity_history;
    liveChart.update();

    if(d.episode_reward_history.length>0){
      epChart.data.labels=d.episode_reward_history.map((_,i)=>i+1);
      epChart.data.datasets[0].data=d.episode_reward_history;
      epChart.update();
    }

    if(d.policy_loss_history.length>0){
      const ll=d.policy_loss_history.map((_,i)=>i+1);
      lossChart.data.labels=ll;
      lossChart.data.datasets[0].data=d.policy_loss_history;
      lossChart.data.datasets[1].data=d.value_loss_history;
      lossChart.update();
    }

    if(d.entropy_history.length>0){
      entChart.data.labels=d.entropy_history.map((_,i)=>i+1);
      entChart.data.datasets[0].data=d.entropy_history;
      entChart.update();
    }
  }catch(e){
    document.getElementById('status-dot').className='disconnected';
    document.getElementById('status-text').textContent='Disconnected';
  }
}
setInterval(poll,POLL_MS);
poll();
</script>
</body>
</html>"""


def _make_dashboard_handler(stats: DashboardStats):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/stats':
                body = stats.get_json().encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.write(body)
            else:
                body = DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.write(body)

        def write(self, data):
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

        def log_message(self, format, *args):
            pass

    return Handler


def start_dashboard_server(stats: DashboardStats, port: int = 8080):
    handler = _make_dashboard_handler(stats)
    server = HTTPServer(('0.0.0.0', port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ============================================================================
# Main ROS2 Node
# ============================================================================

class PPORemoteRunner(Node):
    """PPO runner: RKNN inference on rover, training on remote GPU server via ZMQ."""

    def __init__(self):
        super().__init__('ppo_remote_runner')

        # Parameters
        self.declare_parameter('server_addr', '192.168.1.100')
        self.declare_parameter('server_pull_port', 5555)
        self.declare_parameter('server_pub_port', 5556)
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('rollout_steps', 2048)
        self.declare_parameter('invert_linear_vel', False)
        self.declare_parameter('dashboard_port', 8080)

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.server_pull_port = int(self.get_parameter('server_pull_port').value)
        self.server_pub_port = int(self.get_parameter('server_pub_port').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.rollout_steps = int(self.get_parameter('rollout_steps').value)
        self.invert_linear_vel = bool(self.get_parameter('invert_linear_vel').value)
        self.dashboard_port = int(self.get_parameter('dashboard_port').value)

        # Rollout buffer (no PyTorch, just numpy)
        self.buffer = RolloutBuffer(capacity=self.rollout_steps + 100, proprio_dim=6)

        # Model state
        self._model_version = 0
        self._update_count = 0
        self._total_steps = 0
        self._boot_time = time.time()

        # RKNN inference state
        self._rknn_runtime = None
        self._rknn_available = HAS_RKNN
        self._calibration_dir = Path("./calibration_data")
        self._calibration_dir.mkdir(exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix='ppo_remote_'))
        if self._rknn_available:
            self.get_logger().info('RKNNLite available - will use NPU for inference')
            self._try_load_rknn()
        else:
            self.get_logger().info('RKNNLite not available - random exploration only until server sends model')

        # Sensor state
        self._latest_depth_raw = None
        self._latest_scan = None
        self._latest_bev = None
        self._latest_odom = None
        self._latest_rf2o_odom = None
        self._latest_imu = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._safety_override = False
        self._velocity_confidence = 1.0
        self._latest_fused_yaw = 0.0

        # Curriculum
        self._curriculum_max_speed = self.max_linear

        # Action history
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20)
        self._prev_actions_buffer = deque(maxlen=30)

        # Gap following
        self._target_heading = 0.0
        self._bev_heading = 0.0
        self._max_depth_val = 0.0

        # Phase management
        self.phase_manager = PhaseManager(initial_phase='exploration')
        self._current_episode_length = 0
        self._current_episode_reward = 0.0
        self._episode_reward_history = deque(maxlen=50)
        self._state_visits = {}
        self.MAX_EPISODE_STEPS = 512  # Episode step limit
        self._episode_cooldown = 0  # Steps to wait before allowing next episode end
        self.EPISODE_COOLDOWN_STEPS = 30  # ~1 second at 30Hz to recover

        # Stuck/slip detection
        self.stuck_detector = StuckDetector(stuck_threshold=0.05)
        self._is_stuck = False
        self._consecutive_idle_steps = 0
        self._intent_without_motion_count = 0
        self._prev_min_clearance = 10.0
        self._steps_in_tight_space = 0

        # Wall avoidance
        self._steps_since_wall_stop = 0
        self._wall_stop_active = False
        self._wall_stop_steps = 0

        # Rotation tracking
        self._cumulative_rotation = 0.0
        self._last_yaw_for_rotation = None
        self._forward_progress_threshold = 0.3
        self._last_position_for_rotation = None
        self._revolution_penalty_triggered = False

        # Slip detection
        self._fwd_cmd_no_motion_count = 0
        self._slip_detected = False
        self._slip_recovery_active = False
        self._slip_backup_origin = None
        self._slip_backup_distance = 0.0
        self.SLIP_DETECTION_FRAMES = 15
        self.SLIP_CMD_THRESHOLD = 0.2
        self.SLIP_VEL_THRESHOLD = 0.03
        self.SLIP_BACKUP_LIMIT = 0.15

        # Sensor warmup
        self._sensor_warmup_complete = False
        self._sensor_warmup_countdown = 90

        # Training/upload state
        self._uploading_rollout = False
        self._awaiting_model = False
        self._shipped_model_version = 0
        self._model_ready = True
        self._warmup_active = False

        # ZMQ state
        self._zmq_ctx = None
        self._sub_sock = None
        self._zmq_connected = False
        self._zmq_loop = None
        self._stop_event = threading.Event()

        # Synchronous PUSH socket (used from _send thread — must not be async)
        self._push_ctx = zmq.Context()
        self._push_sock = self._push_ctx.socket(zmq.PUSH)
        self._push_sock.setsockopt(zmq.RECONNECT_IVL, 1000)
        self._push_sock.setsockopt(zmq.LINGER, 5000)
        self._push_sock.setsockopt(zmq.SNDHWM, 2)
        push_addr = f"tcp://{self.server_addr}:{self.server_pull_port}"
        self._push_sock.connect(push_addr)
        self.get_logger().info(f'ZMQ PUSH connected to {push_addr}')

        # BEV processor
        self.occupancy_processor = UnifiedBEVProcessor(grid_size=128, max_range=4.0)

        # Dashboard (must be initialized before ZMQ thread which references it)
        self.dashboard_stats = DashboardStats()
        self._dashboard_server = start_dashboard_server(self.dashboard_stats, self.dashboard_port)
        self._episode_count = 0

        # Start ZMQ SUB thread (after dashboard_stats is ready)
        self._zmq_thread = threading.Thread(target=self._run_zmq_loop, daemon=True)
        self._zmq_thread.start()

        # ROS2 setup
        self.bridge = CvBridge()
        self._setup_subscribers()
        self._setup_publishers()

        # Inference timer
        self.create_timer(1.0 / self.inference_rate, self._control_loop)

        # Episode reset client
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        self.get_logger().info(
            f'PPO Remote Runner initialized: '
            f'rollout={self.rollout_steps}, '
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

    def _setup_publishers(self):
        self.track_cmd_pub = self.create_publisher(Float32MultiArray, 'track_cmd_ai', 10)

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

    def _safety_cb(self, msg):
        self._safety_override = msg.data

    def _vel_conf_cb(self, msg):
        self._velocity_confidence = msg.data

    # ========== LiDAR Processing ==========

    def _find_best_gap_multiscale(self, ranges, angles, valid, scan_msg):
        if not np.any(valid):
            return 0.0, 0.0
        all_ranges = ranges.copy()
        all_ranges[~valid] = 0.0
        sort_idx = np.argsort(angles)
        sorted_angles = angles[sort_idx]
        sorted_ranges = all_ranges[sort_idx]
        if len(sorted_ranges) < 5:
            return 0.0, 0.0
        best_gap = {'angle': 0.0, 'depth': 0.0, 'score': -np.inf}
        for window_deg in [15, 25, 35]:
            window_rad = np.radians(window_deg)
            window_size = int(window_rad / scan_msg.angle_increment)
            window_size = max(3, min(window_size, len(sorted_ranges) // 3))
            if len(sorted_ranges) >= window_size:
                smoothed = np.convolve(sorted_ranges, np.ones(window_size) / window_size, mode='same')
                for i in range(window_size, len(smoothed) - window_size):
                    if smoothed[i] < 0.5:
                        continue
                    if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                        angle = sorted_angles[i]
                        abs_angle = abs(angle)
                        if abs_angle < math.pi / 2:
                            forward_bias = 1.0 - (abs_angle / (math.pi / 2)) * 0.7
                        else:
                            forward_bias = 0.3 - ((abs_angle - math.pi / 2) / (math.pi / 2)) * 0.2
                        width_bonus = 1.0 + (window_deg / 35.0) * 0.3
                        score = smoothed[i] * forward_bias * width_bonus
                        if score > best_gap['score']:
                            best_gap = {'angle': angle, 'depth': smoothed[i], 'score': score}
        return best_gap['angle'], best_gap['depth']

    def _process_lidar_metrics(self, scan_msg):
        if not scan_msg:
            return 0.0, 0.0, 0.0
        ranges = np.array(scan_msg.ranges)
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)
        if not np.any(valid):
            return 3.0, 3.0, 0.0
        valid_ranges = ranges[valid]
        min_dist_all = np.min(valid_ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        left_mask = (angles > 0.78) & (angles < 2.35) & valid
        right_mask = (angles > -2.35) & (angles < -0.78) & valid
        l_dist = np.mean(ranges[left_mask]) if np.any(left_mask) else 3.0
        r_dist = np.mean(ranges[right_mask]) if np.any(right_mask) else 3.0
        mean_side_dist = (l_dist + r_dist) / 2.0
        best_angle, best_depth = self._find_best_gap_multiscale(ranges, angles, valid, scan_msg)
        target = np.clip(best_angle / math.pi, -1.0, 1.0)
        return min_dist_all, mean_side_dist, target

    # ========== Reward Function ==========
    # Identical to ppo_local_runner.py reward function

    def _get_current_phase(self):
        return self.phase_manager.phase

    def _compute_exploration_bonus(self, bev_grid):
        phase = self._get_current_phase()
        hash_size = 8 if phase == 'exploration' else 16
        bev_small = cv2.resize(bev_grid[0], (hash_size, hash_size))
        state_hash = tuple(bev_small.flatten().astype(np.uint8))
        if state_hash not in self._state_visits:
            self._state_visits[state_hash] = 0
        visit_count = self._state_visits[state_hash]
        magnitude = {'exploration': 0.20, 'learning': 0.12, 'refinement': 0.10}[phase]
        bonus = magnitude / (1.0 + np.sqrt(visit_count))
        self._state_visits[state_hash] += 1
        return bonus

    def _compute_action_diversity_bonus(self):
        if len(self._prev_actions_buffer) < 10:
            return 0.0
        actions = np.array(list(self._prev_actions_buffer))
        lin_std = np.std(actions[:, 0])
        ang_std = np.std(actions[:, 1])
        bonus = 0.05 * (lin_std + ang_std)
        return np.clip(bonus, 0.0, 0.15)

    def _compute_reward(self, action, linear_vel, angular_vel, min_lidar_dist,
                        side_clearance, episode_done, is_stuck, is_slipping=False,
                        slip_recovery_active=False, safety_blocked=False):
        VELOCITY_DEADBAND = 0.03
        if abs(linear_vel) < VELOCITY_DEADBAND:
            linear_vel = 0.0

        reward = 0.0
        target_speed = self._curriculum_max_speed
        left_track = action[0]
        right_track = action[1]
        phase = self._get_current_phase()

        if phase == 'exploration':
            forward_bonus_mult = 1.5
            spin_penalty_scale = 0.3
        elif phase == 'learning':
            forward_bonus_mult = 1.0
            spin_penalty_scale = 0.6
        else:
            forward_bonus_mult = 0.8
            spin_penalty_scale = 1.0

        cmd_fwd = (left_track + right_track) / 2.0

        # 1. Alive bonus + stagnation
        if phase == 'exploration':
            reward += 0.08
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
            else:
                self._consecutive_idle_steps = 0
            if self._consecutive_idle_steps > 30:
                ramp = min((self._consecutive_idle_steps - 30) / 60.0, 1.0)
                reward -= 0.05 * ramp
        elif phase == 'learning':
            reward += 0.04
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
                reward -= 0.08
            else:
                self._consecutive_idle_steps = 0
        else:
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
                reward -= 0.1
            else:
                self._consecutive_idle_steps = 0

        # 2a. Intent reward
        intent_reward = 0.0
        if phase == 'exploration':
            if cmd_fwd > 0.05:
                if linear_vel < 0.03:
                    self._intent_without_motion_count += 1
                else:
                    self._intent_without_motion_count = 0
                decay = max(0.0, 1.0 - self._intent_without_motion_count / 60.0)
                intent_reward = 0.12 * min(cmd_fwd, 0.5) * decay
            else:
                self._intent_without_motion_count = 0
        elif phase == 'learning':
            if cmd_fwd > 0.1:
                if linear_vel < 0.03:
                    self._intent_without_motion_count += 1
                else:
                    self._intent_without_motion_count = 0
                decay = max(0.0, 1.0 - self._intent_without_motion_count / 60.0)
                intent_reward = 0.04 * min(cmd_fwd, 0.5) * decay
            else:
                self._intent_without_motion_count = 0
        reward += intent_reward

        # 2. Coupled forward reward
        meas_fwd = linear_vel / target_speed if target_speed > 0 else 0.0
        if cmd_fwd > 0 and meas_fwd > 0:
            max_abs = max(abs(left_track), abs(right_track))
            track_agreement = min(abs(left_track), abs(right_track)) / (max_abs + 1e-6) if max_abs > 0.05 else 1.0
            reward += min(cmd_fwd, meas_fwd) * forward_bonus_mult * (0.3 + 0.7 * track_agreement)

        # 2b. Speed bonus
        if phase == 'exploration':
            if linear_vel > 0.03:
                reward += 0.30 * min(linear_vel / target_speed, 1.0)
        elif phase == 'learning':
            if linear_vel > 0.05:
                reward += 0.20 * min(linear_vel / target_speed, 1.0)
        else:
            if linear_vel > 0.10:
                reward += 0.15 * min(linear_vel / target_speed, 1.0)

        # 3. Backward penalty / recovery reward
        if linear_vel < -0.03:
            if slip_recovery_active:
                reward += 0.1
            elif safety_blocked or min_lidar_dist < 0.25:
                # Reward reversing when blocked or very close to obstacle
                reward += 0.2 * abs(linear_vel)
            else:
                reward -= 0.15 + abs(linear_vel) * 0.8

        # 4. Tank steering
        if abs(cmd_fwd) < 0.1:
            symmetry_error = abs(abs(left_track) - abs(right_track))
            if symmetry_error > 0.2:
                reward -= 0.2 * symmetry_error * spin_penalty_scale

        max_abs_track = max(abs(left_track), abs(right_track))
        utilization = 0.0
        if max_abs_track > 0.1:
            utilization = min(abs(left_track), abs(right_track)) / max_abs_track
            should_penalize = utilization < 0.3
            if phase == 'exploration':
                should_penalize = should_penalize and (cmd_fwd < 0.05)
            if should_penalize:
                reward -= 0.3 * (1.0 - utilization) * spin_penalty_scale

        if cmd_fwd > 0.3:
            min_track = min(left_track, right_track)
            max_track = max(left_track, right_track)
            if max_track > 0.6 and min_track < 0.1:
                reward -= 0.1

        if utilization > 0.6 and cmd_fwd > 0.1:
            coord_bonus = 0.12 if phase == 'exploration' else 0.08
            reward += coord_bonus * utilization

        # 6. Smoothness
        action_diff = np.abs(action - self._prev_action)
        smoothness_mult = {'exploration': 0.02, 'learning': 0.03, 'refinement': 0.05}[phase]
        reward -= np.mean(action_diff) * smoothness_mult

        # 9. Exploration bonus
        if phase != 'refinement' and self._latest_bev is not None:
            reward += self._compute_exploration_bonus(self._latest_bev)

        # 10. Action diversity
        reward += self._compute_action_diversity_bonus()

        # 11. Gap-heading
        if abs(self._target_heading) > 0.1:
            intended_turn = right_track - left_track
            gap_direction = self._target_heading
            if linear_vel > 0.03:
                alignment_with_gap = intended_turn * gap_direction
                if alignment_with_gap > 0:
                    reward += 0.15 * min(alignment_with_gap, 0.4)
                elif abs(gap_direction) > 0.5:
                    reward -= 0.05

        # 12. Unstuck
        TIGHT_SPACE_THRESHOLD = 0.35
        if min_lidar_dist < TIGHT_SPACE_THRESHOLD:
            self._steps_in_tight_space += 1
        else:
            self._steps_in_tight_space = 0

        clearance_delta = min_lidar_dist - self._prev_min_clearance
        if min_lidar_dist < TIGHT_SPACE_THRESHOLD and clearance_delta > 0.02:
            tightness_factor = (TIGHT_SPACE_THRESHOLD - min_lidar_dist) / TIGHT_SPACE_THRESHOLD
            escape_bonus = clearance_delta * 2.0 * (1.0 + tightness_factor)
            reward += np.clip(escape_bonus, 0.0, 0.3)

        if min_lidar_dist < TIGHT_SPACE_THRESHOLD and self._steps_in_tight_space > 15:
            angular_action_magnitude = abs(right_track - left_track)
            if angular_action_magnitude > 0.3:
                reward += 0.1 * angular_action_magnitude

        # 13. Stuck recovery
        if is_stuck:
            fwd_effort = abs(left_track + right_track)
            reward -= 1.0 * fwd_effort
            rot_effort = abs(left_track - right_track)
            reward += 1.0 * rot_effort
            if left_track * right_track > 0:
                reward -= 0.5

        # 14. Slip
        if is_slipping:
            reward -= 0.3 * max(cmd_fwd, 0.0)

        # 15. Arc turn bonus
        if phase == 'exploration':
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.05, 0.2, 0.25
        elif phase == 'learning':
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.10, 0.3, 0.25
        else:
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.2, 0.5, 0.3
        if linear_vel > arc_vel_thresh and abs(angular_vel) > arc_ang_thresh and min_lidar_dist > arc_clear_thresh:
            reward += 0.3 * abs(linear_vel) * abs(angular_vel)

        # 16. Wall proximity
        if min_lidar_dist < 0.4 and linear_vel > 0.1:
            reward -= 0.25 * (0.4 - min_lidar_dist) * linear_vel

        # 17. Slip recovery bonus
        if slip_recovery_active and linear_vel < -0.02:
            reward += 0.2 * abs(linear_vel)

        # 18. Heading tracking
        if cmd_fwd > 0.05:
            track_diff = abs(right_track - left_track)
            straight_intent = max(0.0, 1.0 - track_diff * 5.0)
            if straight_intent > 0.1 and abs(angular_vel) > 0.1:
                reward -= 0.2 * abs(angular_vel) * straight_intent

        # 18b. Straight driving bonus
        if cmd_fwd > 0.05 and linear_vel > 0.05:
            straightness = max(0.0, 1.0 - abs(angular_vel) * 2.5)
            if straightness > 0.3:
                reward += 0.15 * straightness * min(linear_vel / target_speed, 1.0)

        # 19. Wall avoidance system
        wall_stopped = safety_blocked and linear_vel < 0.05
        if wall_stopped:
            self._wall_stop_active = True
            self._wall_stop_steps += 1
            self._steps_since_wall_stop = 0
        else:
            if self._wall_stop_active:
                self._wall_stop_active = False
                self._wall_stop_steps = 0
            self._steps_since_wall_stop += 1

        if not wall_stopped and linear_vel > 0.05:
            streak_factor = min(self._steps_since_wall_stop / 150.0, 1.0)
            vel_factor = min(linear_vel / target_speed, 1.0)
            clearance_factor = np.clip((min_lidar_dist - 0.3) / 0.5, 0.0, 1.0)
            avoidance_base = {'exploration': 0.20, 'learning': 0.25, 'refinement': 0.30}[phase]
            reward += avoidance_base * streak_factor * vel_factor * clearance_factor

        if wall_stopped:
            # Mild initial penalty that ramps up — gives agent time to learn to back out
            ramp = min(self._wall_stop_steps / 450.0, 1.0)
            penalty = -0.2 - 0.5 * ramp
            if phase == 'exploration':
                penalty *= 0.5
            elif phase == 'learning':
                penalty *= 0.75
            reward += penalty
            # Reward active recovery attempts while blocked
            rot_effort = abs(left_track - right_track)
            if rot_effort > 0.3:
                reward += 0.2 * rot_effort
            if linear_vel < -0.02:
                reward += 0.25 * abs(linear_vel)

        self._prev_min_clearance = min_lidar_dist
        return np.clip(reward, -1.0, 1.0)

    # ========== Control Loop ==========

    def _control_loop(self):
        # Stop robot during rollout upload or while waiting for updated model
        if self._uploading_rollout or self._awaiting_model:
            stop_msg = Float32MultiArray()
            stop_msg.data = [0.0, 0.0]
            self.track_cmd_pub.publish(stop_msg)
            return

        if self._latest_depth_raw is None or self._latest_scan is None:
            return

        # Sensor warmup
        if not self._sensor_warmup_complete:
            self._sensor_warmup_countdown -= 1
            if self._sensor_warmup_countdown <= 0:
                self._sensor_warmup_complete = True
                self.get_logger().info('Sensor warmup complete')
            return

        # 1. Build BEV
        bev_grid = self.occupancy_processor.process(
            depth_img=self._latest_depth_raw,
            laser_scan=self._latest_scan
        )
        self._latest_bev = bev_grid

        # BEV gap heading
        laser_channel = bev_grid[0]
        front_half = laser_channel[64:128, :]
        free_space = 1.0 - front_half
        col_scores = np.mean(free_space, axis=0)
        forward_bias = np.zeros(128)
        forward_bias[54:74] = 0.1
        col_scores = col_scores + forward_bias
        col_scores = np.convolve(col_scores, np.ones(13) / 13, mode='same')
        best_col = np.argmax(col_scores)
        raw_heading = (64 - best_col) / 64.0
        alpha = 0.3
        self._bev_heading = alpha * raw_heading + (1 - alpha) * self._bev_heading

        # Min forward distance from BEV
        center_patch = laser_channel[118:128, 59:69]
        obstacle_density = np.mean(center_patch) if center_patch.size > 0 else 0.0
        self._min_forward_dist = (1.0 - obstacle_density) * 4.0

        # LiDAR metrics
        lidar_min, lidar_sides, gap_heading = self._process_lidar_metrics(self._latest_scan)
        self._target_heading = gap_heading

        # Velocity from rf2o or EKF fallback
        current_linear = 0.0
        current_angular = 0.0
        if self._latest_rf2o_odom:
            current_linear = self._latest_rf2o_odom[0]
            current_angular = self._latest_rf2o_odom[1]
        elif self._latest_odom:
            current_linear = self._latest_odom[2]
            current_angular = self._latest_odom[3]

        if self.invert_linear_vel:
            current_linear = -current_linear

        # Slip detection
        cmd_fwd_for_slip = (self._prev_action[0] + self._prev_action[1]) / 2.0
        if cmd_fwd_for_slip > self.SLIP_CMD_THRESHOLD and abs(current_linear) < self.SLIP_VEL_THRESHOLD:
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

        # Rotation tracking
        if self._last_yaw_for_rotation is not None:
            yaw_delta = self._latest_fused_yaw - self._last_yaw_for_rotation
            if yaw_delta > math.pi:
                yaw_delta -= 2 * math.pi
            elif yaw_delta < -math.pi:
                yaw_delta += 2 * math.pi
            self._cumulative_rotation += abs(yaw_delta)
            if self._cumulative_rotation >= 2 * math.pi:
                self._revolution_penalty_triggered = True
                self._cumulative_rotation = 0.0
        self._last_yaw_for_rotation = self._latest_fused_yaw

        if self._latest_odom and self._last_position_for_rotation is not None:
            x, y = self._latest_odom[0], self._latest_odom[1]
            last_x, last_y = self._last_position_for_rotation
            distance = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
            if distance > self._forward_progress_threshold:
                self._cumulative_rotation = 0.0
                self._last_position_for_rotation = (x, y)
        elif self._latest_odom:
            self._last_position_for_rotation = (self._latest_odom[0], self._latest_odom[1])

        # Proprioception
        proprio_raw = np.array([
            lidar_min,
            self._prev_action[0],
            self._prev_action[1],
            current_linear,
            current_angular,
            self._bev_heading
        ], dtype=np.float32)
        proprio = normalize_proprio(proprio_raw)

        # 2. Inference
        # Wait for initial model from server before collecting any data
        # PPO is on-policy: rollouts must come from the server's policy
        if self._model_version == 0:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('Waiting for initial model from server before starting...')
            # Stop robot and skip this tick — don't collect off-policy data
            stop_msg = Float32MultiArray()
            stop_msg.data = [0.0, 0.0]
            self.track_cmd_pub.publish(stop_msg)
            return
        elif self._rknn_runtime is not None:
            # NPU inference (fast path)
            if self._warmup_active:
                self._warmup_active = False
                self.get_logger().info('Warmup complete, switching to RKNN policy')

            bev_input = bev_grid[None, ...].astype(np.float32)
            proprio_input = proprio[None, ...]
            outputs = self._rknn_runtime.inference(inputs=[bev_input, proprio_input])
            action_mean = outputs[0][0]

            if np.isnan(action_mean).any() or np.isinf(action_mean).any():
                self.get_logger().error('RKNN output NaN/Inf, using zeros')
                action_mean = np.zeros(2)

            log_std = outputs[1][0] if len(outputs) > 1 else np.zeros(2)
            std = np.exp(np.clip(log_std, -14, 0.7))
            noise = np.random.normal(0, 1, size=2) * std
            action_np = np.tanh(action_mean + noise).astype(np.float32)
        else:
            # No RKNN and no model - random exploration
            action_np = np.array([np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 1.0)])

        # 3. Execute action
        is_stuck = self._is_stuck
        monitor_blocking = self._safety_override

        def apply_soft_deadzone(val, min_val):
            if abs(val) < 0.001:
                return 0.0
            return math.copysign(min_val + (1.0 - min_val) * abs(val), val)

        MIN_TRACK = 0.25

        if self._slip_detected and self._slip_recovery_active and not monitor_blocking:
            if not hasattr(self, '_slip_recovery_turn_dir'):
                self._slip_recovery_turn_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            recovery_left = -0.4 + self._slip_recovery_turn_dir * 0.15
            recovery_right = -0.4 - self._slip_recovery_turn_dir * 0.15
            left_track = float(math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(recovery_left), recovery_left))
            right_track = float(math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(recovery_right), recovery_right))
            actual_action = np.array([left_track, right_track])
        elif monitor_blocking:
            # Safety monitor blocking — only allow reverse
            left_track = min(apply_soft_deadzone(action_np[0], MIN_TRACK), 0.0)
            right_track = min(apply_soft_deadzone(action_np[1], MIN_TRACK), 0.0)
            actual_action = np.array([left_track, right_track])
        else:
            left_track = apply_soft_deadzone(action_np[0], MIN_TRACK)
            right_track = apply_soft_deadzone(action_np[1], MIN_TRACK)
            actual_action = np.array([left_track, right_track])
            if hasattr(self, '_slip_recovery_turn_dir'):
                delattr(self, '_slip_recovery_turn_dir')

        # Soft episode boundaries — marks done=True in buffer for GAE computation
        # but does NOT stop or reset the rover. The agent drives continuously for
        # the full rollout. This lets it learn to recover from all situations.
        episode_done = False
        done_reason = None
        if monitor_blocking and self._wall_stop_steps >= 450:
            episode_done = True
            done_reason = 'blocked'
        elif is_stuck:
            episode_done = True
            done_reason = 'stuck'
        elif self._revolution_penalty_triggered:
            episode_done = True
            done_reason = 'spinning'
            self._revolution_penalty_triggered = False

        # Track action diversity
        self._prev_actions_buffer.append(actual_action.copy())

        # Publish
        track_msg = Float32MultiArray()
        track_msg.data = [float(left_track), float(right_track)]
        self.track_cmd_pub.publish(track_msg)

        # 4. Reward
        reward = self._compute_reward(
            actual_action, current_linear, current_angular,
            lidar_min, lidar_sides, episode_done, is_stuck,
            is_slipping=self._slip_detected,
            slip_recovery_active=self._slip_recovery_active,
            safety_blocked=monitor_blocking
        )

        if np.isnan(reward) or np.isinf(reward):
            return

        # 5. Store transition (no log_prob or value - server recomputes)
        self.buffer.add(bev_grid, proprio, actual_action, reward, episode_done)
        self._total_steps += 1
        self._current_episode_reward += reward
        self._current_episode_length += 1

        # Soft episode boundary — log and track stats but keep driving
        if episode_done:
            self.phase_manager.record_training_episode(
                reward=self._current_episode_reward,
                collided=(done_reason == 'blocked'),
                length=self._current_episode_length
            )
            self._episode_reward_history.append(self._current_episode_reward)
            self._episode_count += 1
            self.get_logger().info(
                f'Soft episode boundary ({done_reason}) | len={self._current_episode_length} | '
                f'rew={self._current_episode_reward:.2f} | ep#{self._episode_count}'
            )
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        # Update state
        self._prev_action = actual_action
        self._prev_linear_cmds.append(actual_action[0])

        # Dashboard stats
        avg_ep_rew = float(np.mean(list(self._episode_reward_history))) if self._episode_reward_history else 0.0
        backend = 'npu' if self._rknn_runtime else 'cpu'
        self.dashboard_stats.update(
            total_steps=self._total_steps,
            model_version=self._model_version,
            update_count=self._update_count,
            buffer_fill=self.buffer.size,
            buffer_capacity=self.rollout_steps,
            phase=self._get_current_phase(),
            training_active=self._uploading_rollout or self._awaiting_model,
            warmup_active=self._warmup_active,
            inference_backend=backend,
            zmq_connected=self._zmq_connected,
            linear_vel=float(current_linear),
            angular_vel=float(current_angular),
            left_track=float(left_track),
            right_track=float(right_track),
            min_lidar_dist=float(lidar_min),
            reward=float(reward),
            safety_blocked=bool(monitor_blocking),
            is_stuck=bool(is_stuck),
            episode_count=self._episode_count,
            episode_reward_avg=avg_ep_rew,
            episode_reward_history=list(self._episode_reward_history),
        )
        self.dashboard_stats.append_reward(float(reward), float(current_linear))
        self.dashboard_stats.record_step()

        # Log periodically
        if self._total_steps % 300 == 0:
            phase = self._get_current_phase()
            avg_rew = np.mean(list(self._episode_reward_history)) if self._episode_reward_history else 0.0
            self.get_logger().info(
                f'Step {self._total_steps} | Phase: {phase} | '
                f'Buffer: {self.buffer.size}/{self.rollout_steps} | '
                f'AvgRew: {avg_rew:.3f} | v{self._model_version}'
            )

        # Save calibration data occasionally
        if np.random.rand() < 0.1:
            calib_files = list(self._calibration_dir.glob('*.npz'))
            if len(calib_files) < 100:
                timestamp = int(time.time() * 1000)
                np.savez_compressed(
                    self._calibration_dir / f"calib_{timestamp}.npz",
                    bev=bev_grid, proprio=proprio
                )

        # 6. Ship rollout when buffer is full
        if self.buffer.size >= self.rollout_steps:
            self._ship_rollout()

    # ========== Rollout Shipping ==========

    def _ship_rollout(self):
        """Ship the completed rollout to the server via ZMQ."""
        if not self._zmq_connected:
            self.get_logger().warn('ZMQ not connected, discarding rollout.')
            self.buffer.clear()
            return

        self._uploading_rollout = True
        self._awaiting_model = True
        self._shipped_model_version = self._model_version
        self.get_logger().info(
            f'Shipping rollout ({self.buffer.size} steps, policy v{self._model_version}) to server. '
            f'Rover will stop until updated model is received.'
        )

        # Stop robot
        stop_msg = Float32MultiArray()
        stop_msg.data = [0.0, 0.0]
        self.track_cmd_pub.publish(stop_msg)

        # Get rollout data
        rollout = self.buffer.get_rollout()

        # Add metadata
        rollout['metadata'] = {
            'rover_id': 'rover-ppo',
            'model_version': self._model_version,
            'total_steps': self._total_steps,
            'algorithm': 'ppo',
            'timestamp': time.time(),
        }

        # Serialize and send in background thread
        # _push_sock is a plain zmq.Socket (not async) — safe to call from any thread
        def _send():
            try:
                msg_bytes = serialize_batch(rollout)
                msg_size_mb = len(msg_bytes) / (1024 * 1024)
                self.get_logger().info(f'Rollout serialized: {msg_size_mb:.1f} MB')
                self._push_sock.send(msg_bytes)
                self.get_logger().info('Rollout shipped successfully')
            except Exception as e:
                self.get_logger().error(f'Rollout shipping failed: {e}')
            finally:
                self.buffer.clear()
                self._uploading_rollout = False

        thread = threading.Thread(target=_send, daemon=True)
        thread.start()

    # ========== ZMQ Communication ==========

    def _run_zmq_loop(self):
        """Entry point for ZMQ background thread."""
        asyncio.run(self._zmq_main())

    async def _zmq_main(self):
        """Main ZMQ event loop — connects SUB socket and listens for model/status."""
        try:
            self._zmq_loop = asyncio.get_running_loop()
            self._zmq_ctx = zmq.asyncio.Context()

            # SUB socket for receiving models + status from server
            self._sub_sock = self._zmq_ctx.socket(zmq.SUB)
            self._sub_sock.setsockopt(zmq.RECONNECT_IVL, 1000)
            self._sub_sock.subscribe(b"model")
            self._sub_sock.subscribe(b"status")
            pub_addr = f"tcp://{self.server_addr}:{self.server_pub_port}"
            self._sub_sock.connect(pub_addr)
            self.get_logger().info(f'ZMQ SUB connected to {pub_addr}')

            self._zmq_connected = True
            self.dashboard_stats.update(zmq_connected=True)

            # Listen for messages from server
            poll_count = 0
            while not self._stop_event.is_set():
                try:
                    if await self._sub_sock.poll(timeout=1000):
                        frames = await self._sub_sock.recv_multipart()
                        self.get_logger().info(f'ZMQ SUB received {len(frames)} frames, topic={frames[0] if frames else "?"}')
                        if len(frames) >= 2:
                            topic = frames[0]
                            data = frames[1]
                            if topic == b"model":
                                self.get_logger().info(f'Processing model message ({len(data)} bytes)')
                                await self._handle_model_message(data)
                            elif topic == b"status":
                                self._handle_status_message(data)
                    else:
                        poll_count += 1
                        if poll_count % 30 == 0:
                            self.get_logger().warn(f'ZMQ SUB: no messages received in {poll_count}s (server publishing?)')
                except Exception as e:
                    self.get_logger().error(f'ZMQ recv error: {e}')
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1.0)

        except Exception as e:
            self.get_logger().error(f"ZMQ loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._zmq_connected = False
            self.dashboard_stats.update(zmq_connected=False)
            if self._zmq_ctx:
                self._zmq_ctx.destroy(linger=0)

    async def _handle_model_message(self, data):
        """Handle a model update from the server."""
        try:
            model_msg = msgpack.unpackb(data)
            model_version = model_msg["version"]
            onnx_bytes = model_msg["onnx_bytes"]
            rknn_bytes = model_msg.get("rknn_bytes")

            if model_version <= self._model_version:
                return

            if rknn_bytes:
                self.get_logger().info(
                    f'Received model v{model_version} (ONNX: {len(onnx_bytes)}B, RKNN: {len(rknn_bytes)}B)'
                )
            else:
                self.get_logger().info(f'Received model v{model_version}, ONNX: {len(onnx_bytes)} bytes')

            # Update training stats from server if included
            train_stats = model_msg.get("train_stats", {})
            if train_stats:
                self.dashboard_stats.append_training(
                    train_stats.get('policy_loss', 0),
                    train_stats.get('value_loss', 0),
                    train_stats.get('entropy', 0),
                    train_stats.get('train_time', 0),
                )

            # If server sent pre-converted RKNN, use it directly (fast path)
            if rknn_bytes and HAS_RKNN:
                rknn_path = self._temp_dir / "latest_model.rknn"
                with open(rknn_path, 'wb') as f:
                    f.write(rknn_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                self.get_logger().info('Loading server-converted RKNN...')
                if self._load_rknn_model(str(rknn_path)):
                    self._model_version = model_version
                    self._update_count = model_version
                    self.get_logger().info(f'RKNN model v{model_version} loaded on NPU (server-converted)')
                else:
                    self.get_logger().error('Failed to load server-converted RKNN')
            elif HAS_RKNN:
                # Fallback: convert on-device
                onnx_path = self._temp_dir / "latest_model.onnx"
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                self.get_logger().info('No RKNN from server, converting on-device...')
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
                self.get_logger().info(f'Model v{model_version} received (no RKNN, exploration only)')

            # Resume collection now that we have the updated policy
            if self._awaiting_model and model_version > self._shipped_model_version:
                self._awaiting_model = False
                self.get_logger().info(
                    f'New policy v{model_version} loaded, resuming rollout collection'
                )

        except Exception as e:
            self.get_logger().error(f'Model handling failed: {e}')
            self.dashboard_stats.update(rknn_converting=False)

    def _handle_status_message(self, data):
        """Handle a status update from the server."""
        try:
            status = deserialize_status(data)
            server_status = status.get("status", "unknown")
            self.dashboard_stats.update(
                server_status=server_status,
                update_count=status.get("model_version", self._update_count),
            )
        except Exception:
            pass

    # ========== RKNN Management ==========

    def _try_load_rknn(self):
        """Try to load existing RKNN model."""
        # Check temp dir and common locations
        for search_dir in [self._temp_dir, Path('./checkpoints_ppo')]:
            rknn_path = search_dir / 'latest_actor.rknn'
            if rknn_path.exists():
                if self._load_rknn_model(str(rknn_path)):
                    self._model_version = 1  # Mark as having a model
                    return

        # Also check for latest_model.rknn
        rknn_path = self._temp_dir / 'latest_model.rknn'
        if rknn_path.exists():
            if self._load_rknn_model(str(rknn_path)):
                self._model_version = 1
                return

    def _convert_and_load_rknn(self, onnx_path: str, rknn_path: str, model_version: int):
        """Synchronous RKNN conversion + load (runs in thread executor)."""
        try:
            convert_script = "./convert_onnx_to_rknn.sh"
            if not os.path.exists(convert_script):
                convert_script = "/home/benson/Documents/ros2-rover/convert_onnx_to_rknn.sh"

            if os.path.exists(convert_script):
                result = subprocess.run(
                    [convert_script, onnx_path, str(self._calibration_dir)],
                    capture_output=True, text=True, timeout=300
                )

                if os.path.exists(rknn_path):
                    new_runtime = RKNNLite()
                    ret = new_runtime.load_rknn(rknn_path)
                    if ret == 0:
                        ret = new_runtime.init_runtime()
                        if ret == 0:
                            self._rknn_runtime = new_runtime
                            self._model_version = model_version
                            self._update_count = model_version
                            self.get_logger().info(f'RKNN model v{model_version} loaded on NPU')
                        else:
                            self.get_logger().error(f'RKNN runtime init failed: {ret}')
                    else:
                        self.get_logger().error(f'RKNN load failed: {ret}')
                else:
                    self.get_logger().error(f'RKNN conversion failed (no .rknn produced)')
                    self.get_logger().error(f'STDOUT: {result.stdout[-500:]}')
                    self.get_logger().error(f'STDERR: {result.stderr[-500:]}')
            else:
                self.get_logger().warn('convert_onnx_to_rknn.sh not found')
        except Exception as e:
            self.get_logger().error(f'RKNN conversion error: {e}')

    def _load_rknn_model(self, rknn_path: str) -> bool:
        if not HAS_RKNN:
            return False
        try:
            new_runtime = RKNNLite()
            ret = new_runtime.load_rknn(rknn_path)
            if ret != 0:
                self.get_logger().error(f'Failed to load RKNN: {ret}')
                return False
            ret = new_runtime.init_runtime()
            if ret != 0:
                self.get_logger().error(f'Failed to init RKNN runtime: {ret}')
                return False
            self._rknn_runtime = new_runtime
            self.get_logger().info(f'RKNN model loaded from {rknn_path}')
            return True
        except Exception as e:
            self.get_logger().error(f'RKNN load error: {e}')
            return False

    # ========== Episode Reset ==========

    def _trigger_episode_reset(self):
        if not self.reset_episode_client.wait_for_service(timeout_sec=1.0):
            return
        request = Trigger.Request()
        future = self.reset_episode_client.call_async(request)

        def _log(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info(f'Episode reset: {resp.message}')
            except Exception:
                pass

        future.add_done_callback(_log)

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
    node = PPORemoteRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
