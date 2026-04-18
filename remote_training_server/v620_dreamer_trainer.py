#!/usr/bin/env python3
"""DreamerV3 Training Server for Remote Rover Training.

Receives length-CHUNK_LEN trajectory chunks from the rover via ZeroMQ and
trains a world model + actor-critic on a replay buffer. Periodically exports
the actor (encoder + RSSM posterior step + policy head) as a single-file
ONNX and ships it back to the rover for RKNN conversion.

Flow:
  1. Rover collects ~64-step chunks using RKNN NPU inference at 30 Hz.
  2. Rover ships each chunk via ZMQ PUSH → server replay buffer.
  3. Server samples B×T trajectory slices and trains:
     - World model: reconstruction + reward + continue + KL (with free bits
       and dyn/rep balance, per DreamerV3).
     - Actor/Critic: imagination rollouts in latent space, λ-returns, REINFORCE
       with learned baseline + entropy bonus.
  4. Every N updates, export ONNX and publish via ZMQ PUB → rover.

Key differences from the PPO trainer:
  - Off-policy: chunks accumulate in a replay buffer; training does NOT wait
    for full rollouts.
  - The exported ONNX graph is the *step* function (encoder + RSSM posterior +
    actor), not a stateless policy. The rover maintains (h, z) state between ticks.

Reference: https://github.com/danijar/dreamerv3 (official JAX implementation).
"""

import os
import sys
import time
import json
import argparse
import threading
import asyncio
import signal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import zmq
import zmq.asyncio

try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    HAS_RKNN_TOOLKIT = False

from model_architectures import (
    DreamerWorldModel, DreamerActor, DreamerCritic, DreamerActorOnnxWrapper,
    OneHotDist, kl_categorical, symlog, symexp,
)
from serialization_utils import deserialize_batch, serialize_status

# ============================================================================
# Replay buffer — stores whole chunks (length T_chunk) keyed by arrival order
# ============================================================================


@dataclass
class Chunk:
    bev: np.ndarray        # (T, 2, 128, 128) float32 [0,1]
    rgb: np.ndarray        # (T, 3, 84, 84) float32 [0,1]
    proprio: np.ndarray    # (T, 6) float32
    actions: np.ndarray    # (T, 2) float32
    rewards: np.ndarray    # (T,) float32
    dones: np.ndarray      # (T,) bool
    is_first: np.ndarray   # (T,) bool


class ChunkReplay:
    """Simple chunk-based replay. Samples length-T slices uniformly at random."""

    def __init__(self, max_chunks: int = 4000):
        self.max_chunks = max_chunks
        self.chunks = deque(maxlen=max_chunks)
        self._total_steps_seen = 0

    def add(self, chunk: Chunk):
        self.chunks.append(chunk)
        self._total_steps_seen += len(chunk.rewards)

    def __len__(self):
        return len(self.chunks)

    def total_steps(self):
        return self._total_steps_seen

    def sample(self, batch: int, T: int, rng: np.random.Generator):
        """Return a batch of length-T slices. Pads with shorter chunks if needed."""
        assert len(self.chunks) > 0
        chunks = [self.chunks[i] for i in rng.integers(0, len(self.chunks), size=batch)]

        def slice_chunk(c: Chunk) -> Chunk:
            L = len(c.rewards)
            if L <= T:
                pad = T - L
                def _pad(a, mode='edge'):
                    if pad == 0:
                        return a
                    padw = [(0, pad)] + [(0, 0)] * (a.ndim - 1)
                    return np.pad(a, padw, mode=mode)
                return Chunk(
                    bev=_pad(c.bev), rgb=_pad(c.rgb), proprio=_pad(c.proprio),
                    actions=_pad(c.actions),
                    rewards=np.pad(c.rewards, (0, pad), mode='constant'),
                    dones=np.pad(c.dones, (0, pad), mode='constant'),
                    is_first=np.pad(c.is_first, (0, pad), mode='constant'),
                )
            start = rng.integers(0, L - T + 1)
            sl = slice(start, start + T)
            return Chunk(
                bev=c.bev[sl], rgb=c.rgb[sl], proprio=c.proprio[sl],
                actions=c.actions[sl], rewards=c.rewards[sl],
                dones=c.dones[sl], is_first=c.is_first[sl],
            )

        sliced = [slice_chunk(c) for c in chunks]
        return {
            'bev':      np.stack([s.bev for s in sliced]),
            'rgb':      np.stack([s.rgb for s in sliced]),
            'proprio':  np.stack([s.proprio for s in sliced]),
            'actions':  np.stack([s.actions for s in sliced]),
            'rewards':  np.stack([s.rewards for s in sliced]),
            'dones':    np.stack([s.dones for s in sliced]),
            'is_first': np.stack([s.is_first for s in sliced]),
        }


# ============================================================================
# Minimal dashboard (kept lightweight — see PPO trainer for the rich version)
# ============================================================================

from http.server import HTTPServer, BaseHTTPRequestHandler


class DashboardStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'total_steps': 0,
            'model_version': 0,
            'update_count': 0,
            'replay_chunks': 0,
            'replay_steps': 0,
            'training_active': False,
            'wm_loss': 0.0,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'recon_loss': 0.0,
            'reward_loss': 0.0,
            'kl_loss': 0.0,
            'imag_return': 0.0,
            'train_time': 0.0,
            'uptime_s': 0,
            'updates_per_sec': 0.0,
            'loss_history': [],
            'actor_loss_history': [],
            'critic_loss_history': [],
            'recon_history': [],
            'kl_history': [],
            'return_history': [],
        }
        self._start = time.time()
        self._update_times = deque(maxlen=50)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self._data:
                    self._data[k] = v

    def record_update(self):
        now = time.time()
        with self._lock:
            self._update_times.append(now)
            if len(self._update_times) >= 2:
                dt = self._update_times[-1] - self._update_times[0]
                if dt > 0:
                    self._data['updates_per_sec'] = (len(self._update_times) - 1) / dt

    def append_losses(self, wm, actor, critic, recon, kl, ret):
        with self._lock:
            for k, v in [('loss_history', wm), ('actor_loss_history', actor),
                         ('critic_loss_history', critic), ('recon_history', recon),
                         ('kl_history', kl), ('return_history', ret)]:
                self._data[k].append(v)
                if len(self._data[k]) > 200:
                    self._data[k] = self._data[k][-200:]

    def get_json(self):
        with self._lock:
            self._data['uptime_s'] = time.time() - self._start
            return json.dumps(self._data)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>DreamerV3 Training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body{font-family:sans-serif;background:#0f1117;color:#e0e0e0;padding:12px;margin:0}
h1{text-align:center;color:#7eb8ff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-bottom:10px}
.card{background:#1a1d27;border-radius:8px;padding:14px;border:1px solid #2a2d3a}
.card h3{font-size:.85em;color:#888;text-transform:uppercase;margin-bottom:8px}
.stat{font-size:2em;font-weight:700;color:#7eb8ff}
.row{display:flex;justify-content:space-between;margin:4px 0}
.row .label{color:#888}
.chart{position:relative;height:180px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:700px){.row2{grid-template-columns:1fr}}
</style></head><body>
<h1>DreamerV3 Training Dashboard</h1>
<div class="grid">
  <div class="card"><h3>Progress</h3><div class="stat" id="ver">v0</div>
    <div class="row"><span class="label">Updates</span><span id="up">0</span></div>
    <div class="row"><span class="label">Updates/s</span><span id="ups">0.0</span></div>
    <div class="row"><span class="label">Uptime</span><span id="upt">0s</span></div></div>
  <div class="card"><h3>Replay Buffer</h3><div class="stat" id="rc">0</div>
    <div class="row"><span class="label">Chunks</span><span id="rc2">0</span></div>
    <div class="row"><span class="label">Total Steps</span><span id="rs">0</span></div></div>
  <div class="card"><h3>Losses</h3>
    <div class="row"><span class="label">World Model</span><span id="wl">0</span></div>
    <div class="row"><span class="label">Actor</span><span id="al">0</span></div>
    <div class="row"><span class="label">Critic</span><span id="cl">0</span></div>
    <div class="row"><span class="label">Recon</span><span id="rl">0</span></div>
    <div class="row"><span class="label">KL</span><span id="kll">0</span></div></div>
  <div class="card"><h3>Imagined Return</h3><div class="stat" id="ir">0</div>
    <div class="row"><span class="label">Train Time</span><span id="tt">0s</span></div></div>
</div>
<div class="row2">
  <div class="card"><h3>World Model / Recon / KL</h3><div class="chart"><canvas id="c1"></canvas></div></div>
  <div class="card"><h3>Actor / Critic / Imag Return</h3><div class="chart"><canvas id="c2"></canvas></div></div>
</div>
<script>
const opts=()=>({responsive:true,maintainAspectRatio:false,animation:{duration:0},
  scales:{x:{display:false},y:{ticks:{color:'#888'},grid:{color:'#2a2d3a'}}},
  plugins:{legend:{labels:{color:'#ccc',boxWidth:12,padding:8}}}});
const c1=new Chart(document.getElementById('c1'),{type:'line',data:{labels:[],datasets:[
  {label:'WM',data:[],borderColor:'#f87171',borderWidth:2,pointRadius:1},
  {label:'Recon',data:[],borderColor:'#fb923c',borderWidth:2,pointRadius:1},
  {label:'KL',data:[],borderColor:'#a78bfa',borderWidth:2,pointRadius:1}]},options:opts()});
const c2=new Chart(document.getElementById('c2'),{type:'line',data:{labels:[],datasets:[
  {label:'Actor',data:[],borderColor:'#60a5fa',borderWidth:2,pointRadius:1},
  {label:'Critic',data:[],borderColor:'#4ade80',borderWidth:2,pointRadius:1},
  {label:'ImagReturn',data:[],borderColor:'#facc15',borderWidth:2,pointRadius:1}]},options:opts()});
function fmtT(s){s=Math.floor(s);if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m';return Math.floor(s/3600)+'h'}
function fmt(n,d=3){return Number(n).toFixed(d)}
async function poll(){try{
  const r=await fetch('/api/stats');const d=await r.json();
  document.getElementById('ver').textContent='v'+d.model_version;
  document.getElementById('up').textContent=d.update_count;
  document.getElementById('ups').textContent=fmt(d.updates_per_sec,2);
  document.getElementById('upt').textContent=fmtT(d.uptime_s);
  document.getElementById('rc').textContent=d.replay_chunks;
  document.getElementById('rc2').textContent=d.replay_chunks;
  document.getElementById('rs').textContent=d.replay_steps.toLocaleString();
  document.getElementById('wl').textContent=fmt(d.wm_loss);
  document.getElementById('al').textContent=fmt(d.actor_loss);
  document.getElementById('cl').textContent=fmt(d.critic_loss);
  document.getElementById('rl').textContent=fmt(d.recon_loss);
  document.getElementById('kll').textContent=fmt(d.kl_loss);
  document.getElementById('ir').textContent=fmt(d.imag_return);
  document.getElementById('tt').textContent=fmt(d.train_time,2)+'s';
  const n=d.loss_history.length;if(n>0){
    const labs=Array.from({length:n},(_,i)=>i+1);
    c1.data.labels=labs;c1.data.datasets[0].data=d.loss_history;
    c1.data.datasets[1].data=d.recon_history;c1.data.datasets[2].data=d.kl_history;c1.update();
    c2.data.labels=labs;c2.data.datasets[0].data=d.actor_loss_history;
    c2.data.datasets[1].data=d.critic_loss_history;c2.data.datasets[2].data=d.return_history;c2.update();
  }
}catch(e){}}
setInterval(poll,1000);poll();
</script></body></html>"""


def _make_dash_handler(stats):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/stats':
                body = stats.get_json().encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode())
        def log_message(self, *a, **kw): pass
    return H


def start_dashboard(stats, port=8080):
    srv = HTTPServer(('0.0.0.0', port), _make_dash_handler(stats))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard on http://0.0.0.0:{port}")
    return srv


# ============================================================================
# Trainer
# ============================================================================


class V620DreamerTrainer:
    """DreamerV3 trainer — V620 ROCm / CUDA / CPU auto-detected."""

    def __init__(self, args):
        self.args = args

        # --- Device + AMP (same logic as PPO trainer) ---
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} ({total_mem:.1f}GB)")

            self._is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            self._is_blackwell = any(k in gpu_name for k in ('B200', 'B100', 'GB10', 'GB200', 'Blackwell'))

            try:
                _t = torch.randn(256, 512, device='cuda')
                torch.mm(_t, _t.t())
                torch.cuda.synchronize()
                del _t
                torch.cuda.empty_cache()
                print("GPU sanity test passed")
            except Exception as e:
                print(f"GPU SANITY TEST FAILED: {e}")
                sys.exit(1)

            torch.backends.cudnn.benchmark = True
            self.use_amp = True
            if self._is_rocm:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, growth_factor=1.5)
                print("AMP FP16 - ROCm")
            elif self._is_blackwell:
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"AMP BF16 - Blackwell ({gpu_name})")
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda', init_scale=65536.0, growth_factor=2.0)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("AMP FP16 - CUDA")
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = None
            print("Using CPU (slower)")

        # --- Models ---
        self.proprio_dim = 6
        self.action_dim = 2

        self.world_model = DreamerWorldModel(
            proprio_dim=self.proprio_dim, action_dim=self.action_dim,
            embed_dim=1024, hidden_dim=512, classes=32, groups=32,
        ).to(self.device)

        feat_dim = self.world_model.feat_dim
        self.actor = DreamerActor(feat_dim=feat_dim, action_dim=self.action_dim).to(self.device)
        self.critic = DreamerCritic(feat_dim=feat_dim).to(self.device)
        self.target_critic = DreamerCritic(feat_dim=feat_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.wm_opt = optim.Adam(self.world_model.parameters(), lr=args.wm_lr, eps=1e-5)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        # --- Hyperparameters (DreamerV3 defaults) ---
        self.T_chunk = args.chunk_len           # trajectory length per sample (training)
        self.batch_size = args.batch_size       # B in (B, T)
        self.horizon = args.horizon             # imagination horizon H
        self.gamma = args.gamma                 # discount
        self.lam = args.gae_lambda              # λ for GAE-style returns
        self.kl_free = args.kl_free             # free bits
        self.kl_dyn_scale = args.kl_dyn_scale   # β_dyn (dynamics loss weight)
        self.kl_rep_scale = args.kl_rep_scale   # β_rep (representation loss weight)
        self.entropy_coef = args.entropy_coef
        self.target_update_interval = 100
        self.target_ema = 0.98
        self.max_grad_norm = 1000.0  # Dreamer uses very high clip; grads naturally small

        # --- State ---
        self.total_steps = 0
        self.update_count = 0
        self.model_version = 0

        # --- IO ---
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(args.log_dir)

        # --- Replay ---
        self.replay = ChunkReplay(max_chunks=args.replay_chunks)
        self.rng = np.random.default_rng(0)

        # --- ZMQ ---
        self.zmq_ctx = None
        self.pull_sock = None
        self.pub_sock = None

        # --- Dashboard ---
        self.dashboard_stats = DashboardStats()
        self._dash_srv = start_dashboard(self.dashboard_stats, port=args.dashboard_port)

        # --- Load checkpoint if exists ---
        self._load_latest_checkpoint()

        # Export initial ONNX so rover can boot immediately
        self._export_onnx(increment_version=(self.model_version == 0))

        n_wm = sum(p.numel() for p in self.world_model.parameters())
        n_ac = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Dreamer Trainer initialized: WM={n_wm:,}, Actor+Critic={n_ac:,}")
        print(f"  T_chunk={self.T_chunk}, batch={self.batch_size}, horizon={self.horizon}")

    # ========== Checkpoint ==========

    def _load_latest_checkpoint(self):
        latest = Path(self.args.checkpoint_dir) / 'latest_dreamer.pt'
        if not latest.exists():
            print("No checkpoint found. Starting fresh.")
            return
        ckpt = torch.load(latest, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(ckpt['world_model'], strict=False)
        self.actor.load_state_dict(ckpt['actor'], strict=False)
        self.critic.load_state_dict(ckpt['critic'], strict=False)
        self.target_critic.load_state_dict(ckpt['critic'], strict=False)
        try:
            self.wm_opt.load_state_dict(ckpt['wm_opt'])
            self.actor_opt.load_state_dict(ckpt['actor_opt'])
            self.critic_opt.load_state_dict(ckpt['critic_opt'])
        except (ValueError, KeyError):
            print("Optimizer states incompatible, re-initialized")
        self.total_steps = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.model_version = ckpt.get('model_version', 0)
        print(f"Restored: steps={self.total_steps}, v{self.model_version}")

    def _save_checkpoint(self):
        state = {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'wm_opt': self.wm_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'model_version': self.model_version,
        }
        p = Path(self.args.checkpoint_dir) / f'dreamer_update_{self.update_count}.pt'
        latest = Path(self.args.checkpoint_dir) / 'latest_dreamer.pt'
        torch.save(state, p)
        torch.save(state, latest)
        print(f"Checkpoint saved: {p}")

    def _export_onnx(self, increment_version=True):
        """Export the actor-step graph to a single-file ONNX."""
        try:
            onnx_path = Path(self.args.checkpoint_dir) / "latest_actor.onnx"
            wrapper = DreamerActorOnnxWrapper(self.world_model, self.actor).to(self.device)
            wrapper.eval()

            rssm = self.world_model.rssm
            dummy_bev = torch.zeros(1, 2, 128, 128, device=self.device)
            dummy_proprio = torch.zeros(1, self.proprio_dim, device=self.device)
            dummy_rgb = torch.zeros(1, 3, 84, 84, device=self.device)
            dummy_h = torch.zeros(1, rssm.hidden_dim, device=self.device)
            dummy_z = torch.zeros(1, rssm.stoch_dim, device=self.device)
            dummy_a = torch.zeros(1, self.action_dim, device=self.device)

            torch.onnx.export(
                wrapper,
                (dummy_bev, dummy_proprio, dummy_rgb, dummy_h, dummy_z, dummy_a),
                str(onnx_path),
                input_names=['bev', 'proprio', 'rgb', 'prev_h', 'prev_z', 'prev_a'],
                output_names=['action_mean', 'log_std', 'new_h', 'new_z'],
                opset_version=18,
                dynamic_axes={
                    'bev': {0: 'batch'}, 'proprio': {0: 'batch'}, 'rgb': {0: 'batch'},
                    'prev_h': {0: 'batch'}, 'prev_z': {0: 'batch'}, 'prev_a': {0: 'batch'},
                },
            )

            # Collapse external data file (required for ZMQ single-blob transport)
            import onnx
            data_file = str(onnx_path) + ".data"
            if os.path.exists(data_file):
                m = onnx.load(str(onnx_path), load_external_data=True)
                onnx.save(m, str(onnx_path), save_as_external_data=False)
                os.remove(data_file)

            if increment_version:
                self.model_version += 1

            size = os.path.getsize(onnx_path)
            print(f"Exported ONNX: {onnx_path} ({size} bytes)")
            return str(onnx_path)
        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_onnx_to_rknn(self, onnx_path):
        if not HAS_RKNN_TOOLKIT:
            return None
        rknn_path = onnx_path.replace('.onnx', '.rknn')
        try:
            rknn = RKNN(verbose=False)
            import onnx as onnx_lib
            model = onnx_lib.load(onnx_path)
            input_names, input_sizes, mean_values, std_values = [], [], [], []
            for inp in model.graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                shape = [1 if d == 0 else d for d in shape]
                input_names.append(inp.name)
                input_sizes.append(shape)
                n_ch = shape[1] if len(shape) == 4 else shape[-1]
                mean_values.append([0] * n_ch)
                std_values.append([1] * n_ch)
            if rknn.config(mean_values=mean_values, std_values=std_values,
                           target_platform='rk3588', optimization_level=3) != 0:
                return None
            if rknn.load_onnx(model=onnx_path, inputs=input_names, input_size_list=input_sizes) != 0:
                return None
            if rknn.build(do_quantization=False) != 0:
                return None
            if rknn.export_rknn(rknn_path) != 0:
                return None
            print(f"RKNN converted: {rknn_path} ({os.path.getsize(rknn_path)} bytes)")
            return rknn_path
        except Exception as e:
            print(f"RKNN conversion failed: {e}")
            return None

    # ========== Training ==========

    def _to_device(self, arr, dtype=torch.float32):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
        if self.device.type == 'cuda':
            t = t.pin_memory().to(self.device, non_blocking=True)
        else:
            t = t.to(self.device)
        return t

    def _compute_wm_loss(self, batch):
        """World model loss: reconstruction + reward + continue + KL."""
        bev = self._to_device(batch['bev'])
        rgb = self._to_device(batch['rgb'])
        proprio = self._to_device(batch['proprio'])
        actions = self._to_device(batch['actions'])
        rewards = self._to_device(batch['rewards'])
        dones = self._to_device(batch['dones'].astype(np.float32))
        is_first = self._to_device(batch['is_first'].astype(np.float32))

        B, T = bev.shape[0], bev.shape[1]

        # Encode all observations: flatten (B*T, ...) through the encoder
        bev_flat = bev.reshape(B * T, 2, 128, 128)
        rgb_flat = rgb.reshape(B * T, 3, 84, 84)
        pro_flat = proprio.reshape(B * T, -1)
        # Normalize RGB to ImageNet
        RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        RGB_STD = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        rgb_norm = (rgb_flat - RGB_MEAN) / RGB_STD

        obs_embed = self.world_model.encoder(bev_flat, pro_flat, rgb_norm)  # (B*T, embed_dim)
        obs_embed = obs_embed.reshape(B, T, -1)

        out = self.world_model.rssm.observe(obs_embed, actions, is_first)
        h, z = out['h'], out['z']
        prior_lg, post_lg = out['prior_logits'], out['post_logits']

        feat = self.world_model.feat(h, z).reshape(B * T, -1)

        # BEV reconstruction (BCE on logits vs target in [0,1])
        bev_logits = self.world_model.bev_decoder(feat).reshape(B, T, 2, 128, 128)
        recon_loss = F.binary_cross_entropy_with_logits(bev_logits, bev, reduction='none')
        recon_loss = recon_loss.sum(dim=(-1, -2, -3)).mean()

        # Proprio reconstruction (MSE on symlog)
        proprio_pred = self.world_model.proprio_decoder(feat).reshape(B, T, -1)
        proprio_loss = F.mse_loss(proprio_pred, symlog(proprio), reduction='mean')

        # Reward (MSE on symlog-transformed target)
        reward_pred = self.world_model.reward_head(feat).reshape(B, T)
        reward_loss = F.mse_loss(reward_pred, symlog(rewards), reduction='mean')

        # Continue (BCE; target = 1 - done)
        continue_logits = self.world_model.continue_head(feat).reshape(B, T)
        continue_target = 1.0 - dones
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits, continue_target, reduction='mean')

        # KL with free bits + dyn/rep balance (per DreamerV3)
        kl = kl_categorical(post_lg, prior_lg)           # KL(post || prior)
        # dyn: train prior toward (stop-grad) posterior
        dyn_kl = kl_categorical(post_lg.detach(), prior_lg).mean()
        # rep: train posterior toward (stop-grad) prior
        rep_kl = kl_categorical(post_lg, prior_lg.detach()).mean()
        # Free bits — clamp per-step
        dyn_kl = torch.clamp(dyn_kl, min=self.kl_free)
        rep_kl = torch.clamp(rep_kl, min=self.kl_free)
        kl_loss = self.kl_dyn_scale * dyn_kl + self.kl_rep_scale * rep_kl

        wm_loss = recon_loss + proprio_loss + reward_loss + continue_loss + kl_loss

        return wm_loss, {
            'recon': recon_loss.detach(), 'proprio': proprio_loss.detach(),
            'reward': reward_loss.detach(), 'continue': continue_loss.detach(),
            'kl': kl_loss.detach(), 'kl_raw': kl.mean().detach(),
            'wm_total': wm_loss.detach(),
            # For actor-critic step: starting latents (detached) and continue preds
            'h_start': h.detach(), 'z_start': z.detach(),
            'continue_pred': torch.sigmoid(continue_logits).detach(),
        }

    def _compute_actor_critic_loss(self, wm_info):
        """Imagine H steps from every latent in the chunk and update actor/critic."""
        h0 = wm_info['h_start'].reshape(-1, wm_info['h_start'].shape[-1])
        z0 = wm_info['z_start'].reshape(-1, wm_info['z_start'].shape[-1])

        # Imagination rollout (actor gradients flow through dynamics)
        imag = self.world_model.rssm.imagine(h0, z0, self.actor, self.horizon)
        imag_h, imag_z = imag['h'], imag['z']      # (N, H, *)
        imag_acts = imag['actions']                 # (N, H, action_dim)
        imag_logp = imag['log_probs']               # (N, H)
        imag_ent = imag['entropies']                # (N, H)

        N, H, _ = imag_h.shape
        feat_flat = torch.cat([imag_h, imag_z], dim=-1).reshape(N * H, -1)

        # Predict rewards and continues for imagined trajectories (symlog-decoded reward)
        reward_pred_sym = self.world_model.reward_head(feat_flat).reshape(N, H).squeeze(-1) \
            if self.world_model.reward_head(feat_flat).dim() == 1 else \
            self.world_model.reward_head(feat_flat).reshape(N, H)
        reward_pred = symexp(reward_pred_sym)
        continue_pred = torch.sigmoid(self.world_model.continue_head(feat_flat)).reshape(N, H)
        discount = self.gamma * continue_pred

        # Target critic values along the trajectory
        with torch.no_grad():
            value_target = self.target_critic(
                imag_h.reshape(N * H, -1), imag_z.reshape(N * H, -1)
            ).reshape(N, H)

        # λ-returns (bootstrap from final value)
        returns = torch.zeros_like(reward_pred)
        last = value_target[:, -1]
        for t in range(H - 1, -1, -1):
            if t == H - 1:
                returns[:, t] = reward_pred[:, t] + discount[:, t] * last
            else:
                returns[:, t] = reward_pred[:, t] + discount[:, t] * (
                    (1 - self.lam) * value_target[:, t + 1] + self.lam * returns[:, t + 1]
                )

        # Discount weights (product of continues): weight earlier steps more
        weights = torch.cumprod(torch.cat([torch.ones(N, 1, device=self.device), discount[:, :-1]], dim=1), dim=1).detach()

        # Actor loss: REINFORCE with learned baseline, + entropy bonus
        baseline = value_target.detach()
        advantage = (returns - baseline).detach()
        # Per-sample normalization (Dreamer uses percentile-based; mean-std is acceptable)
        adv_std = advantage.std().clamp(min=1.0)
        advantage = advantage / adv_std
        actor_loss = -(imag_logp * advantage + self.entropy_coef * imag_ent)
        actor_loss = (weights * actor_loss).mean()

        # Critic loss: MSE on symlog(return), target detached
        value_pred_sym = self.critic(
            imag_h.detach().reshape(N * H, -1), imag_z.detach().reshape(N * H, -1)
        ).reshape(N, H)
        critic_target = symlog(returns.detach())
        critic_loss = ((value_pred_sym - critic_target) ** 2 * weights).mean()

        imag_return = returns.detach().mean()

        return actor_loss, critic_loss, {
            'imag_return': imag_return,
            'adv_std': adv_std.detach(),
        }

    def _train_step(self):
        batch = self.replay.sample(self.batch_size, self.T_chunk, self.rng)

        # ---- World model update ----
        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                wm_loss, wm_info = self._compute_wm_loss(batch)
            self.wm_opt.zero_grad(set_to_none=True)
            self.scaler.scale(wm_loss).backward()
            self.scaler.unscale_(self.wm_opt)
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
            self.scaler.step(self.wm_opt)
            self.scaler.update()
        elif self.use_amp:
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                wm_loss, wm_info = self._compute_wm_loss(batch)
            self.wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
            self.wm_opt.step()
        else:
            wm_loss, wm_info = self._compute_wm_loss(batch)
            self.wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
            self.wm_opt.step()

        # ---- Actor / Critic update (world model frozen) ----
        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                actor_loss, critic_loss, ac_info = self._compute_actor_critic_loss(wm_info)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.scaler.scale(actor_loss).backward(retain_graph=True)
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            self.scaler.unscale_(self.critic_opt)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_opt)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        elif self.use_amp:
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                actor_loss, critic_loss, ac_info = self._compute_actor_critic_loss(wm_info)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()
        else:
            actor_loss, critic_loss, ac_info = self._compute_actor_critic_loss(wm_info)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()

        # Target critic EMA
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(self.target_ema).add_(p.data, alpha=1.0 - self.target_ema)

        return {
            'wm_total': wm_info['wm_total'].item(),
            'recon': wm_info['recon'].item(),
            'proprio': wm_info['proprio'].item(),
            'reward': wm_info['reward'].item(),
            'continue': wm_info['continue'].item(),
            'kl': wm_info['kl'].item(),
            'kl_raw': wm_info['kl_raw'].item(),
            'actor': actor_loss.item(),
            'critic': critic_loss.item(),
            'imag_return': ac_info['imag_return'].item(),
        }

    # ========== ZMQ ==========

    def setup_zmq(self):
        self.zmq_ctx = zmq.asyncio.Context()

        self.pull_sock = self.zmq_ctx.socket(zmq.PULL)
        self.pull_sock.bind(f"tcp://*:{self.args.zmq_pull_port}")
        print(f"ZMQ PULL bound on tcp://*:{self.args.zmq_pull_port}")

        self.pub_sock = self.zmq_ctx.socket(zmq.XPUB)
        self.pub_sock.setsockopt(zmq.XPUB_VERBOSE, 1)
        self.pub_sock.bind(f"tcp://*:{self.args.zmq_pub_port}")
        print(f"ZMQ XPUB bound on tcp://*:{self.args.zmq_pub_port}")

    async def consume_rollouts(self):
        print("Waiting for rover chunks...")
        while True:
            try:
                data = await self.pull_sock.recv()
                rollout = deserialize_batch(data)
                n = len(rollout['rewards'])
                self.total_steps += n
                chunk = Chunk(
                    bev=rollout['bev'].astype(np.float32),
                    rgb=rollout['rgb'].astype(np.float32) if 'rgb' in rollout else np.zeros((n, 3, 84, 84), dtype=np.float32),
                    proprio=rollout['proprio'].astype(np.float32),
                    actions=rollout['actions'].astype(np.float32),
                    rewards=rollout['rewards'].astype(np.float32),
                    dones=rollout['dones'].astype(bool),
                    is_first=rollout['is_first'].astype(bool),
                )
                self.replay.add(chunk)
                self.dashboard_stats.update(
                    total_steps=self.total_steps,
                    replay_chunks=len(self.replay),
                    replay_steps=self.replay.total_steps(),
                )
                print(f"Chunk received: {n} steps (buffer: {len(self.replay)} chunks, {self.replay.total_steps()} steps)")
            except Exception as e:
                print(f"Rollout consumption error: {e}")
                await asyncio.sleep(1.0)

    async def train_loop(self):
        """Run one training step per tick once the replay has enough data."""
        print(f"Waiting for {self.args.min_replay_chunks} chunks before training starts...")
        while len(self.replay) < self.args.min_replay_chunks:
            await asyncio.sleep(1.0)
        print("Training started")

        while True:
            try:
                t0 = time.time()
                self.dashboard_stats.update(training_active=True)
                metrics = await asyncio.get_event_loop().run_in_executor(None, self._train_step)
                dt = time.time() - t0
                self.update_count += 1

                # TensorBoard
                self.writer.add_scalar('wm/total', metrics['wm_total'], self.update_count)
                self.writer.add_scalar('wm/recon', metrics['recon'], self.update_count)
                self.writer.add_scalar('wm/proprio', metrics['proprio'], self.update_count)
                self.writer.add_scalar('wm/reward', metrics['reward'], self.update_count)
                self.writer.add_scalar('wm/continue', metrics['continue'], self.update_count)
                self.writer.add_scalar('wm/kl', metrics['kl'], self.update_count)
                self.writer.add_scalar('wm/kl_raw', metrics['kl_raw'], self.update_count)
                self.writer.add_scalar('actor/loss', metrics['actor'], self.update_count)
                self.writer.add_scalar('critic/loss', metrics['critic'], self.update_count)
                self.writer.add_scalar('actor/imag_return', metrics['imag_return'], self.update_count)
                self.writer.add_scalar('training/train_time_s', dt, self.update_count)
                self.writer.add_scalar('training/replay_chunks', len(self.replay), self.update_count)

                # Dashboard
                self.dashboard_stats.update(
                    update_count=self.update_count,
                    model_version=self.model_version,
                    wm_loss=metrics['wm_total'],
                    actor_loss=metrics['actor'],
                    critic_loss=metrics['critic'],
                    recon_loss=metrics['recon'],
                    reward_loss=metrics['reward'],
                    kl_loss=metrics['kl'],
                    imag_return=metrics['imag_return'],
                    train_time=dt,
                    training_active=False,
                )
                self.dashboard_stats.record_update()
                self.dashboard_stats.append_losses(
                    metrics['wm_total'], metrics['actor'], metrics['critic'],
                    metrics['recon'], metrics['kl'], metrics['imag_return'],
                )

                if self.update_count % 50 == 0:
                    print(f"[upd {self.update_count}] wm={metrics['wm_total']:.3f} "
                          f"recon={metrics['recon']:.3f} kl={metrics['kl_raw']:.3f} "
                          f"actor={metrics['actor']:.3f} critic={metrics['critic']:.3f} "
                          f"imag_R={metrics['imag_return']:.3f} ({dt:.2f}s)")

                if self.update_count % self.args.publish_interval == 0:
                    onnx_path = self._export_onnx(increment_version=True)
                    if onnx_path:
                        await self._publish_model(onnx_path)

                if self.update_count % self.args.checkpoint_interval == 0:
                    self._save_checkpoint()

            except Exception as e:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def _publish_model(self, onnx_path):
        try:
            with open(onnx_path, 'rb') as f:
                onnx_bytes = f.read()
            rknn_bytes = None
            rknn_path = self._convert_onnx_to_rknn(onnx_path)
            if rknn_path and os.path.exists(rknn_path):
                with open(rknn_path, 'rb') as f:
                    rknn_bytes = f.read()

            import msgpack
            msg = {
                "version": self.model_version,
                "onnx_bytes": onnx_bytes,
                "timestamp": time.time(),
                "train_stats": {
                    'wm_loss': self.dashboard_stats._data['wm_loss'],
                    'actor_loss': self.dashboard_stats._data['actor_loss'],
                    'critic_loss': self.dashboard_stats._data['critic_loss'],
                    'train_time': self.dashboard_stats._data['train_time'],
                },
            }
            if rknn_bytes:
                msg["rknn_bytes"] = rknn_bytes
            await self.pub_sock.send_multipart([b"model", msgpack.packb(msg)])
            if rknn_bytes:
                print(f"Published v{self.model_version} (ONNX: {len(onnx_bytes)}B, RKNN: {len(rknn_bytes)}B)")
            else:
                print(f"Published v{self.model_version} (ONNX: {len(onnx_bytes)}B, no RKNN)")
        except Exception as e:
            print(f"Publish failed: {e}")

    async def publish_status(self):
        while True:
            try:
                msg = serialize_status(
                    status='ready' if len(self.replay) >= self.args.min_replay_chunks else 'waiting',
                    model_version=self.model_version,
                    total_steps=self.total_steps,
                    update_count=self.update_count,
                )
                await self.pub_sock.send_multipart([b"status", msg])
            except Exception:
                pass
            await asyncio.sleep(5.0)

    async def watch_subscriptions(self):
        while True:
            try:
                event = await self.pub_sock.recv()
                if len(event) > 0 and event[0] == 1:
                    topic = event[1:].decode('utf-8', errors='replace')
                    print(f"Subscriber: '{topic}'")
                    if topic == "model":
                        onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
                        if os.path.exists(onnx_path):
                            await asyncio.sleep(0.2)
                            await self._publish_model(onnx_path)
            except Exception as e:
                print(f"Subscription watch error: {e}")
                await asyncio.sleep(1.0)

    async def run(self):
        self.setup_zmq()
        await asyncio.sleep(0.5)

        onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
        if os.path.exists(onnx_path):
            await self._publish_model(onnx_path)

        asyncio.create_task(self.publish_status())
        asyncio.create_task(self.watch_subscriptions())
        asyncio.create_task(self.consume_rollouts())
        await self.train_loop()

    def start(self):
        def _shutdown(signum, frame):
            print("\nShutdown signal received")
            self._save_checkpoint()
            self.writer.close()
            sys.exit(0)
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        asyncio.run(self.run())


def main():
    parser = argparse.ArgumentParser(description='V620 DreamerV3 Training Server')
    parser.add_argument('--zmq_pull_port', type=int, default=5555)
    parser.add_argument('--zmq_pub_port', type=int, default=5556)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_dreamer')
    parser.add_argument('--log_dir', type=str, default='./logs_dreamer')
    parser.add_argument('--dashboard_port', type=int, default=8080)
    parser.add_argument('--wm_lr', type=float, default=1e-4)
    parser.add_argument('--actor_lr', type=float, default=3e-5)
    parser.add_argument('--critic_lr', type=float, default=3e-5)
    parser.add_argument('--chunk_len', type=int, default=64, help='Training sequence length T')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--kl_free', type=float, default=1.0, help='Free bits per step')
    parser.add_argument('--kl_dyn_scale', type=float, default=0.5)
    parser.add_argument('--kl_rep_scale', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=3e-4)
    parser.add_argument('--replay_chunks', type=int, default=4000)
    parser.add_argument('--min_replay_chunks', type=int, default=4,
                        help='Wait for this many chunks before starting training')
    parser.add_argument('--publish_interval', type=int, default=50,
                        help='Export + publish new ONNX every N updates')
    parser.add_argument('--checkpoint_interval', type=int, default=200)
    args = parser.parse_args()

    trainer = V620DreamerTrainer(args)
    trainer.start()


if __name__ == '__main__':
    main()
