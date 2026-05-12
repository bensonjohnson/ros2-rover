#!/usr/bin/env python3
"""RLPD + HIL-SERL Training Server for Remote Rover Training.

Model-free SAC with:
  - Critic ensemble (N critics with LayerNorm, REDQ-style target subset).
  - 50/50 online / demo replay sampling (RLPD).
  - HIL-SERL intervention reward (channel 4 = -1.0 when the user grabs the
    deadman during autonomy) — policy learns to avoid needing correction.
  - Frozen ImageNet ResNet18 RGB encoder (huge sample-efficiency lever).
  - Drop the world model entirely — much simpler and empirically faster on
    real robots than Dreamer's imagination training.

Flow mirrors v620_dreamer_trainer.py:
  1. Rover ships ~64-step chunks via ZMQ PUSH on port 5555.
  2. Trainer ingests, splits into per-step transitions, appends to OnlineReplay.
     Chunks marked `is_demo` or containing `is_intervention` also append to
     DemoReplay (which persists to disk so demos survive restarts).
  3. SAC update: sample 50/50 online+demo, UTD critic updates per env step,
     1 actor + 1 alpha update per UTD block.
  4. Every `publish_interval` updates, export ONNX, optionally convert to
     unquantized RKNN, publish via ZMQ XPUB on port 5556.

Coexists with the DreamerV3 pipeline — separate checkpoint dir, log dir,
dashboard port, TensorBoard port. Reuses serialization_utils, LifetimeSimHash.
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
from collections import deque
from pathlib import Path

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

from rlpd_networks import (
    RLPDVisionEncoder, RLPDActor, RLPDCriticEnsemble, RLPDActorOnnxWrapper,
    REWARD_CHANNELS_RLPD,
)
from rlpd_replay import OnlineReplay, DemoReplay
from serialization_utils import deserialize_batch, serialize_status
from intrinsic_rewards import LifetimeSimHash


# ============================================================================
# Dashboard
# ============================================================================

from http.server import HTTPServer, BaseHTTPRequestHandler


class DashboardStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'total_steps': 0,
            'model_version': 0,
            'update_count': 0,
            'online_size': 0,
            'demo_size': 0,
            'demo_ratio_actual': 0.0,
            'training_active': False,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha': 0.0,
            'alpha_loss': 0.0,
            'q_mean': 0.0,
            'q_std_ensemble': 0.0,
            'target_q_mean': 0.0,
            'log_prob_mean': 0.0,
            'intervention_rate_recent': 0.0,
            'train_time': 0.0,
            'uptime_s': 0,
            'updates_per_sec': 0.0,
            'actor_loss_history': [],
            'critic_loss_history': [],
            'q_history': [],
            'alpha_history': [],
        }
        self._start = time.time()
        self._update_times = deque(maxlen=50)
        self._intervention_window = deque(maxlen=500)  # rolling intervention counter

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

    def record_intervention_steps(self, n_intervention: int, n_total: int):
        with self._lock:
            for _ in range(n_total):
                self._intervention_window.append(0)
            for _ in range(n_intervention):
                if self._intervention_window:
                    self._intervention_window[-1] = 1
            if self._intervention_window:
                self._data['intervention_rate_recent'] = (
                    sum(self._intervention_window) / len(self._intervention_window)
                )

    def append_losses(self, actor, critic, q, alpha):
        with self._lock:
            for k, v in [
                ('actor_loss_history', actor),
                ('critic_loss_history', critic),
                ('q_history', q),
                ('alpha_history', alpha),
            ]:
                self._data[k].append(v)
                if len(self._data[k]) > 200:
                    self._data[k] = self._data[k][-200:]

    def get_json(self):
        with self._lock:
            self._data['uptime_s'] = time.time() - self._start
            return json.dumps(self._data)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>RLPD + HIL-SERL Training</title>
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
<h1>RLPD + HIL-SERL Training Dashboard</h1>
<div class="grid">
  <div class="card"><h3>Progress</h3><div class="stat" id="ver">v0</div>
    <div class="row"><span class="label">Updates</span><span id="up">0</span></div>
    <div class="row"><span class="label">Updates/s</span><span id="ups">0.0</span></div>
    <div class="row"><span class="label">Uptime</span><span id="upt">0s</span></div></div>
  <div class="card"><h3>Replay</h3>
    <div class="row"><span class="label">Online</span><span id="ol">0</span></div>
    <div class="row"><span class="label">Demo</span><span id="dm">0</span></div>
    <div class="row"><span class="label">Demo ratio</span><span id="dr">0.50</span></div>
    <div class="row"><span class="label">Total steps</span><span id="ts">0</span></div></div>
  <div class="card"><h3>SAC</h3>
    <div class="row"><span class="label">Actor loss</span><span id="al">0</span></div>
    <div class="row"><span class="label">Critic loss</span><span id="cl">0</span></div>
    <div class="row"><span class="label">Q mean</span><span id="qm">0</span></div>
    <div class="row"><span class="label">Q std ens</span><span id="qs">0</span></div>
    <div class="row"><span class="label">α</span><span id="ap">0</span></div></div>
  <div class="card"><h3>HIL</h3>
    <div class="stat" id="ir">0%</div>
    <div class="row"><span class="label">Intervention rate</span><span id="ir2">0</span></div>
    <div class="row"><span class="label">Train time</span><span id="tt">0s</span></div></div>
</div>
<div class="row2">
  <div class="card"><h3>Actor / Critic</h3><div class="chart"><canvas id="c1"></canvas></div></div>
  <div class="card"><h3>Q / α</h3><div class="chart"><canvas id="c2"></canvas></div></div>
</div>
<script>
const opts=()=>({responsive:true,maintainAspectRatio:false,animation:{duration:0},
  scales:{x:{display:false},y:{ticks:{color:'#888'},grid:{color:'#2a2d3a'}}},
  plugins:{legend:{labels:{color:'#ccc',boxWidth:12,padding:8}}}});
const c1=new Chart(document.getElementById('c1'),{type:'line',data:{labels:[],datasets:[
  {label:'Actor',data:[],borderColor:'#60a5fa',borderWidth:2,pointRadius:1},
  {label:'Critic',data:[],borderColor:'#4ade80',borderWidth:2,pointRadius:1}]},options:opts()});
const c2=new Chart(document.getElementById('c2'),{type:'line',data:{labels:[],datasets:[
  {label:'Q',data:[],borderColor:'#facc15',borderWidth:2,pointRadius:1},
  {label:'α',data:[],borderColor:'#a78bfa',borderWidth:2,pointRadius:1}]},options:opts()});
function fmtT(s){s=Math.floor(s);if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m';return Math.floor(s/3600)+'h'}
function fmt(n,d=3){return Number(n).toFixed(d)}
async function poll(){try{
  const r=await fetch('/api/stats');const d=await r.json();
  document.getElementById('ver').textContent='v'+d.model_version;
  document.getElementById('up').textContent=d.update_count;
  document.getElementById('ups').textContent=fmt(d.updates_per_sec,2);
  document.getElementById('upt').textContent=fmtT(d.uptime_s);
  document.getElementById('ol').textContent=d.online_size.toLocaleString();
  document.getElementById('dm').textContent=d.demo_size.toLocaleString();
  document.getElementById('dr').textContent=fmt(d.demo_ratio_actual,2);
  document.getElementById('ts').textContent=d.total_steps.toLocaleString();
  document.getElementById('al').textContent=fmt(d.actor_loss);
  document.getElementById('cl').textContent=fmt(d.critic_loss);
  document.getElementById('qm').textContent=fmt(d.q_mean);
  document.getElementById('qs').textContent=fmt(d.q_std_ensemble);
  document.getElementById('ap').textContent=fmt(d.alpha);
  document.getElementById('ir').textContent=(d.intervention_rate_recent*100).toFixed(1)+'%';
  document.getElementById('ir2').textContent=fmt(d.intervention_rate_recent,3);
  document.getElementById('tt').textContent=fmt(d.train_time,2)+'s';
  const n=d.actor_loss_history.length;if(n>0){
    const labs=Array.from({length:n},(_,i)=>i+1);
    c1.data.labels=labs;c1.data.datasets[0].data=d.actor_loss_history;
    c1.data.datasets[1].data=d.critic_loss_history;c1.update();
    c2.data.labels=labs;c2.data.datasets[0].data=d.q_history;
    c2.data.datasets[1].data=d.alpha_history;c2.update();
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


def start_dashboard(stats, port=8081):
    srv = HTTPServer(('0.0.0.0', port), _make_dash_handler(stats))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard on http://0.0.0.0:{port}")
    return srv


# ============================================================================
# Trainer
# ============================================================================


class V620RLPDTrainer:
    """RLPD + HIL-SERL trainer — V620 ROCm / CUDA / CPU auto-detected."""

    def __init__(self, args):
        self.args = args

        # --- Device + AMP (same auto-detect logic as Dreamer trainer) ---
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
            self._is_rocm = False
            self._is_blackwell = False
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = None
            print("Using CPU (slower)")

        # --- Dimensions ---
        self.proprio_dim = 6
        self.action_dim = 2
        self.frame_stack = args.frame_stack
        self.state_dim = args.state_dim
        self.n_critics = args.n_critics
        self.m_subsample = args.m_subsample
        assert self.m_subsample <= self.n_critics

        # --- Models ---
        # The frozen ResNet18 trunk is cast to the AMP dtype so its forward
        # runs natively in BF16/FP16, halving memory bandwidth and roughly
        # doubling encoder throughput. The trainable parts of the encoder run
        # in autocast dtype as usual.
        backbone_dtype = self.amp_dtype if self.use_amp else torch.float32
        self.encoder = RLPDVisionEncoder(
            proprio_dim=self.proprio_dim,
            frame_stack=self.frame_stack,
            state_dim=self.state_dim,
            backbone_dtype=backbone_dtype,
        ).to(self.device)
        self.actor = RLPDActor(
            state_dim=self.state_dim, action_dim=self.action_dim,
        ).to(self.device)
        self.critic = RLPDCriticEnsemble(
            state_dim=self.state_dim, action_dim=self.action_dim,
            n_critics=self.n_critics,
        ).to(self.device)
        self.target_critic = RLPDCriticEnsemble(
            state_dim=self.state_dim, action_dim=self.action_dim,
            n_critics=self.n_critics,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # SAC temperature, auto-tuned
        self.log_alpha = nn.Parameter(
            torch.tensor(float(np.log(args.init_alpha)), device=self.device)
        )
        self.target_entropy = float(args.target_entropy)

        # --- Optimizers ---
        # Encoder is shared by actor and critic. Standard SAC: encoder grads
        # flow from the critic loss only (actor uses encoder.detach()).
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.encoder_opt = optim.Adam(encoder_params, lr=args.encoder_lr, eps=1e-5)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=args.alpha_lr, eps=1e-5)

        # Reward channel weights (channel order matches REWARD_CHANNELS_RLPD)
        self.reward_weights = {
            'coverage':     args.w_coverage,
            'frontier':     args.w_frontier,
            'collision':    args.w_collision,
            'episodic':     args.w_episodic,
            'intervention': args.w_intervention,
        }

        # Lifetime SimHash — same idea as Dreamer's, but over the encoder state.
        simhash_path = Path(args.checkpoint_dir) / 'lifetime_simhash.pkl'
        loaded = LifetimeSimHash.load(simhash_path)
        if loaded is not None and loaded.embed_dim == self.state_dim and loaded.n_bits == args.simhash_bits:
            self.lifetime_simhash = loaded
            self.lifetime_simhash.modulator_strength = args.lifetime_strength
            print(f"Lifetime SimHash restored: {len(loaded.counts)} buckets populated "
                  f"({loaded.hit_rate() * 100:.2f}% of {1 << loaded.n_bits})")
        else:
            self.lifetime_simhash = LifetimeSimHash(
                embed_dim=self.state_dim, n_bits=args.simhash_bits,
                modulator_strength=args.lifetime_strength,
            )
            print(f"Lifetime SimHash initialized fresh ({args.simhash_bits} bits)")
        self._simhash_path = simhash_path

        # --- Hyperparameters ---
        self.batch_size = args.batch_size
        self.utd = args.utd
        self.gamma = args.gamma
        self.polyak = args.polyak
        self.demo_ratio = args.demo_ratio
        self.max_grad_norm = 100.0

        # --- State ---
        self.total_steps = 0
        self.update_count = 0
        self.model_version = 0

        # --- IO ---
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(args.log_dir)

        # --- Replay ---
        self.online = OnlineReplay(
            capacity=args.online_capacity,
            n_reward_channels=len(REWARD_CHANNELS_RLPD),
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
        )
        demo_path = Path(args.checkpoint_dir) / 'demos.npz'
        self.demo_path = demo_path
        # Clean up stale `.tmp` files from previous interrupted saves so they
        # don't sit on disk eating space forever.
        for stale in Path(args.checkpoint_dir).glob('*.tmp'):
            try:
                size_mb = stale.stat().st_size / (1024 * 1024)
                stale.unlink()
                print(f"Removed stale {stale.name} ({size_mb:.1f} MB)")
            except Exception:
                pass
        self.demo = DemoReplay.load(
            demo_path,
            n_reward_channels=len(REWARD_CHANNELS_RLPD),
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
        )
        if self.demo is None:
            self.demo = DemoReplay(
                n_reward_channels=len(REWARD_CHANNELS_RLPD),
                proprio_dim=self.proprio_dim,
                action_dim=self.action_dim,
            )
            print("Demo replay: starting empty")
        else:
            print(f"Demo replay restored: {len(self.demo)} transitions")
        # Demo dirty flag — set in consume_rollouts, drained periodically in
        # train_loop. Avoids synchronously rewriting the (potentially 1+ GB)
        # demos.npz inside the asyncio event loop on every chunk arrival,
        # which previously starved the training task entirely.
        self._demo_dirty = False
        self._demo_save_lock = threading.Lock()
        self.rng = np.random.default_rng(0)

        # --- ZMQ ---
        self.zmq_ctx = None
        self.pull_sock = None
        self.pub_sock = None

        # --- Dashboard ---
        self.dashboard_stats = DashboardStats()
        self._dash_srv = start_dashboard(self.dashboard_stats, port=args.dashboard_port)

        # --- Load checkpoint if present ---
        self._load_latest_checkpoint()

        # Export initial ONNX so rover can boot immediately
        self._export_onnx(increment_version=(self.model_version == 0))

        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_act = sum(p.numel() for p in self.actor.parameters())
        n_cri = sum(p.numel() for p in self.critic.parameters())
        print(
            f"RLPD Trainer initialized: Encoder(trainable)={n_enc:,}, "
            f"Actor={n_act:,}, Critics(N={self.n_critics})={n_cri:,}"
        )
        print(
            f"  batch={self.batch_size}, UTD={self.utd}, demo_ratio={self.demo_ratio}, "
            f"frame_stack={self.frame_stack}, state_dim={self.state_dim}"
        )

    # ========== Checkpoint ==========

    def _load_latest_checkpoint(self):
        latest = Path(self.args.checkpoint_dir) / 'latest_rlpd.pt'
        if not latest.exists():
            print("No checkpoint found. Starting fresh.")
            return
        ckpt = torch.load(latest, map_location=self.device, weights_only=False)

        def _safe_load(module, sd, name):
            try:
                module.load_state_dict(sd, strict=False)
            except (RuntimeError, ValueError):
                own = module.state_dict()
                kept, skipped = 0, []
                for k, v in sd.items():
                    if k in own and own[k].shape == v.shape:
                        own[k] = v
                        kept += 1
                    else:
                        skipped.append(k)
                module.load_state_dict(own, strict=False)
                print(f"  {name}: shape-mismatch tolerant load — kept {kept}, "
                      f"re-initialized {len(skipped)} ({skipped[:3]}{'…' if len(skipped) > 3 else ''})")

        _safe_load(self.encoder, ckpt['encoder'], 'encoder')
        _safe_load(self.actor, ckpt['actor'], 'actor')
        _safe_load(self.critic, ckpt['critic'], 'critic')
        # Reload target critic from current critic state to match shapes
        _safe_load(self.target_critic, ckpt.get('target_critic', ckpt['critic']), 'target_critic')
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                if ckpt['log_alpha'].shape == self.log_alpha.shape:
                    self.log_alpha.copy_(ckpt['log_alpha'])
        try:
            self.encoder_opt.load_state_dict(ckpt['encoder_opt'])
            self.actor_opt.load_state_dict(ckpt['actor_opt'])
            self.critic_opt.load_state_dict(ckpt['critic_opt'])
            self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
        except (ValueError, KeyError):
            print("Optimizer states incompatible, re-initialized")
        self.total_steps = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.model_version = ckpt.get('model_version', 0)
        print(f"Restored: steps={self.total_steps}, v{self.model_version}, update={self.update_count}")

    def _encoder_trainable_state(self):
        """Encoder state dict with the frozen ResNet18 backbone filtered out.

        The backbone never changes — its weights come from torchvision's
        ImageNet pretrain and are re-fetched on every fresh trainer init. By
        skipping them we cut the per-step .pt size by ~4x (190 MB → 50 MB) so
        save-every-step doesn't dominate the training loop.
        """
        return {
            k: v for k, v in self.encoder.state_dict().items()
            if not k.startswith('rgb_backbone.')
        }

    def _save_checkpoint(self):
        state = {
            'encoder': self._encoder_trainable_state(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'model_version': self.model_version,
        }
        # Atomic write: torch.save to a tmp path then os.replace so a Ctrl+C
        # mid-write never leaves `latest_rlpd.pt` half-written.
        latest = Path(self.args.checkpoint_dir) / 'latest_rlpd.pt'
        tmp = latest.with_name(latest.name + '.tmp')
        torch.save(state, tmp)
        os.replace(tmp, latest)

        # Rotating snapshot at a much wider cadence so disk doesn't fill up.
        snapshot_int = max(1, int(self.args.snapshot_interval))
        if self.update_count % snapshot_int == 0:
            p = Path(self.args.checkpoint_dir) / f'rlpd_update_{self.update_count}.pt'
            torch.save(state, p)
            print(f"Snapshot saved: {p.name}")

        self.lifetime_simhash.save(self._simhash_path)
        # Persist demos at every model checkpoint so they stay in lockstep
        # with the trained weights.
        self._save_demos_now()
        if self.update_count % 10 == 0:
            print(f"latest_rlpd.pt updated @ update {self.update_count}")

    def _save_demos_now(self):
        """Synchronous demo save (lock-protected). Use _save_demos_async from
        the asyncio event loop to avoid blocking it."""
        if not self._demo_dirty:
            return
        with self._demo_save_lock:
            try:
                self.demo.save(self.demo_path)
                self._demo_dirty = False
            except Exception as e:
                print(f"Demo save failed: {e}")

    async def _save_demos_async(self):
        """Run the (potentially slow) demo save off the event loop thread."""
        if not self._demo_dirty:
            return
        await asyncio.get_event_loop().run_in_executor(None, self._save_demos_now)

    def _export_onnx(self, increment_version: bool = True):
        try:
            onnx_path = Path(self.args.checkpoint_dir) / "latest_actor.onnx"
            wrapper = RLPDActorOnnxWrapper(self.encoder, self.actor).to(self.device)
            wrapper.eval()

            K = self.frame_stack
            dummy_rgb = torch.zeros(1, 3 * K, 84, 84, device=self.device)
            dummy_bev = torch.zeros(1, 2 * K, 128, 128, device=self.device)
            dummy_proprio = torch.zeros(1, self.proprio_dim * K, device=self.device)

            # RKNN's graph optimizer uses onnxruntime to fold constants, and
            # onnxruntime materializes constants as numpy arrays — which fails
            # on BF16 because numpy has no BF16 dtype ("No corresponding Numpy
            # type for Tensor Type. bfloat16"). The encoder backbone is BF16
            # for training speed, but for the export we temporarily cast it
            # back to FP32 so the resulting ONNX graph contains only dtypes
            # that downstream tooling can handle. Restored on the way out.
            original_dtype = self.encoder.backbone_dtype
            need_cast = original_dtype != torch.float32
            if need_cast:
                self.encoder.rgb_backbone.to(dtype=torch.float32)
            try:
                torch.onnx.export(
                    wrapper,
                    (dummy_rgb, dummy_bev, dummy_proprio),
                    str(onnx_path),
                    input_names=['rgb_stack', 'bev_stack', 'proprio_stack'],
                    output_names=['mean_logstd'],
                    opset_version=17,
                    dynamic_axes={
                        'rgb_stack': {0: 'batch'},
                        'bev_stack': {0: 'batch'},
                        'proprio_stack': {0: 'batch'},
                    },
                    dynamo=False,
                )
            finally:
                if need_cast:
                    self.encoder.rgb_backbone.to(dtype=original_dtype)

            # Collapse external data file (required for single-blob ZMQ transport)
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

    def _convert_onnx_to_rknn(self, onnx_path: str):
        """Server-side RKNN conversion fallback (unquantized).

        Rover prefers its own local quantized conversion via
        convert_onnx_to_rknn.py with calibration data; this is just a backup so
        the wire envelope can ship a usable .rknn even if the rover hasn't
        collected calibration samples yet.
        """
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

    def _batch_to_tensors(self, batch: dict):
        """Convert a sampled batch (numpy) to device tensors. Casts uint8→float/255."""
        rgb = self._to_device(batch['rgb_stack'].astype(np.float32) / 255.0)
        bev = self._to_device(batch['bev_stack'].astype(np.float32) / 255.0)
        pro = self._to_device(batch['proprio_stack'])
        next_rgb = self._to_device(batch['next_rgb_stack'].astype(np.float32) / 255.0)
        next_bev = self._to_device(batch['next_bev_stack'].astype(np.float32) / 255.0)
        next_pro = self._to_device(batch['next_proprio_stack'])
        action = self._to_device(batch['action'])
        reward = self._to_device(batch['reward'])  # (B, 5)
        done = self._to_device(batch['done'].astype(np.float32))
        return rgb, bev, pro, next_rgb, next_bev, next_pro, action, reward, done

    def _sample_mixed(self):
        """RLPD 50/50 sampler. Degrades to 100% online if demo is empty."""
        B = self.batch_size
        if len(self.demo) == 0:
            return self.online.sample(B, self.frame_stack, self.rng), 0.0
        n_demo = int(round(B * self.demo_ratio))
        n_online = B - n_demo
        online_b = self.online.sample(n_online, self.frame_stack, self.rng) if n_online > 0 else None
        demo_b = self.demo.sample(n_demo, self.frame_stack, self.rng)

        if online_b is None:
            return demo_b, 1.0

        merged = {}
        for k in online_b:
            merged[k] = np.concatenate([online_b[k], demo_b[k]], axis=0)
        return merged, n_demo / B

    def _scalar_reward(self, reward_channels: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Combine 5 reward channels + SimHash modulator on episodic.

        reward_channels: (B, 5), state: (B, state_dim).
        Returns scalar reward (B,).
        """
        r = torch.zeros(reward_channels.shape[0], device=reward_channels.device)
        for i, name in enumerate(REWARD_CHANNELS_RLPD):
            r = r + self.reward_weights[name] * reward_channels[:, i]
        # SimHash modulator gives a count-decaying bonus on the episodic channel
        with torch.no_grad():
            mod = self.lifetime_simhash.modulator(state) - 1.0  # ≥ 0
        ep_idx = REWARD_CHANNELS_RLPD.index('episodic')
        r = r + mod * self.reward_weights['episodic'] * reward_channels[:, ep_idx]
        return r

    def _train_step(self):
        """One outer step: UTD critic updates, 1 actor update, 1 alpha update."""
        critic_losses = []
        q_means = []
        q_stds = []
        target_q_means = []

        last_state_detached = None  # cached from last critic iter for the
                                    # actor update (saves one encoder forward)
        # Pre-sample a single batch and reuse for UTD updates (RLPD paper does
        # one sample per UTD iter; we batch for throughput on the V620).
        for _ in range(self.utd):
            batch, demo_frac_actual = self._sample_mixed()
            (rgb, bev, pro, next_rgb, next_bev, next_pro,
             action, reward_channels, done) = self._batch_to_tensors(batch)
            B = rgb.shape[0]

            # --- Critic update ---
            if self.use_amp:
                ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype)
            else:
                ctx = torch.amp.autocast('cuda', enabled=False) if self.device.type == 'cuda' else \
                      torch.amp.autocast('cpu', enabled=False)
            with ctx:
                state = self.encoder(rgb, bev, pro)                  # encoder grads on
                with torch.no_grad():
                    next_state_tgt = self.encoder(next_rgb, next_bev, next_pro)
                    next_action, next_log_prob, _ = self.actor.sample(next_state_tgt)
                    # REDQ: M-subset min over target critics
                    idx = torch.from_numpy(
                        self.rng.choice(self.n_critics, self.m_subsample, replace=False)
                    ).to(self.device)
                    q_tgt_subset = self.target_critic.q_subset(next_state_tgt, next_action, idx)  # (M, B)
                    q_tgt_min = q_tgt_subset.min(dim=0).values
                    alpha_now = self.log_alpha.exp().detach()
                    target_q_value = q_tgt_min - alpha_now * next_log_prob
                    # Scalar reward (combine 5 channels + SimHash on episodic)
                    scalar_r = self._scalar_reward(reward_channels, state.detach())
                    y = scalar_r + self.gamma * (1.0 - done) * target_q_value

                # Online critic Q for the actual action — gradients flow to
                # critic ensemble AND encoder.
                q_all = self.critic(state, action)                  # (N, B)
                critic_loss = F.mse_loss(q_all, y.unsqueeze(0).expand_as(q_all))

            self.critic_opt.zero_grad(set_to_none=True)
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_opt)
                self.scaler.unscale_(self.encoder_opt)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    [p for p in self.encoder.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.scaler.step(self.critic_opt)
                self.scaler.step(self.encoder_opt)
                self.scaler.update()
            else:
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    [p for p in self.encoder.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.critic_opt.step()
                self.encoder_opt.step()

            # Polyak update of target critics
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                    tp.data.mul_(1.0 - self.polyak).add_(p.data, alpha=self.polyak)

            critic_losses.append(float(critic_loss.detach().item()))
            with torch.no_grad():
                q_mean_per_critic = q_all.mean(dim=1)
                q_means.append(float(q_all.mean().item()))
                q_stds.append(float(q_all.mean(dim=1).std(unbiased=False).item()))
                target_q_means.append(float(y.mean().item()))
            # Cache the state we just computed for the actor update so we
            # don't re-encode the same (rgb, bev, pro) batch a second time.
            last_state_detached = state.detach()

        # --- Actor update (one per UTD block) ---
        if self.use_amp:
            ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype)
        else:
            ctx = torch.amp.autocast('cuda', enabled=False) if self.device.type == 'cuda' else \
                  torch.amp.autocast('cpu', enabled=False)
        with ctx:
            # Block encoder grads from the actor objective (standard SAC).
            # Reuse the encoded state from the final critic iteration instead
            # of running another encoder forward — same batch, same dtype,
            # already detached, so semantics are identical.
            state_for_actor = last_state_detached
            new_action, new_log_prob, _ = self.actor.sample(state_for_actor)
            q_new = self.critic(state_for_actor, new_action).mean(dim=0)  # mean across ensemble
            alpha_now = self.log_alpha.exp()
            actor_loss = (alpha_now.detach() * new_log_prob - q_new).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

        # --- Alpha update ---
        # alpha_loss = -log_alpha * (log_prob + target_entropy).detach()
        alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Lifetime SimHash update on training states (count-only, no grad) ---
        with torch.no_grad():
            state_np = state_for_actor.reshape(-1, state_for_actor.shape[-1]).cpu().numpy()
            self.lifetime_simhash.update_and_query(state_np)

        return {
            'actor_loss': float(actor_loss.detach().item()),
            'critic_loss': float(np.mean(critic_losses)),
            'q_mean': float(np.mean(q_means)),
            'q_std_ensemble': float(np.mean(q_stds)),
            'target_q_mean': float(np.mean(target_q_means)),
            'alpha': float(alpha_now.detach().item()),
            'alpha_loss': float(alpha_loss.detach().item()),
            'log_prob_mean': float(new_log_prob.detach().mean().item()),
            'demo_ratio_actual': float(demo_frac_actual),
            'simhash_hit_rate': float(self.lifetime_simhash.hit_rate()),
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
                K_chunk = rollout['rewards'].shape[1]
                if K_chunk != len(REWARD_CHANNELS_RLPD):
                    print(f"WARNING: Dropping chunk with K={K_chunk} (expected {len(REWARD_CHANNELS_RLPD)})")
                    continue
                n = len(rollout['rewards'])
                self.total_steps += n

                chunk = {
                    'bev': rollout['bev'].astype(np.uint8) if rollout['bev'].dtype != np.uint8 else rollout['bev'],
                    'rgb': rollout.get('rgb', np.zeros((n, 3, 84, 84), dtype=np.uint8)),
                    'proprio': rollout['proprio'].astype(np.float32),
                    'actions': rollout['actions'].astype(np.float32),
                    'rewards': rollout['rewards'].astype(np.float32),
                    'dones': rollout['dones'].astype(bool),
                    'is_first': rollout['is_first'].astype(bool),
                    'is_intervention': rollout.get('is_intervention', np.zeros(n, dtype=bool)),
                    'is_demo': rollout.get('is_demo', np.zeros(n, dtype=bool)),
                }

                # Always add to online
                self.online.add_chunk(chunk)

                # Add to demo if marked as demo OR contains any interventions.
                # Mark the demo buffer dirty so the periodic background save
                # in train_loop picks it up — DO NOT save synchronously here.
                # A 1+ GB rewrite inside the event loop blocks chunk
                # reception and training entirely.
                is_demo_chunk = bool(chunk['is_demo'].any())
                has_intervention = bool(chunk['is_intervention'].any())
                if is_demo_chunk or has_intervention:
                    self.demo.add_chunk(chunk)
                    self._demo_dirty = True

                # Stats
                n_intervention = int(chunk['is_intervention'].sum())
                self.dashboard_stats.record_intervention_steps(n_intervention, n)
                self.dashboard_stats.update(
                    total_steps=self.total_steps,
                    online_size=len(self.online),
                    demo_size=len(self.demo),
                )
                tag = []
                if is_demo_chunk: tag.append('demo')
                if has_intervention: tag.append(f'intervene×{n_intervention}')
                tag_str = f" [{','.join(tag)}]" if tag else ''
                print(f"Chunk received: {n} steps{tag_str} "
                      f"(online: {len(self.online)}, demo: {len(self.demo)})")
            except Exception as e:
                print(f"Rollout consumption error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def train_loop(self):
        """One outer step per tick once the buffer is warm."""
        print(f"Waiting for {self.args.min_online_size} online transitions before training starts...")
        while len(self.online) < self.args.min_online_size:
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
                self.writer.add_scalar('actor/loss', metrics['actor_loss'], self.update_count)
                self.writer.add_scalar('actor/log_prob', metrics['log_prob_mean'], self.update_count)
                self.writer.add_scalar('actor/entropy', -metrics['log_prob_mean'], self.update_count)
                self.writer.add_scalar('critic/loss_mean', metrics['critic_loss'], self.update_count)
                self.writer.add_scalar('critic/q_mean', metrics['q_mean'], self.update_count)
                self.writer.add_scalar('critic/q_std_across_ensemble', metrics['q_std_ensemble'], self.update_count)
                self.writer.add_scalar('critic/target_q_mean', metrics['target_q_mean'], self.update_count)
                self.writer.add_scalar('alpha/value', metrics['alpha'], self.update_count)
                self.writer.add_scalar('alpha/loss', metrics['alpha_loss'], self.update_count)
                self.writer.add_scalar('replay/online_size', len(self.online), self.update_count)
                self.writer.add_scalar('replay/demo_size', len(self.demo), self.update_count)
                self.writer.add_scalar('replay/demo_ratio_actual', metrics['demo_ratio_actual'], self.update_count)
                self.writer.add_scalar('simhash/hit_rate', metrics['simhash_hit_rate'], self.update_count)
                self.writer.add_scalar('training/train_time_s', dt, self.update_count)

                # Dashboard
                self.dashboard_stats.update(
                    update_count=self.update_count,
                    model_version=self.model_version,
                    actor_loss=metrics['actor_loss'],
                    critic_loss=metrics['critic_loss'],
                    alpha=metrics['alpha'],
                    alpha_loss=metrics['alpha_loss'],
                    q_mean=metrics['q_mean'],
                    q_std_ensemble=metrics['q_std_ensemble'],
                    target_q_mean=metrics['target_q_mean'],
                    log_prob_mean=metrics['log_prob_mean'],
                    demo_ratio_actual=metrics['demo_ratio_actual'],
                    train_time=dt,
                    training_active=False,
                )
                self.dashboard_stats.record_update()
                self.dashboard_stats.append_losses(
                    metrics['actor_loss'], metrics['critic_loss'],
                    metrics['q_mean'], metrics['alpha'],
                )

                if self.update_count % 10 == 0:
                    print(
                        f"[upd {self.update_count}] actor={metrics['actor_loss']:.3f} "
                        f"critic={metrics['critic_loss']:.3f} q={metrics['q_mean']:.3f} "
                        f"α={metrics['alpha']:.3f} demo_r={metrics['demo_ratio_actual']:.2f} "
                        f"({dt:.2f}s)"
                    )

                if self.update_count % self.args.publish_interval == 0:
                    onnx_path = self._export_onnx(increment_version=True)
                    if onnx_path:
                        await self._publish_model(onnx_path)

                if self.update_count % self.args.checkpoint_interval == 0:
                    self._save_checkpoint()

                # Periodic demo save off the event loop. Cadence is keyed to
                # publish_interval (default 50 updates) so demos and ONNX
                # publishes share roughly the same wall-clock pacing without
                # writing the huge .npz on every chunk.
                if self.update_count % self.args.demo_save_interval == 0:
                    await self._save_demos_async()

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
                    'actor_loss': self.dashboard_stats._data['actor_loss'],
                    'critic_loss': self.dashboard_stats._data['critic_loss'],
                    'q_mean': self.dashboard_stats._data['q_mean'],
                    'alpha': self.dashboard_stats._data['alpha'],
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
                    status='ready' if len(self.online) >= self.args.min_online_size else 'waiting',
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
        # Track whether shutdown ran so the finally-clause doesn't double-save.
        self._shutdown_done = False

        def _do_final_save(reason: str):
            if self._shutdown_done:
                return
            self._shutdown_done = True
            try:
                print(f"\nFinal save ({reason})...")
                self._save_checkpoint()
                self.writer.close()
            except Exception as e:
                print(f"Final save failed: {e}")

        def _shutdown(signum, frame):
            _do_final_save(f"signal {signum}")
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            # asyncio sometimes swallows the SIGINT into KeyboardInterrupt
            # before our signal handler runs.
            _do_final_save("KeyboardInterrupt")
            sys.exit(0)
        finally:
            # Belt-and-suspenders: if we reach here without having saved,
            # save now so we never lose >50 updates of work.
            _do_final_save("normal exit")


def main():
    parser = argparse.ArgumentParser(description='V620 RLPD + HIL-SERL Training Server')
    # IO
    parser.add_argument('--zmq_pull_port', type=int, default=5555)
    parser.add_argument('--zmq_pub_port', type=int, default=5556)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_rlpd')
    parser.add_argument('--log_dir', type=str, default='./logs_rlpd')
    parser.add_argument('--dashboard_port', type=int, default=8081)
    # Architecture
    parser.add_argument('--frame_stack', type=int, default=4)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--n_critics', type=int, default=10)
    parser.add_argument('--m_subsample', type=int, default=2)
    # SAC
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--utd', type=int, default=10, help='Update-to-data ratio')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--init_alpha', type=float, default=0.1)
    parser.add_argument('--target_entropy', type=float, default=-2.0,
                        help='Default -|A| = -2 for 2D action')
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)
    # RLPD
    parser.add_argument('--demo_ratio', type=float, default=0.5,
                        help='Fraction of each batch sampled from demo buffer')
    parser.add_argument('--online_capacity', type=int, default=100_000)
    parser.add_argument('--min_online_size', type=int, default=1000,
                        help='Wait for this many online transitions before training starts')
    # Reward channel weights (channel order: coverage, frontier, collision, episodic, intervention)
    parser.add_argument('--w_coverage', type=float, default=1.0)
    parser.add_argument('--w_frontier', type=float, default=0.5)
    parser.add_argument('--w_collision', type=float, default=1.0)
    parser.add_argument('--w_episodic', type=float, default=0.05)
    parser.add_argument('--w_intervention', type=float, default=1.0,
                        help='Multiplier on the intervention reward channel '
                             '(channel already carries -1 from the rover on intervened steps)')
    # Lifetime SimHash
    parser.add_argument('--lifetime_strength', type=float, default=1.0)
    parser.add_argument('--simhash_bits', type=int, default=16)
    # Logging cadence
    parser.add_argument('--publish_interval', type=int, default=50)
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Save `latest_rlpd.pt` every N updates. '
                             'Default 1 (every step) — RLPD inner loops are '
                             'slow enough that a fresh checkpoint per step '
                             'is the right safety/perf trade-off.')
    parser.add_argument('--snapshot_interval', type=int, default=200,
                        help='Write rotating `rlpd_update_N.pt` snapshot every '
                             'N updates. Wider than checkpoint_interval so '
                             'historical checkpoints exist without filling '
                             'the disk with one .pt per step.')
    parser.add_argument('--demo_save_interval', type=int, default=50,
                        help='Persist demos.npz every N updates (off the '
                             'event loop). The buffer can be 1+ GB so we '
                             'do not synchronously rewrite it on every chunk.')

    args = parser.parse_args()

    trainer = V620RLPDTrainer(args)
    trainer.start()


if __name__ == '__main__':
    main()
