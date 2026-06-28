#!/usr/bin/env python3
"""Deep Explorer Network — Remote Training Server (NVIDIA GPU).

Trains a DeepExplorerNetwork policy using TD-MPC2-style objective (temporal
difference learning with a learned world model + actor-critic). Designed for
GPUs (CUDA or ROCm).

Architecture:
  - World model: learns to predict next latent state and reward given (z, a)
  - Actor (policy): learns to maximize discounted sum of rewards
  - Value (critic): learns to estimate state value (TD(lambda))

Training flow:
  1. Rover sends experience chunks via ZMQ or saved locally and rsynced
  2. Server loads chunks into a replay buffer
  3. Every N steps: sample batch, train world model, actor, critic
  4. Periodically export ONNX, convert to RKNN, ship back to rover

Usage:
  # Start server (waits for rover connection or local data)
  python3 explorer_trainer.py --port 5557 --checkpoint-dir ./checkpoints

  # Train from local data (pre-collected chunks)
  python3 explorer_trainer.py --data-dir ~/.ros/explorer_chunks/ --checkpoint-dir ./checkpoints

  # Full training + export pipeline
  python3 explorer_trainer.py --port 5557 --export --export-dir ./export
"""

import os
import sys
import time
import json
import math
import argparse
import threading
import asyncio
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Try ZMQ for live rover connection
try:
    import zmq
    import zmq.asyncio
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

# Try msgpack for chunk serialization
try:
    import msgpack
    import msgpack_numpy as mpn
    mpn.patch()
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Import the network architecture from the rover package
# (works if tractor_explorer is on the PYTHONPATH or installed)
try:
    from tractor_explorer.deep_explorer_network import (
        DeepExplorerNetwork, ExplorerConfig, normalize_lidar,
    )
except ImportError:
    # Fallback: add the rover src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src",
                                     "tractor_explorer"))
    from tractor_explorer.deep_explorer_network import (
        DeepExplorerNetwork, ExplorerConfig, normalize_lidar,
    )


# ============================================================================
# Replay Buffer
# ============================================================================

@dataclass
class Step:
    lidar: np.ndarray        # (72,)
    occ: np.ndarray          # (64, 64)
    proprio: np.ndarray      # (5,)
    action: np.ndarray       # (2,)
    reward: np.ndarray       # (5,) — multi-channel reward
    done: bool
    is_first: bool


class ReplayBuffer:
    """FIFO replay buffer for off-policy training.

    Stores individual steps, samples random sequences of length T.
    """
    def __init__(self, capacity: int = 200_000, seq_len: int = 32):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)
        self._rng = np.random.default_rng(0)

    def add(self, step: Step):
        self.buffer.append(step)

    def add_chunk(self, chunk: dict):
        """Add a full chunk (dict of numpy arrays) to replay."""
        T = len(chunk["action"])
        for t in range(T):
            self.add(Step(
                lidar=chunk["lidar"][t],
                occ=chunk["occ"][t],
                proprio=chunk["proprio"][t],
                action=chunk["action"][t],
                reward=chunk["reward"][t],
                done=bool(chunk["done"][t]) if "done" in chunk else False,
                is_first=bool(chunk.get("is_first", [False] * T)[t]),
            ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of random sequences.

        Returns dict of batched tensors [B, T, ...].
        """
        T = self.seq_len
        B = batch_size
        N = len(self.buffer)

        # Sample random start indices
        starts = self._rng.integers(0, max(1, N - T), size=B)

        def _gather(key, default=0.0):
            """Gather a sequence from each step at given start indices."""
            rows = []
            for s in starts:
                seq = [getattr(self.buffer[int(s) + t], key, default)
                       for t in range(T)]
                # Handle numpy arrays vs scalars
                if isinstance(seq[0], np.ndarray):
                    rows.append(np.stack(seq))
                else:
                    rows.append(np.array(seq, dtype=np.float32))
            return np.stack(rows)

        return {
            "lidar": torch.from_numpy(_gather("lidar")).float(),
            "occ": torch.from_numpy(_gather("occ")).float(),
            "proprio": torch.from_numpy(_gather("proprio")).float(),
            "action": torch.from_numpy(_gather("action")).float(),
            "reward": torch.from_numpy(_gather("reward")).float(),
            "done": torch.from_numpy(_gather("done", 0.0)).float(),
            "is_first": torch.from_numpy(_gather("is_first", 0.0)).float(),
        }


# ============================================================================
# TD-MPC2-style World Model + Actor-Critic
# ============================================================================

class WorldModel(nn.Module):
    """Learned latent dynamics: predicts next latent, reward, and termination.

    Encoder uses the DeepExplorerNetwork's encoder (shared weights). The
    transition model is an MLP: (z, a) -> z'.

    This is a simplified TD-MPC: one-step latent dynamics, no RSSM.
    """
    def __init__(self, encoder: nn.Module, latent_dim: int = 128,
                 action_dim: int = 2):
        super().__init__()
        self.encoder = encoder
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 reward channels
        )
        self.continue_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, lidar, occ, proprio, depth=None):
        _, _ = self.encoder(lidar, occ, proprio, depth)  # We need the fusion features
        # Actually let's just compute the output properly
        return self.encoder.fusion(
            torch.cat([
                self.encoder.lidar_enc(lidar),
                self.encoder.occ_enc(occ),
                self.encoder.proprio_enc(proprio),
            ], dim=-1)
        )

    def forward(self, z, action):
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        reward = self.reward_head(z)
        cont = torch.sigmoid(self.continue_head(z))
        return z_next, reward, cont


class TDMPC2Agent(nn.Module):
    """TD-MPC2-style agent with encoder, dynamics, actor, and value critics.

    Key idea: the actor is trained to maximize Q-values from the learned
    value critics, which are trained with TD(lambda) + target networks.
    """
    def __init__(self, cfg: ExplorerConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.fusion_latent

        # Build encoder (shares features with the rover's network)
        self.encoder = DeepExplorerNetwork(cfg)
        # Freeze the actor and value heads of the encoder — we train separate ones
        for p in self.encoder.parameters():
            p.requires_grad = True  # Fine-tune encoder during training

        # Latent projection (align encoder output to world model latent)
        self.z_proj = nn.Linear(self.latent_dim, self.latent_dim)

        # World model dynamics
        self.dynamics = nn.Sequential(
            nn.Linear(self.latent_dim + cfg.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.action_dim),
            nn.Tanh(),
        )

        # Value critics (ensemble of 2 for TD-MPC style min-Q target)
        self.q1 = nn.Sequential(
            nn.Linear(self.latent_dim + cfg.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.latent_dim + cfg.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q1_target = nn.Sequential(
            nn.Linear(self.latent_dim + cfg.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2_target = nn.Sequential(
            nn.Linear(self.latent_dim + cfg.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._sync_targets()

    def _sync_targets(self, tau: float = 1.0):
        for tp, sp in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
        for tp, sp in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    @torch.no_grad()
    def encode_obs(self, lidar, occ, proprio, depth=None) -> torch.Tensor:
        """Encode observation to latent z."""
        feat = self.encoder.fusion(
            torch.cat([
                self.encoder.lidar_enc(lidar),
                self.encoder.occ_enc(occ),
                self.encoder.proprio_enc(proprio),
            ], dim=-1)
        )
        return self.z_proj(feat)

    def forward(self, lidar, occ, proprio, depth=None):
        """Full forward: observations -> z -> action."""
        z = self.encode_obs(lidar, occ, proprio, depth)
        action = self.actor(z)
        return z, action

    def predict_next(self, z, action):
        """Predict next latent, reward, and continue probability."""
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        reward = self.reward_head(z)
        cont = torch.sigmoid(self.continue_head(z))
        return z_next, reward, cont

    def compute_q(self, z, action):
        """Ensemble Q-values."""
        za = torch.cat([z, action], dim=-1)
        return self.q1(za), self.q2(za)

    def compute_target_q(self, z_next):
        """Target Q at next state (using target networks)."""
        a_next = self.actor(z_next)
        za = torch.cat([z_next, a_next], dim=-1)
        q1 = self.q1_target(za)
        q2 = self.q2_target(za)
        return torch.min(q1, q2)


# ============================================================================
# Trainer
# ============================================================================

class ExplorerTrainer:
    """Main training loop for the Deep Explorer Network."""

    def __init__(self, args):
        self.args = args
        self.device = self._init_device()
        self.cfg = ExplorerConfig()

        # Build agent
        self.agent = TDMPC2Agent(self.cfg).to(self.device)
        print(f"Agent: {sum(p.numel() for p in self.agent.parameters()):,} params")

        # Optimizers
        self.enc_opt = optim.Adam(self.agent.encoder.parameters(), lr=args.enc_lr)
        self.wm_opt = optim.Adam([
            {'params': self.agent.z_proj.parameters()},
            {'params': self.agent.dynamics.parameters()},
            {'params': self.agent.reward_head.parameters()},
            {'params': self.agent.continue_head.parameters()},
        ], lr=args.wm_lr)
        self.actor_opt = optim.Adam(self.agent.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = optim.Adam(
            list(self.agent.q1.parameters()) + list(self.agent.q2.parameters()),
            lr=args.critic_lr)

        # Replay buffer
        self.replay = ReplayBuffer(
            capacity=args.replay_capacity,
            seq_len=args.seq_len,
        )

        # Training state
        self.step = 0
        self.update_count = 0
        self.model_version = 0
        self.discount = args.discount
        self.tau = args.tau  # target network EMA
        self.rho = args.rho  # TD-MPC reward weighting

        # IO
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(args.log_dir) if args.log_dir else None
        self._load_checkpoint()

        # ZMQ
        self.zmq_ctx = None
        self.pull_sock = None

        # Dashboard
        self._start_dashboard()

    def _init_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Device: CUDA {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
            return device
        print("Device: CPU (training will be slow)")
        return torch.device("cpu")

    def _load_checkpoint(self):
        ckpt_path = Path(self.args.checkpoint_dir) / "latest.pt"
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.agent.load_state_dict(sd["agent"], strict=False)
            self.step = sd.get("step", 0)
            self.update_count = sd.get("update_count", 0)
            self.model_version = sd.get("model_version", 0)
            print(f"Loaded checkpoint: {ckpt_path} (step={self.step})")

    def _save_checkpoint(self):
        ckpt_dir = Path(self.args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "agent": self.agent.state_dict(),
            "step": self.step,
            "update_count": self.update_count,
            "model_version": self.model_version,
        }
        latest = ckpt_dir / "latest.pt"
        torch.save(state, latest)
        # Periodic checkpoint
        if self.update_count % 50 == 0:
            periodic = ckpt_dir / f"update_{self.update_count}.pt"
            torch.save(state, periodic)
        print(f"Checkpoint saved: {latest} (v{self.model_version}, step={self.step})")

    def _export(self):
        """Export the encoder + actor to ONNX, then RKNN."""
        from tractor_explorer.convert_to_rknn import export_onnx, convert_to_rknn

        onnx_path = Path(self.args.checkpoint_dir) / "explorer_model.onnx"
        rknn_path = Path(self.args.checkpoint_dir) / "explorer_model.rknn"

        # Build a standalone inference network with current weights
        inference_net = DeepExplorerNetwork(self.cfg)
        inference_net.load_state_dict(self.agent.encoder.state_dict(), strict=False)
        inference_net.eval()

        export_onnx(str(onnx_path), str(onnx_path), self.cfg)
        convert_to_rknn(str(onnx_path), str(rknn_path))

        self.model_version += 1
        print(f"Exported model v{self.model_version}: {rknn_path}")

    def _start_dashboard(self):
        """Minimal HTTP dashboard."""
        port = self.args.dashboard_port
        if port <= 0:
            return
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            trainer = self

            def do_GET(self):
                body = (
                    f'<html><body style="font-family:sans-serif;background:#111;color:#eee">'
                    f'<h1>Deep Explorer Trainer</h1>'
                    f'<p>Model v{self.trainer.model_version}</p>'
                    f'<p>Steps: {self.trainer.step}</p>'
                    f'<p>Updates: {self.trainer.update_count}</p>'
                    f'<p>Replay: {len(self.trainer.replay)} steps</p>'
                    f'</body></html>'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a): pass

        srv = HTTPServer(("0.0.0.0", port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"Dashboard: http://0.0.0.0:{port}")

    # ---- Training core ----

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        """Compute all losses for one training batch."""
        B, T = batch["lidar"].shape[:2]
        lidar = batch["lidar"].to(self.device).reshape(B * T, -1)
        occ = batch["occ"].to(self.device).reshape(B * T, self.cfg.occ_grid_size,
                                                     self.cfg.occ_grid_size)
        proprio = batch["proprio"].to(self.device).reshape(B * T, -1)
        actions = batch["action"].to(self.device).reshape(B * T, -1)
        rewards = batch["reward"].to(self.device).reshape(B * T, -1)
        dones = batch["done"].to(self.device).reshape(B * T, 1)
        is_first = batch["is_first"].to(self.device).reshape(B * T, 1)

        # Encode all observations
        z = self.agent.encode_obs(lidar, occ, proprio)

        # Dynamics prediction (predict next z from current z + action)
        z_next_pred, reward_pred, cont_pred = self.agent.predict_next(z, actions)

        # Shift targets: next latent (from encoder), actual reward
        z_target = torch.roll(z, -1, dims=0)  # z_{t+1}
        reward_target = torch.roll(rewards, -1, dims=0)
        done_target = torch.roll(dones, -1, dims=0)

        # Mask last timestep (no target)
        mask = torch.ones(B * T, 1, device=self.device)
        # Reset at episode boundaries (is_first -> no dynamics prediction)
        mask = mask * (1.0 - is_first)

        # Losses
        # 1. Latent dynamics (MSE on z_next)
        dyn_loss = F.mse_loss(z_next_pred * mask, z_target * mask)

        # 2. Reward prediction (MSE)
        rw_loss = F.mse_loss(reward_pred * mask, reward_target * mask)

        # 3. Continue prediction (BCE with 1 - done)
        cont_target = (1.0 - done_target) * mask
        cont_loss = F.binary_cross_entropy(
            cont_pred * mask, cont_target, reduction="mean")

        # 4. Actor: maximize Q-value (from critic ensemble)
        z_detach = z.detach()
        action_pred = self.agent.actor(z_detach)
        q1, q2 = self.agent.compute_q(z_detach, action_pred)
        q_min = torch.min(q1, q2)
        actor_loss = -q_min.mean()

        # 5. Critic: TD error with target network
        with torch.no_grad():
            z_next = z_target  # use encoded next latent
            q_target = self.agent.compute_target_q(z_next)
            reward_sum = rewards.sum(dim=-1, keepdim=True)  # weighted sum of reward channels
            td_target = reward_sum + self.discount * (1.0 - dones) * q_target
        q1_pred, q2_pred = self.agent.compute_q(z_detach, actions)
        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        return {
            "dynamics": dyn_loss,
            "reward": rw_loss,
            "continue": cont_loss,
            "actor": actor_loss,
            "critic": critic_loss,
            "total": dyn_loss + rw_loss + cont_loss + actor_loss + critic_loss,
        }

    def train_step(self) -> dict:
        if len(self.replay) < self.args.batch_size * self.args.seq_len:
            return {}

        batch = self.replay.sample(self.args.batch_size)
        losses = self.compute_loss(batch)

        # Update encoder + world model
        self.enc_opt.zero_grad()
        self.wm_opt.zero_grad()
        wm_loss = losses["dynamics"] + losses["reward"] + losses["continue"]
        wm_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
        self.enc_opt.step()
        self.wm_opt.step()

        # Update actor
        self.actor_opt.zero_grad()
        losses["actor"].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.max_grad_norm)
        self.actor_opt.step()

        # Update critic
        self.critic_opt.zero_grad()
        losses["critic"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.q1.parameters()) + list(self.agent.q2.parameters()),
            self.args.max_grad_norm)
        self.critic_opt.step()

        # Target network EMA
        self.agent._sync_targets(tau=self.tau)

        self.update_count += 1

        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    # ---- ZMQ data ingestion ----

    async def setup_zmq(self):
        if not HAS_ZMQ or not self.args.port:
            return
        self.zmq_ctx = zmq.asyncio.Context()
        self.pull_sock = self.zmq_ctx.socket(zmq.PULL)
        self.pull_sock.bind(f"tcp://*:{self.args.port}")
        print(f"ZMQ PULL on tcp://*:{self.args.port}")

    async def consume_chunks(self):
        if not self.pull_sock:
            return
        print("Waiting for rover chunks...")
        while True:
            try:
                data = await self.pull_sock.recv()
                if HAS_MSGPACK:
                    chunk = msgpack.unpackb(data)
                else:
                    chunk = np.load(data, allow_pickle=True)
                n = len(chunk["action"])
                self.replay.add_chunk(chunk)
                self.step += n
                print(f"Chunk: {n} steps (replay: {len(self.replay)})")
            except Exception as e:
                print(f"ZMQ error: {e}")
                await asyncio.sleep(1.0)

    # ---- Local data loading ----

    def load_local_chunks(self, data_dir: str):
        path = Path(data_dir)
        npz_files = sorted(path.glob("*.npz"))
        if not npz_files:
            print(f"No chunks found in {data_dir}")
            return
        print(f"Loading {len(npz_files)} chunks from {data_dir}...")
        for f in npz_files:
            try:
                chunk = np.load(str(f))
                n = len(chunk["action"])
                self.replay.add_chunk(chunk)
                self.step += n
            except Exception as e:
                print(f"  Error loading {f}: {e}")
        print(f"Loaded {len(npz_files)} chunks ({self.step} total steps)")

    # ---- Main loop ----

    async def run(self):
        if self.args.data_dir:
            self.load_local_chunks(self.args.data_dir)

        # Start data consumption task
        if self.args.port:
            await self.setup_zmq()
            consume_task = asyncio.create_task(self.consume_chunks())
        else:
            consume_task = None

        print(f"Training loop starting (batch_size={self.args.batch_size}, "
              f"seq_len={self.args.seq_len})")

        last_export = time.time()
        last_save = time.time()

        try:
            while True:
                # Train step
                losses = await asyncio.get_event_loop().run_in_executor(
                    None, self.train_step)
                if losses:
                    if self.writer:
                        for k, v in losses.items():
                            self.writer.add_scalar(f"train/{k}", v, self.update_count)

                    if self.update_count % 10 == 0:
                        print(f"Update {self.update_count}: "
                              f"dyn={losses['dynamics']:.4f} "
                              f"rw={losses['reward']:.4f} "
                              f"actor={losses['actor']:.4f} "
                              f"critic={losses['critic']:.4f}")

                # Export periodically
                if self.args.export and time.time() - last_export > self.args.export_interval:
                    self._export()
                    last_export = time.time()

                # Save checkpoint
                if time.time() - last_save > 60.0:
                    self._save_checkpoint()
                    last_save = time.time()

                # Small sleep so we don't peg the CPU waiting for data
                if not losses:
                    await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self._save_checkpoint()
            if self.args.export:
                self._export()


def main():
    parser = argparse.ArgumentParser(
        description="Deep Explorer Network — Remote Training Server")
    parser.add_argument("--port", type=int, default=0,
                        help="ZMQ port for live rover data (0=disable)")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Load pre-collected chunks from directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--export", action="store_true",
                        help="Periodically export ONNX/RKNN")
    parser.add_argument("--export-interval", type=int, default=300,
                        help="Seconds between exports")
    parser.add_argument("--dashboard-port", type=int, default=8085)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--enc-lr", type=float, default=3e-4)
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    args = parser.parse_args()

    trainer = ExplorerTrainer(args)
    asyncio.run(trainer.run())


if __name__ == "__main__":
    main()
