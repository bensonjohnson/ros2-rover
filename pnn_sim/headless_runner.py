"""Headless brain-in-sim loop — the ROS runner's _control_step, ROS-free.

One process, one loop: sim step -> sensors -> the exact tick logic of
pc_active_inference_runner (novelty EMA, obs layout, infer/learn, action
persistence, the safety-hold persist drop, smoothing, scaling, gate, slow
layer, replay, experience log, checkpointing). The brain modules are imported
unmodified from tractor_bringup.active_inference, so a checkpoint trained
here loads on the rover byte-for-byte (same state_dict schema, including
action_scale).

Anything the runner does that has no sim counterpart is intentionally
absent: teleop (no human), lift detection (no hands — world switches call
the same reset path explicitly), dashboard, ROS diagnostics.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.pc_world_model import PCWorldModel, PCConfig
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig
from tractor_bringup.active_inference.replay import SequenceReplay
from tractor_bringup.active_inference.place_memory import PlaceMemory
from tractor_bringup.active_inference.slow_layer import SlowLayer, SlowLayerConfig

from .world import make_house
from .rover import SimRover, RoverConfig
from .safety_gate import SimSafetyGate, GateConfig


@dataclass
class SimBrainConfig:
    # Brain — defaults mirror pc_active_inference.launch.py / the runner.
    num_bins: int = 72
    max_range: float = 5.0
    latent_dim: int = 64
    ensemble_size: int = 5
    n_infer_iters: int = 24
    replay_infer_iters: int = 10
    proprio_precision: float = 4.0
    novelty_precision: float = 6.0
    replay_capacity: int = 4000
    replay_passes: int = 1
    replay_seq_len: int = 16
    replay_burn_in: int = 4
    replay_min: int = 256
    action_scale: float = 0.6
    action_smoothing: float = 0.4
    forward_bias: float = 0.3
    pragmatic_weight: float = 0.4
    target_wl: float = 0.65
    target_wr: float = 0.65
    target_yaw: float = 0.5
    horizon: int = 8
    action_persist: int = 5
    target_novelty: float = 0.8
    novelty_pref_weight: float = 2.0
    hold_pref_weight: float = 2.0
    novelty_ema_tau_s: float = 1.0
    slow_enabled: bool = True
    slow_latent_dim: int = 16
    slow_period_ticks: int = 15
    slow_horizon: int = 8
    slow_warmup_ticks: int = 30
    td_precision: float = 0.2
    slow_prior_weight: float = 0.25
    control_rate_hz: float = 15.0
    learn: bool = True
    seed: int = 0
    device: str = "cpu"
    torch_threads: int = 0          # 0 = leave torch's default

    # Sim
    switch_world_every: int = 0     # ticks; 0 = single world forever
    rover: RoverConfig = field(default_factory=RoverConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    # Persistence
    out_dir: str = "sim_out"
    model_path: str = ""            # default: <out_dir>/pnn_brain.pt
    slow_model_path: str = ""       # default: <out_dir>/pnn_brain_slow.pt
    load_path: str = ""             # optional existing checkpoint to start from
    save_interval_s: float = 60.0   # wall time
    log_experience: bool = True
    experience_log_max_mb: float = 256.0


class SimBrainRunner:
    N_PROPRIO = 8
    N_INTERO = 2

    def __init__(self, cfg: SimBrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.model_path = cfg.model_path or os.path.join(cfg.out_dir, "pnn_brain.pt")
        self.slow_model_path = cfg.slow_model_path \
            or os.path.join(cfg.out_dir, "pnn_brain_slow.pt")
        self.exp_log_path = os.path.join(cfg.out_dir, "pnn_experience.jsonl")

        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)
        self.device = torch.device(cfg.device)

        self.obs_dim = cfg.num_bins + self.N_PROPRIO + self.N_INTERO
        self.tick_period = 1.0 / cfg.control_rate_hz
        self.sim_time = 0.0           # the accelerated clock everything sees

        # --- sim ---
        self.world_rng = np.random.default_rng(cfg.seed)
        self.rover = SimRover(make_house(self.world_rng), cfg.rover,
                              rng=np.random.default_rng(cfg.seed + 1))
        self.gate = SimSafetyGate(cfg.gate, time_fn=lambda: self.sim_time)

        # --- brain (identical construction to the ROS runner) ---
        self.model = PCWorldModel(PCConfig(
            obs_dim=self.obs_dim,
            latent_dim=cfg.latent_dim,
            ensemble_size=cfg.ensemble_size,
            n_infer_iters=cfg.n_infer_iters,
            n_proprio=self.N_PROPRIO,
            precision_proprio=cfg.proprio_precision,
            n_intero=self.N_INTERO,
            precision_intero=cfg.novelty_precision,
            seed=cfg.seed,
        ), device=cfg.device)
        self.actor = EFEActor(ActorConfig(
            action_dim=2,
            pragmatic_weight=(cfg.forward_bias if cfg.forward_bias > 0.0
                              else cfg.pragmatic_weight),
            target_wl=cfg.target_wl, target_wr=cfg.target_wr,
            target_yaw=cfg.target_yaw,
            horizon=cfg.horizon, num_bins=cfg.num_bins,
            use_proprio=True, n_intero=self.N_INTERO,
            target_novelty=cfg.target_novelty,
            novelty_pref_weight=cfg.novelty_pref_weight,
            hold_pref_weight=cfg.hold_pref_weight,
            slow_prior_weight=cfg.slow_prior_weight,
            seed=cfg.seed,
        ), device=cfg.device)
        self.slow: SlowLayer | None = None
        if cfg.slow_enabled:
            self.slow = SlowLayer(SlowLayerConfig(
                fast_latent_dim=cfg.latent_dim,
                latent_dim=cfg.slow_latent_dim,
                period_ticks=cfg.slow_period_ticks,
                horizon=cfg.slow_horizon,
                warmup_ticks=cfg.slow_warmup_ticks,
                target_novelty=cfg.target_novelty,
                novelty_pref_weight=cfg.novelty_pref_weight,
                seed=cfg.seed,
            ))

        self.place_memory = PlaceMemory(time_fn=lambda: self.sim_time)
        self._nov_ema = 1.0

        self.replay = SequenceReplay(capacity=cfg.replay_capacity,
                                     obs_dim=self.obs_dim, action_dim=2,
                                     seed=cfg.seed)

        self.last_action = torch.zeros(2, device=self.device)
        self.exec_action = np.zeros(2)
        self._held_raw = np.zeros(2, dtype=np.float32)
        self._persist_ctr = 0
        self._step = 0
        self._last_save = time.time()
        self._info = {"epistemic": 0.0, "epistemic_max": 0.0,
                      "pragmatic": 0.0}
        self._last_F = 0.0
        self._last_err = 0.0
        self._collisions = 0
        self._dist = 0.0
        self._exp_file = None
        self._exp_bytes = 0

        self._maybe_load()

    # ------------------------------------------------------------------

    def _maybe_load(self):
        path = self.cfg.load_path or (
            self.model_path if os.path.exists(self.model_path) else "")
        if not path or not os.path.exists(path):
            return
        sd = torch.load(path, map_location="cpu", weights_only=False)
        saved_dim = sd["W_o"].shape[0]
        if saved_dim != self.obs_dim:
            raise SystemExit(
                f"Checkpoint {path} has obs_dim={saved_dim}, sim builds "
                f"{self.obs_dim} — observation layouts must match")
        self.model.load_state_dict(sd)
        self._move_model_to_device()
        saved_scale = sd.get("action_scale")
        if saved_scale is not None \
                and abs(saved_scale - self.cfg.action_scale) > 1e-6:
            print(f"WARNING: checkpoint action_scale={saved_scale:.3f} vs "
                  f"sim {self.cfg.action_scale:.3f} — dynamics will re-adapt")
        print(f"Loaded brain from {path}")
        if self.slow is not None:
            ok, reason = self.slow.load(self.slow_model_path)
            if ok:
                print(f"Loaded slow layer from {self.slow_model_path}")

    def _move_model_to_device(self):
        m = self.model
        m.W_o = m.W_o.to(self.device)
        m.b_o = m.b_o.to(self.device)
        m.W_z = [w.to(self.device) for w in m.W_z]
        m.b_z = [b.to(self.device) for b in m.b_z]
        m.pi_o = m.pi_o.to(self.device)
        m.pi_prior = m.pi_prior.to(self.device)
        m.z_prev = m.z_prev.to(self.device)
        if m._err_sq_ema is not None:
            m._err_sq_ema = m._err_sq_ema.to(self.device)

    def save(self, force: bool = False):
        if not force and time.time() - self._last_save < self.cfg.save_interval_s:
            return
        self._last_save = time.time()
        sd = self.model.state_dict()
        # Save CPU tensors so the rover (and a CPU sim) load without remap.
        sd = {k: ([t.cpu() for t in v] if isinstance(v, list)
                  else v.cpu() if torch.is_tensor(v) else v)
              for k, v in sd.items()}
        sd["action_scale"] = self.cfg.action_scale
        tmp = self.model_path + ".tmp"
        torch.save(sd, tmp)
        os.replace(tmp, self.model_path)
        if self.slow is not None:
            self.slow.save(self.slow_model_path)

    # ------------------------------------------------------------------

    def _log_experience(self, obs_np: np.ndarray, action_prev_np: np.ndarray):
        if not self.cfg.log_experience:
            return
        if self._exp_file is None:
            self._exp_file = open(self.exp_log_path, "a")
            self._exp_bytes = self._exp_file.tell()
        line = json.dumps({
            "obs": [round(float(x), 5) for x in obs_np.tolist()],
            "act": [round(float(x), 5) for x in action_prev_np.tolist()],
        }) + "\n"
        self._exp_bytes += self._exp_file.write(line)
        if self._step % 50 == 0:
            self._exp_file.flush()
        if self._exp_bytes >= self.cfg.experience_log_max_mb * 1024 * 1024:
            self._exp_file.close()
            ts = time.strftime("%Y%m%d_%H%M%S")
            os.rename(self.exp_log_path,
                      self.exp_log_path.replace(".jsonl", f"_part_{ts}.jsonl"))
            self._exp_file = open(self.exp_log_path, "a")
            self._exp_bytes = 0

    def _switch_world(self):
        """New house — the sim's version of being carried somewhere new.

        Mirrors the runner's lift path: place memory cleared, slow context
        dropped, learned weights kept.
        """
        self.rover.set_world(make_house(self.world_rng))
        self.place_memory.clear()
        self._nov_ema = 1.0
        if self.slow is not None:
            self.slow.reset_state()

    # ------------------------------------------------------------------

    def tick(self):
        cfg = self.cfg
        if cfg.switch_world_every > 0 and self._step > 0 \
                and self._step % cfg.switch_world_every == 0:
            self._switch_world()

        # --- sense (lidar rev + gate state update, as on the rover) ---
        ranges, amin, ainc = self.rover.scan()
        scan72 = preprocess_scan(ranges, amin, ainc,
                                 num_bins=cfg.num_bins, max_range=cfg.max_range)
        self.gate.process_scan(ranges, amin, ainc)
        safety_hold = self.gate.front_blocked

        # --- interoceptive novelty (runner step 0) ---
        nov_raw = self.place_memory.update(scan72)
        alpha = min(1.0, self.tick_period
                    / max(cfg.novelty_ema_tau_s, self.tick_period))
        self._nov_ema += alpha * (nov_raw - self._nov_ema)

        # --- observation vector, identical layout to the runner ---
        rc = self.rover.cfg
        max_wheel_vel, max_yaw_rate, max_accel = 8.0, 2.5, 19.6
        wl = np.clip(0.5 + 0.5 * self.rover.wheel_l / max_wheel_vel, 0.0, 1.0)
        wr = np.clip(0.5 + 0.5 * self.rover.wheel_r / max_wheel_vel, 0.0, 1.0)
        # Sim has no roll/pitch dynamics: rates are pure sensor noise, like a
        # rover on flat floor.
        noise = self.rover.rng.normal
        roll = np.clip(0.5 + 0.5 * noise(0, rc.gyro_noise_std) / max_yaw_rate, 0.0, 1.0)
        pitch = np.clip(0.5 + 0.5 * noise(0, rc.gyro_noise_std) / max_yaw_rate, 0.0, 1.0)
        yaw = np.clip(0.5 + 0.5 * self.rover.yaw_rate / max_yaw_rate, 0.0, 1.0)
        ax = np.clip(0.5 + 0.5 * self.rover.accel[0] / max_accel, 0.0, 1.0)
        ay = np.clip(0.5 + 0.5 * self.rover.accel[1] / max_accel, 0.0, 1.0)
        az = np.clip(0.5 + 0.5 * self.rover.accel[2] / max_accel, 0.0, 1.0)
        proprio = np.array([wl, wr, roll, pitch, yaw, ax, ay, az],
                           dtype=np.float32)
        intero = np.array([1.0 if safety_hold else 0.0, self._nov_ema],
                          dtype=np.float32)
        obs_np = np.concatenate([scan72, proprio, intero])

        o_t = torch.from_numpy(obs_np).to(self.device)
        action_prev_np = self.last_action.cpu().numpy().copy()
        self._log_experience(obs_np, action_prev_np)

        # 1. Infer under the slow layer's top-down prior.
        td = self.slow.td_target if self.slow is not None else None
        z, F, obs_err = self.model.infer(
            o_t, self.last_action,
            td_target=(td.to(self.device) if td is not None else None),
            td_precision=cfg.td_precision)
        self._last_F, self._last_err = F, obs_err

        # 2. Learn.
        if cfg.learn:
            self.model.learn(z, self.last_action, o_t)

        # 3. Action: persist / gate-drop / select — runner logic verbatim
        #    (minus teleop: there is no human in the loop here).
        if (safety_hold and self._persist_ctr > 0
                and float(self._held_raw[0] + self._held_raw[1]) > 0.0):
            self._persist_ctr = 0
        if self._persist_ctr <= 0:
            action, info = self.actor.select(
                self.model, z, prev_action=self._held_raw,
                slow_action=(self.slow.macro_action
                             if self.slow is not None else None),
                forward_blocked=safety_hold)
            self._held_raw = action.cpu().numpy()
            self._info = info
            self._persist_ctr = max(1, cfg.action_persist) - 1
        else:
            self._persist_ctr -= 1

        # 4. Smooth + scale + gate -> "motors".
        a = cfg.action_smoothing * self._held_raw \
            + (1.0 - cfg.action_smoothing) * self.exec_action
        self.exec_action = a
        out = np.clip(a * cfg.action_scale, -1.0, 1.0)
        gl, gr = self.gate.gate(float(out[0]), float(out[1]))

        px, py = self.rover.x, self.rover.y
        self.rover.step(gl, gr, self.tick_period)
        self._dist += math.hypot(self.rover.x - px, self.rover.y - py)
        if self.rover.collided:
            self._collisions += 1

        self.last_action = torch.from_numpy(a.astype(np.float32)).to(self.device)

        # Slow layer accumulation + tick (CPU tensors, like the runner).
        if self.slow is not None:
            self.slow.accumulate(
                torch.tanh(z).cpu(),
                float(np.mean(scan72)),
                min(1.0, obs_err / math.sqrt(self.obs_dim)),
                self._nov_ema,
                a)
            if self.slow.ready():
                self.slow.tick(learn=cfg.learn)

        # Replay.
        self.replay.append(obs_np, action_prev_np)
        if cfg.learn and cfg.replay_passes > 0 \
                and len(self.replay) >= cfg.replay_min:
            self._replay_step()

        self._step += 1
        self.sim_time += self.tick_period
        self.save()

    def _replay_step(self):
        cfg = self.cfg
        L = cfg.replay_seq_len
        D = cfg.latent_dim
        for _ in range(cfg.replay_passes):
            sample = self.replay.sample_sequence(L)
            if sample is None:
                return
            obs_seq, act_seq = sample
            zp = torch.zeros(D, device=self.device)
            for t in range(L):
                o = torch.from_numpy(obs_seq[t]).to(self.device)
                a = torch.from_numpy(act_seq[t]).to(self.device)
                z, _, _ = self.model.infer(o, a, z_prev=zp,
                                           n_iters=cfg.replay_infer_iters)
                if t >= cfg.replay_burn_in:
                    self.model.learn(z, a, o, z_prev=zp, advance=False,
                                     update_precision=False)
                zp = z

    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "step": self._step,
            "sim_hours": self.sim_time / 3600.0,
            "F": self._last_F,
            "obs_err": self._last_err,
            "epi": self._info.get("epistemic", 0.0),
            "novelty": self._nov_ema,
            "places": self.place_memory.n_places(),
            "gate_stops": self.gate.stops,
            "collisions": self._collisions,
            "dist_m": self._dist,
        }

    def close(self):
        self.save(force=True)
        if self._exp_file is not None:
            self._exp_file.close()
