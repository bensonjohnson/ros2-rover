"""One shared brain, B parallel worlds — the batched training loop.

Mirrors pnn_sim/headless_runner.py tick-for-tick, with every per-env scalar
promoted to a [B] tensor/array and the weight updates batch-averaged. Two
deliberate departures from the single-stream loop:

  - no sequence replay: replay exists to squeeze extra passes out of a
    scarce single stream; with B fresh streams per tick the live gradient
    is already a B-sample estimate (the sleep consolidator still works on
    the logged experience),
  - experience logging covers the first --log-envs streams only (B streams
    at 15 Hz would be gigabytes an hour of redundant jsonl).

Checkpoints use the reference schema — the trained pnn_brain.pt loads on
the rover and in the single-stream sim unchanged.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tractor_bringup.active_inference.pc_world_model import PCConfig
from tractor_bringup.active_inference.efe_actor import ActorConfig
from tractor_bringup.active_inference.slow_layer import SlowLayerConfig

from ..rover import RoverConfig
from ..safety_gate import GateConfig
from .model import BatchedPCWorldModel, BatchedEFEActor, BatchedSlowLayer
from .env import BatchedEnv, BatchedGate, batched_preprocess
from .place import BatchedPlaceMemory


@dataclass
class BatchedTrainConfig:
    envs: int = 256
    device: str = "cuda"
    lr_scale: float = 1.0           # batched gradient has ~1/sqrt(B) the
                                    # variance; values of 2-4 are worth trying
    # Brain — identical defaults to the single-stream sim / launch file.
    num_bins: int = 72
    max_range: float = 5.0
    latent_dim: int = 64
    ensemble_size: int = 5
    n_infer_iters: int = 24
    proprio_precision: float = 4.0
    novelty_precision: float = 6.0
    action_scale: float = 0.6
    action_smoothing: float = 0.4
    forward_bias: float = 0.3
    target_wl: float = 0.65
    target_wr: float = 0.65
    target_yaw: float = 0.5
    horizon: int = 8
    action_persist: int = 5
    target_novelty: float = 0.8
    novelty_pref_weight: float = 2.0
    hold_pref_weight: float = 2.0
    # Confidence gate: the epistemic (curiosity) term is scaled by
    # min(1, epi_max / epi_floor), so once the ensemble agrees (disagreement
    # < epi_floor) curiosity gates OFF and the brain coasts on pragmatic
    # preference alone — the dark-room trap in a familiar room. LOWER this to
    # keep curiosity alive longer for a confident brain.
    epi_floor: float = 0.02
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
    torch_threads: int = 0

    # Motor babbling: a decaying forced-exploration schedule that breaks the
    # chicken-and-egg of model-based exploration (the brain won't move to
    # learn that moving is informative until it already knows that). Each
    # re-decision, an env babbles with prob eps(t) = eps0*(1 - step/decay),
    # executing a forward arc (or a spin when the gate blocks) instead of the
    # actor's pick — so the transition model and the place memory see
    # action-rich, varied-room data. Training-only (gated by `learn`); never
    # affects eval or the rover. 0 = off.
    babble_eps0: float = 0.4
    babble_decay_ticks: int = 30_000

    switch_world_every: int = 27_000   # per env, staggered across the batch
    rover: RoverConfig = field(default_factory=RoverConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    out_dir: str = "sim_out_batched"
    load_path: str = ""
    save_interval_s: float = 60.0
    snapshot_every: int = 0         # batch ticks; 0 = no numbered snapshots
    log_envs: int = 1
    experience_log_max_mb: float = 256.0


class BatchedTrainer:
    N_PROPRIO = 8
    N_INTERO = 2

    def __init__(self, cfg: BatchedTrainConfig):
        self.cfg = cfg
        B = cfg.envs
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.model_path = os.path.join(cfg.out_dir, "pnn_brain.pt")
        self.slow_model_path = os.path.join(cfg.out_dir, "pnn_brain_slow.pt")

        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        self.obs_dim = cfg.num_bins + self.N_PROPRIO + self.N_INTERO
        self.tick_period = 1.0 / cfg.control_rate_hz
        self.sim_time = 0.0

        self.env = BatchedEnv(B, cfg.rover, seed=cfg.seed, device=cfg.device)
        self.gate = BatchedGate(cfg.gate, B, time_fn=lambda: self.sim_time,
                                device=cfg.device)

        self.model = BatchedPCWorldModel(PCConfig(
            obs_dim=self.obs_dim, latent_dim=cfg.latent_dim,
            ensemble_size=cfg.ensemble_size, n_infer_iters=cfg.n_infer_iters,
            n_proprio=self.N_PROPRIO, precision_proprio=cfg.proprio_precision,
            n_intero=self.N_INTERO, precision_intero=cfg.novelty_precision,
            seed=cfg.seed,
        ), batch=B, device=cfg.device)
        self.actor = BatchedEFEActor(ActorConfig(
            action_dim=2, pragmatic_weight=cfg.forward_bias,
            target_wl=cfg.target_wl, target_wr=cfg.target_wr,
            target_yaw=cfg.target_yaw, horizon=cfg.horizon,
            num_bins=cfg.num_bins, use_proprio=True, n_intero=self.N_INTERO,
            target_novelty=cfg.target_novelty,
            novelty_pref_weight=cfg.novelty_pref_weight,
            hold_pref_weight=cfg.hold_pref_weight, epi_floor=cfg.epi_floor,
            slow_prior_weight=cfg.slow_prior_weight, seed=cfg.seed,
        ), batch=B, device=cfg.device)
        self.slow: BatchedSlowLayer | None = None
        if cfg.slow_enabled:
            self.slow = BatchedSlowLayer(SlowLayerConfig(
                fast_latent_dim=cfg.latent_dim,
                latent_dim=cfg.slow_latent_dim,
                period_ticks=cfg.slow_period_ticks,
                horizon=cfg.slow_horizon,
                warmup_ticks=cfg.slow_warmup_ticks,
                target_novelty=cfg.target_novelty,
                novelty_pref_weight=cfg.novelty_pref_weight,
                seed=cfg.seed,
            ), batch=B, device=cfg.device)

        self.place = BatchedPlaceMemory(B, device=cfg.device)
        self.nov_ema = torch.ones(B, device=self.device)
        self._babble_rng = np.random.default_rng(cfg.seed + 12_345)

        self.last_action = torch.zeros(B, 2, device=self.device)
        self.exec_action = np.zeros((B, 2), dtype=np.float32)
        self.held_raw = np.zeros((B, 2), dtype=np.float32)
        self.persist = np.zeros(B, dtype=int)
        self._step = 0
        self._last_save = time.time()
        self._collisions = 0
        self._F = self._err = 0.0
        self._exp_files = [None] * min(cfg.log_envs, B)

        if cfg.load_path:
            path = os.path.expanduser(cfg.load_path)
            sd = torch.load(path, map_location="cpu", weights_only=False)
            if sd["W_o"].shape[0] != self.obs_dim:
                raise SystemExit(f"Checkpoint obs_dim={sd['W_o'].shape[0]} "
                                 f"!= {self.obs_dim}")
            self.model.load_state_dict(sd)
            print(f"Loaded brain from {path}")
            if self.slow is not None:
                ok, _ = self.slow.load(self.slow_model_path)
                if ok:
                    print(f"Loaded slow layer from {self.slow_model_path}")

    # ------------------------------------------------------------------

    def save(self, force: bool = False):
        if not force and time.time() - self._last_save < self.cfg.save_interval_s:
            return
        self._last_save = time.time()
        sd = self.model.state_dict()
        sd["action_scale"] = self.cfg.action_scale
        tmp = self.model_path + ".tmp"
        torch.save(sd, tmp)
        os.replace(tmp, self.model_path)
        if self.slow is not None:
            self.slow.save(self.slow_model_path)

    def snapshot(self):
        """Numbered checkpoint pair for later frozen-eval comparison."""
        snap_dir = os.path.join(self.cfg.out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        sd = self.model.state_dict()
        sd["action_scale"] = self.cfg.action_scale
        path = os.path.join(snap_dir, f"pnn_brain_step{self._step:08d}.pt")
        torch.save(sd, path)
        if self.slow is not None:
            self.slow.save(os.path.join(
                snap_dir, f"pnn_brain_slow_step{self._step:08d}.pt"))
        return path

    def _log_experience(self, obs_np: np.ndarray, act_prev: np.ndarray):
        for i in range(len(self._exp_files)):
            if self._exp_files[i] is None:
                suffix = "" if i == 0 else f"_env{i:02d}"
                self._exp_files[i] = open(
                    os.path.join(self.cfg.out_dir,
                                 f"pnn_experience{suffix}.jsonl"), "a")
            f = self._exp_files[i]
            f.write(json.dumps({
                "obs": [round(float(x), 5) for x in obs_np[i].tolist()],
                "act": [round(float(x), 5) for x in act_prev[i].tolist()],
            }) + "\n")
            if self._step % 50 == 0:
                f.flush()

    # ------------------------------------------------------------------

    def tick(self):
        cfg = self.cfg
        B = cfg.envs

        # Staggered world switching: env b swaps houses every
        # switch_world_every ticks, offset so the batch never swaps at once.
        if cfg.switch_world_every > 0 and self._step > 0:
            stride = max(1, cfg.switch_world_every // B)
            due = (self._step + np.arange(B) * stride) \
                % cfg.switch_world_every == 0
            if due.any():
                self.env.switch_world(due)
                idx = torch.from_numpy(np.flatnonzero(due)).to(self.device)
                self.place.clear(idx)
                self.nov_ema[idx] = 1.0
                if self.slow is not None:
                    self.slow.reset_env(idx)

        # --- sense (device-resident: raycast, binning, gate) ---
        ranges = self.env.scan()                       # [B, beams] device
        scan72 = batched_preprocess(ranges, self.env.angle_min,
                                    self.env.angle_increment,
                                    num_bins=cfg.num_bins,
                                    max_range=cfg.max_range)
        self.gate.process_scan(ranges, self.env.angle_min,
                               self.env.angle_increment)
        hold = self.gate.front_blocked                 # [B] bool, device
        hold_np = hold.cpu().numpy()

        # --- interoceptive novelty (batched place memory, on device) ---
        nov_raw = self.place.update(scan72, self.tick_period)
        alpha = min(1.0, self.tick_period
                    / max(cfg.novelty_ema_tau_s, self.tick_period))
        self.nov_ema += alpha * (nov_raw - self.nov_ema)
        nov = self.nov_ema

        # --- observation [B, obs_dim], reference layout/normalization ---
        e = self.env
        proprio = torch.stack([
            (0.5 + 0.5 * e.wheel_l / 8.0).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.wheel_r / 8.0).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.yaw_rate / 2.5).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.accel[:, 0] / 19.6).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.accel[:, 1] / 19.6).clamp(0.0, 1.0),
            (0.5 + 0.5 * e.accel[:, 2] / 19.6).clamp(0.0, 1.0),
        ], dim=1)
        intero = torch.stack([hold.float(), nov], dim=1)
        o_t = torch.cat([scan72, proprio, intero], dim=1)

        act_prev = self.last_action.cpu().numpy()
        if self.cfg.log_envs > 0:
            self._log_experience(
                o_t[:len(self._exp_files)].cpu().numpy(), act_prev)

        # 1. Infer (slow top-down prior masked by per-env warmup).
        td = None
        if self.slow is not None and self.slow.td_target is not None:
            td = self.slow.td_target * self.slow.warm.unsqueeze(1)
        z, F, obs_err = self.model.infer(
            o_t, self.last_action, td_target=td,
            td_precision=cfg.td_precision if td is not None else 0.0)
        self._F = float(F.mean())
        self._err = float(obs_err.mean())
        self.last_obs_err = obs_err          # [B], for per-env eval metrics

        # 2. Learn — ONE shared update from B streams. learn() advances the
        #    recurrent state, so frozen evals must advance it explicitly.
        if cfg.learn:
            self.model.learn(z, self.last_action, o_t,
                             lr_scale=cfg.lr_scale)
        else:
            self.model.z_prev = z.detach().clone()

        # 3. Action per env: persist / gate-drop / batched select.
        held_fwd = self.held_raw.sum(axis=1) > 0.0
        self.persist[hold_np & (self.persist > 0) & held_fwd] = 0
        redecide = self.persist <= 0

        slow_act = slow_warm = None
        if self.slow is not None:
            slow_act = self.slow.macro_action
            slow_warm = self.slow.warm
        action, info = self.actor.select(
            self.model, z,
            prev_action=torch.from_numpy(self.held_raw).to(self.device),
            slow_action=slow_act, slow_warm=slow_warm,
            forward_blocked=hold)
        action_np = action.cpu().numpy()

        # Motor babbling: override the actor's pick for a decaying fraction of
        # re-deciding envs (training only). Forward arc normally, spin when
        # the gate blocks forward — coverage that feeds varied places/actions
        # into learning. The executed (babbled) action is what flows into
        # last_action and thus into the next tick's learn(), so the transition
        # model sees the true action->effect.
        if cfg.learn and cfg.babble_eps0 > 0.0:
            eps = cfg.babble_eps0 * max(
                0.0, 1.0 - self._step / max(1, cfg.babble_decay_ticks))
            if eps > 0.0:
                rng = self._babble_rng
                bm = redecide & (rng.random(B) < eps)
                if bm.any():
                    f = rng.uniform(0.4, 1.0, B)
                    d = rng.uniform(-0.4, 0.4, B)
                    fwd = np.stack([f + d, f - d], axis=1)
                    s = rng.uniform(0.5, 1.0, B) \
                        * np.where(rng.random(B) < 0.5, 1.0, -1.0)
                    spin = np.stack([s, -s], axis=1)
                    bab = np.where(hold_np[:, None], spin, fwd)
                    action_np = action_np.copy()
                    action_np[bm] = np.clip(bab[bm], -1.0, 1.0)

        self.held_raw = np.where(redecide[:, None], action_np, self.held_raw)
        self.persist = np.where(redecide,
                                max(1, cfg.action_persist) - 1,
                                self.persist - 1)

        # 4. Smooth + scale + gate -> step the worlds.
        a = cfg.action_smoothing * self.held_raw \
            + (1.0 - cfg.action_smoothing) * self.exec_action
        self.exec_action = a
        out = np.clip(a * cfg.action_scale, -1.0, 1.0)
        gated = self.gate.gate(
            torch.from_numpy(out.astype(np.float32)).to(self.device))
        self.env.step(gated, self.tick_period)
        self._collisions += int(self.env.collided.sum())

        self.last_action = torch.from_numpy(a).to(self.device)

        # Slow layer.
        if self.slow is not None:
            self.slow.accumulate(
                torch.tanh(z),
                scan72.mean(dim=1),
                (obs_err / math.sqrt(self.obs_dim)).clamp(max=1.0),
                nov,
                self.last_action)
            if self.slow.ready():
                self.slow.tick(learn=cfg.learn, lr_scale=cfg.lr_scale)

        self._step += 1
        self.sim_time += self.tick_period
        if self.cfg.snapshot_every > 0 \
                and self._step % self.cfg.snapshot_every == 0:
            self.snapshot()
        self.save()

    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "step": self._step,
            "env_ticks": self._step * self.cfg.envs,
            "sim_hours_total": self._step * self.cfg.envs
            * self.tick_period / 3600.0,
            "F": self._F,
            "obs_err": self._err,
            "novelty": float(self.nov_ema.mean()),
            "gate_stops": self.gate.stops,
            "collisions": self._collisions,
        }

    def close(self):
        self.save(force=True)
        for f in self._exp_files:
            if f is not None:
                f.close()
