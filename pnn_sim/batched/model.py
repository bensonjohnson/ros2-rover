"""Batched predictive-coding world model, EFE actor, and slow layer.

Same generative model, inference settle, local learning rules, and EFE
scoring as the reference implementation (pc_world_model.py, efe_actor.py,
slow_layer.py) with a leading batch dimension B:

  - weights are SHARED across the batch (one brain),
  - recurrent state z_prev, precision context, actor persistence etc. are
    per-env,
  - learn() averages the local updates over the batch — the mean of the
    per-sample outer products, i.e. the same update the reference applies,
    estimated from B samples instead of 1.

Checkpoint compatibility: state_dict()/load_state_dict() use the reference
schema (W_o, b_o, W_z list, b_z list, z_prev[D], cfg: PCConfig, pi_o,
err_sq_ema), so brains move freely between rover, single-stream sim, and
this trainer. batched/verify.py asserts numerical equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from tractor_bringup.active_inference.pc_world_model import PCConfig
from tractor_bringup.active_inference.efe_actor import ActorConfig, _STRUCTURED


class BatchedPCWorldModel:
    def __init__(self, cfg: PCConfig, batch: int, device: str = "cpu"):
        self.cfg = cfg
        self.B = batch
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)

        D, O, A, M = cfg.latent_dim, cfg.obs_dim, cfg.action_dim, cfg.ensemble_size

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(self.device)

        # Same init draw order as the reference -> same starting weights.
        self.W_o = randn(O, D, scale=0.1)
        self.b_o = torch.zeros(O, device=self.device)
        self.W_z = [randn(D, D + A, scale=0.1) for _ in range(M)]
        self.b_z = [torch.zeros(D, device=self.device) for _ in range(M)]
        # Stacked views for batched ops, refreshed after any weight mutation.
        self._restack()

        n_prop = int(getattr(cfg, "n_proprio", 0))
        pi_prop = float(getattr(cfg, "precision_proprio", cfg.precision_obs))
        n_int = int(getattr(cfg, "n_intero", 0))
        pi_int = float(getattr(cfg, "precision_intero", cfg.precision_obs))
        self.pi_o = torch.full((O,), float(cfg.precision_obs), device=self.device)
        if n_prop > 0:
            self.pi_o[O - n_int - n_prop:O - n_int] = pi_prop
        if n_int > 0:
            self.pi_o[O - n_int:] = pi_int
        self.pi_prior = self.pi_o.clone()
        self._err_sq_ema: torch.Tensor | None = None
        self.pi_z = cfg.precision_z

        self.z_prev = torch.zeros(self.B, D, device=self.device)
        self._g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    def _restack(self):
        self.Wz_stack = torch.stack(self.W_z)            # [M, D, D+A]
        self.bz_stack = torch.stack(self.b_z)            # [M, D]

    # ---- generative pieces (batched) ----------------------------------

    def _decode(self, z: torch.Tensor):
        """z [..., D] -> (o_hat [..., O], s [..., D])"""
        s = torch.tanh(z)
        o_hat = torch.sigmoid(s @ self.W_o.t() + self.b_o)
        return o_hat, s

    def _trans_input(self, z_prev: torch.Tensor, action: torch.Tensor):
        return torch.tanh(torch.cat([z_prev, action], dim=-1))   # [..., D+A]

    def _member_preds(self, s_in: torch.Tensor) -> torch.Tensor:
        """s_in [..., D+A] -> per-member predictions [M, ..., D]"""
        return torch.einsum("mdi,...i->m...d", self.Wz_stack, s_in) \
            + self.bz_stack.view(self.cfg.ensemble_size,
                                 *([1] * (s_in.dim() - 1)), -1)

    def _prior_mean(self, s_in: torch.Tensor):
        return self._member_preds(s_in).mean(dim=0)

    @torch.no_grad()
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)[0]

    # ---- inference -----------------------------------------------------

    @torch.no_grad()
    def infer(self, o_t: torch.Tensor, action_prev: torch.Tensor,
              z_prev: torch.Tensor | None = None,
              n_iters: int | None = None,
              td_target: torch.Tensor | None = None,
              td_precision: float = 0.0):
        """Batched settle. o_t [B,O], action_prev [B,A] ->
        (z [B,D], F [B], obs_err [B]) — same math as the reference per row."""
        zp = self.z_prev if z_prev is None else z_prev
        s_in = self._trans_input(zp, action_prev)
        z_hat = self._prior_mean(s_in)
        z = z_hat.clone()

        use_td = td_target is not None and td_precision > 0.0
        iters = self.cfg.n_infer_iters if n_iters is None else n_iters
        for _ in range(iters):
            o_hat, s = self._decode(z)
            e_o = o_t - o_hat
            e_z = z - z_hat
            dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
            g_obs = (dF_du @ self.W_o) * (1.0 - s * s)
            g_z = g_obs + self.pi_z * e_z
            if use_td:
                g_z = g_z + td_precision * (s - td_target) * (1.0 - s * s)
            z = z - self.cfg.infer_lr * g_z

        o_hat, s = self._decode(z)
        e_o = o_t - o_hat
        e_z = z - z_hat
        F = 0.5 * ((self.pi_o * e_o) * e_o).sum(dim=-1) \
            + 0.5 * self.pi_z * (e_z * e_z).sum(dim=-1)
        if use_td:
            e_td = s - td_target
            F = F + 0.5 * td_precision * (e_td * e_td).sum(dim=-1)
        return z, F, torch.linalg.norm(e_o, dim=-1)

    # ---- learning (batch-averaged local updates) ------------------------

    @torch.no_grad()
    def learn(self, z_settled: torch.Tensor, action_prev: torch.Tensor,
              o_t: torch.Tensor, z_prev: torch.Tensor | None = None,
              advance: bool = True, update_precision: bool = True,
              lr_scale: float = 1.0):
        """Batched local PC update: the mean over the batch of the
        per-sample updates the reference would apply. lr_scale lets the
        lower-variance batched gradient run a hotter learning rate."""
        cfg = self.cfg
        B = z_settled.shape[0]
        zp = self.z_prev if z_prev is None else z_prev

        o_hat, s = self._decode(z_settled)
        e_o = o_t - o_hat
        if update_precision and getattr(cfg, "learn_precision", True):
            self._update_precision(e_o.mean(dim=0))
        dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
        lr_o = cfg.lr_obs * lr_scale
        self.W_o -= lr_o * (dF_du.t() @ s) / B
        self.b_o -= lr_o * dF_du.mean(dim=0)

        s_in = self._trans_input(zp, action_prev)
        preds = self._member_preds(s_in)                  # [M, B, D]
        lr_t = cfg.lr_trans * lr_scale * self.pi_z
        # Per-(member, env) bootstrap keep mask — the reference's per-step
        # bernoulli, drawn per member per sample.
        keep = (torch.rand(cfg.ensemble_size, B, generator=self._g)
                <= cfg.bootstrap_prob).to(self.device)
        for m in range(cfg.ensemble_size):
            k = keep[m]
            n = int(k.sum())
            if n == 0:
                continue
            e_zm = (z_settled - preds[m]) * k.unsqueeze(1)   # masked [B, D]
            self.W_z[m] += lr_t * (e_zm.t() @ s_in) / n
            self.b_z[m] += lr_t * e_zm.sum(dim=0) / n
        self._restack()

        if advance:
            self.z_prev = z_settled.detach().clone()

    @torch.no_grad()
    def _update_precision(self, e_o_mean: torch.Tensor):
        """Reference rule fed the batch-mean error as the per-call sample."""
        cfg = self.cfg
        e2 = e_o_mean * e_o_mean
        if self._err_sq_ema is None:
            self._err_sq_ema = e2.clone()
        else:
            alpha = 1.0 / float(getattr(cfg, "precision_tau_steps", 2000.0))
            self._err_sq_ema += alpha * (e2 - self._err_sq_ema)
        ref = float(self._err_sq_ema.mean())
        mult = torch.clamp(ref / (self._err_sq_ema + 1e-12),
                           float(getattr(cfg, "precision_mult_min", 0.25)),
                           float(getattr(cfg, "precision_mult_max", 4.0)))
        self.pi_o = self.pi_prior * mult

    # ---- per-env state management ---------------------------------------

    def reset_env(self, idx):
        self.z_prev[idx] = 0.0

    # ---- persistence (reference schema) ----------------------------------

    def state_dict(self):
        D = self.cfg.latent_dim
        return {
            "W_o": self.W_o.cpu(), "b_o": self.b_o.cpu(),
            "W_z": [w.cpu() for w in self.W_z],
            "b_z": [b.cpu() for b in self.b_z],
            # Single-env schema: a fresh recurrent state (the rover zeroes
            # it on load anyway).
            "z_prev": torch.zeros(D), "cfg": self.cfg,
            "pi_o": self.pi_o.cpu(),
            "err_sq_ema": (self._err_sq_ema.cpu()
                           if self._err_sq_ema is not None else None),
        }

    def load_state_dict(self, sd):
        # Clone: .to() on a same-device tensor is a no-op view, and these
        # weights are updated IN-PLACE — never share storage with the source.
        dev = self.device
        self.W_o = sd["W_o"].clone().to(dev)
        self.b_o = sd["b_o"].clone().to(dev)
        self.W_z = [w.clone().to(dev) for w in sd["W_z"]]
        self.b_z = [b.clone().to(dev) for b in sd["b_z"]]
        self._restack()
        if sd.get("pi_o") is not None and sd["pi_o"].shape == self.pi_o.shape:
            self.pi_o = sd["pi_o"].clone().to(dev)
        if sd.get("err_sq_ema") is not None \
                and sd["err_sq_ema"].shape == self.pi_prior.shape:
            self._err_sq_ema = sd["err_sq_ema"].clone().to(dev)
        self.z_prev = torch.zeros(self.B, self.cfg.latent_dim, device=dev)


class BatchedEFEActor:
    """Batched EFE scoring: reference select() per env in one pass."""

    def __init__(self, cfg: ActorConfig, batch: int, device: str = "cpu"):
        self.cfg = cfg
        self.B = batch
        self.device = torch.device(device)
        self._structured = torch.tensor(cfg.structured, dtype=torch.float32,
                                        device=self.device)      # [S, 2]
        self._g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def _candidates(self) -> torch.Tensor:
        """[B, N, 2] — structured set shared, random per env (decorrelates
        the batch's exploration)."""
        S = self._structured.shape[0]
        struct = self._structured.unsqueeze(0).expand(self.B, S, 2)
        if self.cfg.n_random == 0:
            return struct
        rand = (torch.rand(self.B, self.cfg.n_random, self.cfg.action_dim,
                           generator=self._g) * 2.0 - 1.0).to(self.device)
        return torch.cat([struct, rand], dim=1)

    @torch.no_grad()
    def select(self, model: BatchedPCWorldModel, z_from: torch.Tensor,
               prev_action: torch.Tensor | None = None,
               slow_action: torch.Tensor | None = None,
               slow_warm: torch.Tensor | None = None,
               forward_blocked: torch.Tensor | None = None,
               cands: torch.Tensor | None = None):
        """z_from [B,D]; prev_action/slow_action [B,2]; slow_warm /
        forward_blocked bool [B]. Returns (action [B,2], info dict of [B])."""
        cfg = self.cfg
        H = cfg.horizon
        if cands is None:
            cands = self._candidates()                    # [B, N, 2]
        B, N, _ = cands.shape

        z = z_from.unsqueeze(1).expand(B, N, -1)          # [B, N, D]
        total_epi = torch.zeros(B, N, device=self.device)
        total_prag = torch.zeros(B, N, device=self.device)

        for _ in range(H):
            s_in = torch.tanh(torch.cat([z, cands], dim=-1))   # [B,N,D+2]
            preds = model._member_preds(s_in)                  # [M,B,N,D]
            z_next = preds.mean(dim=0)
            disagreement = preds.var(dim=0, unbiased=False).sum(dim=-1)
            scale = z_next.pow(2).sum(dim=-1) + 1.0
            total_epi += disagreement / scale

            o_hat, _ = model._decode(z_next)                   # [B,N,O]
            nb = cfg.num_bins
            if cfg.use_proprio and o_hat.shape[-1] > nb:
                cost = (o_hat[..., nb] - cfg.target_wl).pow(2) \
                    + (o_hat[..., nb + 1] - cfg.target_wr).pow(2) \
                    + (o_hat[..., nb + 4] - cfg.target_yaw).pow(2)
                total_prag -= cost
            if cfg.n_intero > 0:
                total_prag -= cfg.novelty_pref_weight * \
                    (o_hat[..., -1] - cfg.target_novelty).pow(2)
            if cfg.n_intero >= 2:
                total_prag -= cfg.hold_pref_weight * \
                    (o_hat[..., -2] - cfg.target_hold).pow(2)
            z = z_next

        epi_min = total_epi.min(dim=1, keepdim=True).values
        epi_max = total_epi.max(dim=1, keepdim=True).values
        epi_norm = (total_epi - epi_min) / (epi_max - epi_min + 1e-6)
        epi_gate = (epi_max.squeeze(1)
                    / max(cfg.epi_floor, 1e-9)).clamp(max=1.0)   # [B]

        prag_min = total_prag.min(dim=1, keepdim=True).values
        prag_max = total_prag.max(dim=1, keepdim=True).values
        prag_norm = (total_prag - prag_min) / (prag_max - prag_min + 1e-6)

        beta = cfg.pragmatic_weight
        score = (1.0 - beta) * epi_gate.unsqueeze(1) * epi_norm \
            + beta * prag_norm

        if prev_action is not None and cfg.smooth_weight > 0.0:
            dist = (cands - prev_action.unsqueeze(1)).norm(dim=-1) \
                / (2.0 * math.sqrt(2.0))
            score = score + cfg.smooth_weight * (1.0 - dist)

        if slow_action is not None and cfg.slow_prior_weight > 0.0:
            dist = (cands - slow_action.unsqueeze(1)).norm(dim=-1) \
                / (2.0 * math.sqrt(2.0))
            bonus = cfg.slow_prior_weight * (1.0 - dist)
            if slow_warm is not None:
                bonus = bonus * slow_warm.to(bonus.dtype).unsqueeze(1)
            score = score + bonus

        if forward_blocked is not None and cfg.blocked_penalty > 0.0:
            fwd = (cands[..., 0] + cands[..., 1]) * 0.5
            score = score - cfg.blocked_penalty \
                * forward_blocked.to(score.dtype).unsqueeze(1) \
                * torch.clamp(fwd, min=0.0)

        if cfg.deterministic:
            idx = score.argmax(dim=1)
        else:
            logits = score / max(cfg.temperature, 1e-6)
            logits = logits - logits.max(dim=1, keepdim=True).values
            probs = torch.softmax(logits, dim=1)
            idx = torch.multinomial(probs.cpu(), 1,
                                    generator=self._g).squeeze(1).to(self.device)

        action = cands.gather(
            1, idx.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)    # [B, 2]
        ar = torch.arange(B, device=self.device)
        info = {
            "epistemic": total_epi[ar, idx],
            "epistemic_max": total_epi.max(dim=1).values,
            "pragmatic": total_prag[ar, idx],
            "epi_gate": epi_gate,
        }
        return action, info


class BatchedSlowLayer:
    """Slow contextual layer with shared weights, per-env context.

    Mirrors slow_layer.SlowLayer: window accumulation per env, one slow
    tick per period (all envs in lockstep), per-env warmup after resets.
    """

    def __init__(self, cfg, batch: int, device: str = "cpu"):
        # cfg: tractor_bringup SlowLayerConfig
        self.cfg = cfg
        self.B = batch
        self.device = torch.device(device)
        self.obs_dim = cfg.fast_latent_dim + 3
        self.model = BatchedPCWorldModel(PCConfig(
            obs_dim=self.obs_dim,
            latent_dim=cfg.latent_dim,
            ensemble_size=cfg.ensemble_size,
            n_infer_iters=cfg.n_infer_iters,
            n_intero=1,
            precision_intero=cfg.novelty_precision,
            seed=cfg.seed + 77,
        ), batch=batch, device=device)
        self.actor = BatchedEFEActor(ActorConfig(
            n_random=cfg.n_random,
            pragmatic_weight=cfg.pragmatic_weight,
            num_bins=self.obs_dim - 1,
            use_proprio=False,
            n_intero=1,
            target_novelty=cfg.target_novelty,
            novelty_pref_weight=cfg.novelty_pref_weight,
            horizon=cfg.horizon,
            smooth_weight=0.0,
            seed=cfg.seed + 78,
        ), batch=batch, device=device)

        D = cfg.fast_latent_dim
        self._sum_s = torch.zeros(batch, D, device=self.device)
        self._sum_open = torch.zeros(batch, device=self.device)
        self._sum_err = torch.zeros(batch, device=self.device)
        self._sum_nov = torch.zeros(batch, device=self.device)
        self._sum_act = torch.zeros(batch, 2, device=self.device)
        self._n = torch.zeros(batch, device=self.device)   # per-env (resets)

        self.ticks = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.macro_action = torch.zeros(batch, 2, device=self.device)
        self.warm = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.td_target: torch.Tensor | None = None         # [B, fast_D]
        self._window = 0

    def accumulate(self, s_fast, openness, err, novelty, action):
        """All args batched: [B, D], [B], [B], [B], [B, 2]."""
        self._sum_s += s_fast
        self._sum_open += openness
        self._sum_err += err
        self._sum_nov += novelty
        self._sum_act += action
        self._n += 1.0
        self._window += 1

    def ready(self) -> bool:
        return self._window >= self.cfg.period_ticks

    @torch.no_grad()
    def tick(self, learn: bool = True, lr_scale: float = 1.0):
        n = self._n.clamp(min=1.0)
        s_bar = self._sum_s / n.unsqueeze(1)
        obs = torch.cat([
            0.5 + 0.5 * s_bar,
            (self._sum_open / n).unsqueeze(1),
            (self._sum_err / n).clamp(max=1.0).unsqueeze(1),
            (self._sum_nov / n).unsqueeze(1),
        ], dim=1).float()
        a2 = self._sum_act / n.unsqueeze(1)

        z2, F2, err2 = self.model.infer(obs, a2)
        if learn:
            self.model.learn(z2, a2, obs, lr_scale=lr_scale)
        else:
            self.model.z_prev = z2.detach().clone()

        action, _ = self.actor.select(self.model, z2)
        self.ticks += 1
        self.warm = self.ticks >= self.cfg.warmup_ticks

        s_in = self.model._trans_input(z2, action)
        o2_hat = self.model.reconstruct(self.model._prior_mean(s_in))
        self.macro_action = action
        self.td_target = 2.0 * o2_hat[:, :self.cfg.fast_latent_dim] - 1.0

        self._sum_s.zero_()
        self._sum_open.zero_()
        self._sum_err.zero_()
        self._sum_nov.zero_()
        self._sum_act.zero_()
        self._n.zero_()
        self._window = 0
        return {"slow_F": F2, "slow_err": err2}

    def reset_env(self, idx):
        """New-building reset for some envs: context dropped, weights kept."""
        self.model.z_prev[idx] = 0.0
        self._sum_s[idx] = 0.0
        self._sum_open[idx] = 0.0
        self._sum_err[idx] = 0.0
        self._sum_nov[idx] = 0.0
        self._sum_act[idx] = 0.0
        self._n[idx] = 0.0
        self.ticks[idx] = 0
        self.warm[idx] = False

    # td/macro masked by warm at the call site (matches the reference's
    # `None until warm` contract).

    def save(self, path: str):
        import os
        from tractor_bringup.active_inference.slow_layer import SLOW_SCHEMA_VERSION
        meta = {
            "schema": SLOW_SCHEMA_VERSION,
            "fast_latent_dim": self.cfg.fast_latent_dim,
            "obs_dim": self.obs_dim,
            "latent_dim": self.cfg.latent_dim,
            "period_ticks": self.cfg.period_ticks,
        }
        tmp = path + ".tmp"
        torch.save({"model": self.model.state_dict(), "meta": meta}, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> tuple[bool, str]:
        import os
        from tractor_bringup.active_inference.slow_layer import SLOW_SCHEMA_VERSION
        if not os.path.exists(path):
            return False, "no checkpoint"
        sd = torch.load(path, map_location="cpu", weights_only=False)
        meta = sd.get("meta", {})
        mine = {"schema": SLOW_SCHEMA_VERSION,
                "fast_latent_dim": self.cfg.fast_latent_dim,
                "obs_dim": self.obs_dim, "latent_dim": self.cfg.latent_dim}
        for k in ("schema", "fast_latent_dim", "obs_dim", "latent_dim"):
            if meta.get(k) != mine[k]:
                return False, f"incompatible ({k})"
        self.model.load_state_dict(sd["model"])
        return True, "loaded"
