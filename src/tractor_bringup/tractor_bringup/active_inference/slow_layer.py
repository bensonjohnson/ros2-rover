"""Slow contextual layer — the second story of the hierarchical brain.

The fast layer predicts the next lidar scan, 66 ms ahead; its planning
horizon is half a second. "Go through the doorway" can never be a plan at
that timescale. This layer is the same predictive-coding machine instantiated
one level up: its SENSORY input is a summary of the fast layer's internal
state over a ~1 s window, its transition ensemble predicts how that context
evolves over seconds, and its EFE actor plans 8 slow steps (~10–30 s) ahead.

Observations (per slow tick, all in [0,1]):
    [ mean fast activation tanh(z1) over the window   (fast_latent_dim)
    ; mean lidar openness                              (1)
    ; mean sensory prediction error                    (1)
    ; place novelty                                    (1)  <- intero tail ]

Couplings:
  - bottom-up: the fast latents themselves (averaged) are this layer's world.
  - top-down:  this layer's prediction of the NEXT window's mean fast
    activation becomes an empirical prior the fast layer settles under
    (PCWorldModel.infer's td_target) — the deeper layer's expectation gently
    biasing what the shallower layer perceives.
  - policy prior: the macro action this layer's EFE selects biases the fast
    actor's candidate scoring — a prior over policies from above, not a
    command. The fast layer remains free to overrule it near obstacles.

Persistence is deliberately a SEPARATE checkpoint from the fast brain: the
fast layer must stay loadable standalone, the slow layer will churn through
redesigns without costing fast experience, and "new building" resets can drop
slow context alone. Compatibility metadata guards against loading slow
weights trained on a different fast-layer latent space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from tractor_bringup.active_inference.pc_world_model import PCWorldModel, PCConfig
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig

SLOW_SCHEMA_VERSION = 1


@dataclass
class SlowLayerConfig:
    fast_latent_dim: int = 64
    latent_dim: int = 16
    ensemble_size: int = 5
    period_ticks: int = 15        # one slow tick ~ 1 s at 15 Hz
    n_infer_iters: int = 20
    horizon: int = 8              # slow steps -> roughly 8-30 s of lookahead
    n_random: int = 16
    pragmatic_weight: float = 0.5
    target_novelty: float = 0.8
    novelty_pref_weight: float = 2.0
    novelty_precision: float = 8.0
    # Slow ticks before the top-down prior and the policy prior switch on:
    # an infant slow layer's predictions are noise and must not steer the
    # mature fast layer. (Counts the current process's ticks, so every boot
    # re-earns influence after the recurrent state starts from zero.)
    warmup_ticks: int = 30
    seed: int = 0


class SlowLayer:
    def __init__(self, cfg: SlowLayerConfig):
        self.cfg = cfg
        # [mean tanh(z1); openness; err; novelty] — novelty last so the
        # intero-tail convention (precision boost + actor preference on the
        # final channel) is reused verbatim.
        self.obs_dim = cfg.fast_latent_dim + 3
        self.model = PCWorldModel(PCConfig(
            obs_dim=self.obs_dim,
            latent_dim=cfg.latent_dim,
            ensemble_size=cfg.ensemble_size,
            n_infer_iters=cfg.n_infer_iters,
            n_intero=1,
            precision_intero=cfg.novelty_precision,
            seed=cfg.seed + 77,
        ))
        self.actor = EFEActor(ActorConfig(
            n_random=cfg.n_random,
            pragmatic_weight=cfg.pragmatic_weight,
            num_bins=self.obs_dim - 1,
            use_proprio=False,          # no proprio block in slow obs
            n_intero=1,
            target_novelty=cfg.target_novelty,
            novelty_pref_weight=cfg.novelty_pref_weight,
            horizon=cfg.horizon,
            smooth_weight=0.0,          # slow re-decisions are already rare
            seed=cfg.seed + 78,
        ))

        D = cfg.fast_latent_dim
        self._sum_s = torch.zeros(D)
        self._sum_open = 0.0
        self._sum_err = 0.0
        self._sum_nov = 0.0
        self._sum_act = np.zeros(2, dtype=np.float64)
        self._n = 0

        self.ticks = 0                      # slow ticks this process
        self.macro_action: np.ndarray | None = None   # policy prior for fast actor
        self.td_target: torch.Tensor | None = None    # predicted mean tanh(z1)
        self.s2: torch.Tensor | None = None           # tanh(z2) for the dashboard
        self.info: dict = {}

    @property
    def window_fill(self) -> int:
        """Fast ticks accumulated toward the next slow tick."""
        return self._n

    # ---- window accumulation (every fast tick) -----------------------------

    def accumulate(self, s_fast: torch.Tensor, openness: float, err: float,
                   novelty: float, action: np.ndarray):
        self._sum_s += s_fast
        self._sum_open += float(openness)
        self._sum_err += float(err)
        self._sum_nov += float(novelty)
        self._sum_act += action
        self._n += 1

    def ready(self) -> bool:
        return self._n >= self.cfg.period_ticks

    # ---- one slow tick (every period_ticks fast ticks) ---------------------

    @torch.no_grad()
    def tick(self, learn: bool = True) -> dict:
        """Close the window: infer the slow context, learn, plan a macro
        action, and emit the top-down prediction for the next window."""
        n = max(1, self._n)
        s_bar = self._sum_s / n
        obs = torch.cat([
            0.5 + 0.5 * s_bar,             # tanh(z1) in (-1,1) -> [0,1]
            torch.tensor([self._sum_open / n,
                          min(1.0, self._sum_err / n),
                          self._sum_nov / n]),
        ]).float()
        a2 = torch.from_numpy((self._sum_act / n).astype(np.float32))

        z2, F2, err2 = self.model.infer(obs, a2)
        if learn:
            self.model.learn(z2, a2, obs)
        else:
            self.model.z_prev = z2.detach().clone()

        action, ainfo = self.actor.select(self.model, z2)
        self.ticks += 1
        warm = self.ticks >= self.cfg.warmup_ticks

        # Top-down prediction: the next window's mean fast activation under
        # the chosen macro action, decoded back to tanh range.
        s_in = self.model._trans_input(z2, action)
        o2_hat = self.model.reconstruct(self.model._prior_mean(s_in))
        self.macro_action = action.numpy() if warm else None
        self.td_target = (2.0 * o2_hat[:self.cfg.fast_latent_dim] - 1.0) \
            if warm else None

        self.s2 = torch.tanh(z2)
        self.info = {
            "slow_F": F2, "slow_err": err2,
            "slow_epi": float(ainfo["epistemic"]),
            "slow_nov_pred": float(o2_hat[-1]),
            "slow_ticks": self.ticks,
            "slow_s": [round(float(x), 4) for x in self.s2],
        }

        self._sum_s.zero_()
        self._sum_open = self._sum_err = self._sum_nov = 0.0
        self._sum_act[:] = 0.0
        self._n = 0
        return self.info

    def reset_state(self):
        """Drop context (rover moved to a new building); keep learned weights.

        Influence is also revoked until the layer re-settles into the new
        context — stale top-down expectations are worse than none.
        """
        self.model.z_prev = torch.zeros_like(self.model.z_prev)
        self._sum_s.zero_()
        self._sum_open = self._sum_err = self._sum_nov = 0.0
        self._sum_act[:] = 0.0
        self._n = 0
        self.ticks = 0
        self.macro_action = None
        self.td_target = None
        self.s2 = None

    # ---- persistence (own checkpoint, never fused with the fast brain) -----

    def _meta(self) -> dict:
        return {
            "schema": SLOW_SCHEMA_VERSION,
            "fast_latent_dim": self.cfg.fast_latent_dim,
            "obs_dim": self.obs_dim,
            "latent_dim": self.cfg.latent_dim,
            "period_ticks": self.cfg.period_ticks,
        }

    def save(self, path: str):
        # Write-then-rename so a crash mid-save can't corrupt the layer.
        tmp = path + ".tmp"
        torch.save({"model": self.model.state_dict(), "meta": self._meta()}, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> tuple[bool, str]:
        """Load if structurally compatible. Returns (loaded, reason).

        The slow layer's sensory world is the fast layer's latent space —
        weights trained against a different fast architecture are meaningless,
        so dimension metadata mismatch means start fresh, not crash.
        """
        if not os.path.exists(path):
            return False, "no checkpoint"
        sd = torch.load(path, map_location="cpu", weights_only=False)
        meta = sd.get("meta", {})
        mine = self._meta()
        for k in ("schema", "fast_latent_dim", "obs_dim", "latent_dim"):
            if meta.get(k) != mine[k]:
                return False, (f"incompatible ({k}: saved={meta.get(k)} "
                               f"vs current={mine[k]}) — starting fresh")
        self.model.load_state_dict(sd["model"])
        return True, "loaded"
