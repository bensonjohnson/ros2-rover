"""Action selection by expected free energy — epistemic + pragmatic value.

Active inference picks actions that minimize expected free energy: an
epistemic term (expected information gain, estimated as the world model's
transition-ensemble disagreement about the resulting latent) plus a pragmatic
term (squared error between the DECODED predicted observation and prior
preferences over it).

Curiosity about PLACES lives in the pragmatic term, not in bolt-on score
machinery: the observation vector carries an interoceptive place-novelty
channel (see PlaceMemory) and the actor holds a prior preference for it being
high — an appetite for new rooms. Action sequences whose predicted
observations keep novelty up are preferred, exactly like the preference for
forward wheel speed.

We don't run a planner. Each control step we propose a batch of candidate
track commands (a structured set + random samples), score each by rollout
expected free energy, and softmax-sample one. Sampling (rather than hard
argmax) keeps early behavior from locking onto a single direction and lets
the brain keep probing. One forward pass over the ensemble — cheap enough for
the CPU control loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


# Structured candidates always evaluated: stop, forward, reverse, spin L/R,
# gentle arcs. [left, right] in [-1, 1].
_STRUCTURED = [
    [0.0, 0.0],
    [1.0, 1.0], [0.5, 0.5],
    [-1.0, -1.0], [-0.5, -0.5],
    [1.0, -1.0], [-1.0, 1.0],
    [0.5, -0.5], [-0.5, 0.5],
    [1.0, 0.3], [0.3, 1.0],
]


@dataclass
class ActorConfig:
    action_dim: int = 2
    n_random: int = 48          # random candidates per step
    temperature: float = 0.25   # softmax temperature over the blended score
    # Confidence gate: min-max normalization stretches the epistemic scores
    # to [0,1] even when the ensemble has converged and the differences are
    # noise — a competent model then dithers on amplified randomness. Gate
    # the term by raw disagreement magnitude: below epi_floor the model is
    # confident and defers to pragmatic + novelty; genuine novelty spikes
    # disagreement and restores full curiosity weight.
    epi_floor: float = 0.02     # summed-over-horizon disagreement at full gate
    # Commitment: bonus for candidates near the currently-held action, so
    # re-decisions refine the move instead of twitching to a new one.
    smooth_weight: float = 0.25
    # Policy prior from the slow layer: bonus for candidates near the macro
    # action the slow context chose. A prior over policies from above, not a
    # command — obstacle terms and epistemic value can still overrule it.
    slow_prior_weight: float = 0.25
    deterministic: bool = False # if True, argmax instead of sampling
    pragmatic_weight: float = 0.4 # beta weight: 0 = pure epistemic, 1 = pure pragmatic
    num_bins: int = 72          # lidar bins preceding the proprio channels in obs
    use_proprio: bool = True    # False = no proprio channels, proprio preference disabled
    target_wl: float = 0.65     # target left wheel speed in [0,1] (0.5 is stationary)
    target_wr: float = 0.65     # target right wheel speed in [0,1]
    target_yaw: float = 0.5     # target yaw rate in [0,1] (0.5 is straight, no spin)
    # Interoceptive novelty preference (the appetite for new rooms). The
    # observation's LAST channel is place novelty; preferring it high makes
    # exploration ordinary goal-seeking under EFE. The weight is per-step
    # against the 3 proprio preference terms — one channel needs the boost.
    n_intero: int = 1           # interoceptive channels at the obs tail (0 disables)
    target_novelty: float = 0.8 # preferred place novelty in [0,1]
    novelty_pref_weight: float = 2.0
    # Safety-hold preference (only with n_intero >= 2, tail = [hold, novelty]):
    # the obs carries the safety gate's state as an interoceptive channel, and
    # the actor prefers rollouts predicted NOT to trip it — collision
    # avoidance as ordinary goal-seeking, learned from the gate's own firings.
    target_hold: float = 0.0
    hold_pref_weight: float = 2.0
    # While the gate is actively clamping forward motion, forward candidates
    # are wasted ticks — penalize them so escapes start immediately instead
    # of pushing into the clamp for the persist window.
    blocked_penalty: float = 4.0
    horizon: int = 8            # planning horizon length
    seed: int = 0
    structured: list = field(default_factory=lambda: list(_STRUCTURED))


class EFEActor:
    def __init__(self, cfg: ActorConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._structured = torch.tensor(cfg.structured, dtype=torch.float32,
                                        device=self.device)
        self._g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def _candidates(self, H: int) -> torch.Tensor:
        # Repeat structured candidates over the horizon H: [S, H, 2]
        struct_seq = self._structured.unsqueeze(1).repeat(1, H, 1)

        # Generate random base actions: [n_random, 2]
        rand_base = (torch.rand(self.cfg.n_random, self.cfg.action_dim,
                               generator=self._g) * 2.0 - 1.0).to(self.device)
        # Repeat random candidates over the horizon H: [n_random, H, 2]
        rand_seq = rand_base.unsqueeze(1).repeat(1, H, 1)

        # Concatenate along batch dimension: [N, H, 2]
        return torch.cat([struct_seq, rand_seq], dim=0)

    @torch.no_grad()
    def select(self, model, z_from: torch.Tensor, prev_action=None,
               slow_action=None, forward_blocked: bool = False):
        """Return (action[2], info) for the current latent state.

        Performs multi-step trajectory rollouts over the world model ensemble,
        accumulating epistemic value (ensemble disagreement) and pragmatic
        value (prior preference matching over the decoded observation: moving
        forward, no spin, and high place novelty — the curiosity appetite).
        """
        H = self.cfg.horizon
        cands = self._candidates(H)  # [N, H, 2]
        N = cands.shape[0]

        # Initialize rollout states
        z = z_from.unsqueeze(0).expand(N, -1)  # [N, D]
        total_epistemic = torch.zeros(N, device=self.device)
        total_pragmatic = torch.zeros(N, device=self.device)

        # Target proprioceptives
        t_wl = self.cfg.target_wl
        t_wr = self.cfg.target_wr
        t_yaw = self.cfg.target_yaw

        for h in range(H):
            a_h = cands[:, h, :]  # [N, 2]

            # 1. State transition step: s_in = tanh([z; a_h]) -> [N, D+A]
            s_in = torch.tanh(torch.cat([z, a_h], dim=1))

            # Project next state for each ensemble member: [M, N, D]
            preds = torch.stack([s_in @ model.W_z[m].t() + model.b_z[m]
                                 for m in range(model.cfg.ensemble_size)])

            # Next mean latent state: [N, D]
            z_next = preds.mean(dim=0)

            # Epistemic value: ensemble variance summed over latent dims,
            # normalized by mean prediction magnitude (matches
            # PCWorldModel.epistemic_value) so it stays scale-invariant. -> [N]
            disagreement = preds.var(dim=0, unbiased=False).sum(dim=1)
            scale = z_next.pow(2).sum(dim=1) + 1.0
            total_epistemic += disagreement / scale

            # 2. Decode the next mean state to check proprioception alignment:
            # o_hat = sigmoid(W_o @ tanh(z_next) + b_o)
            s_latent = torch.tanh(z_next)
            u = s_latent @ model.W_o.t() + model.b_o
            o_hat = torch.sigmoid(u)  # [N, obs_dim]

            # Proprio channels follow the lidar bins:
            # [wl, wr, roll, pitch, yaw, ax, ay, az] at num_bins..num_bins+7.
            nb = self.cfg.num_bins
            if self.cfg.use_proprio and o_hat.shape[1] > nb:
                wl = o_hat[:, nb]
                wr = o_hat[:, nb + 1]
                yaw = o_hat[:, nb + 4]
                # Pragmatic cost is squared error from target proprio
                cost = (wl - t_wl).pow(2) + (wr - t_wr).pow(2) + (yaw - t_yaw).pow(2)
                total_pragmatic -= cost

            # Interoceptive preference: the obs tail carries place novelty;
            # prefer rollouts whose PREDICTED novelty stays near the target.
            # This is where curiosity about places lives now — in the
            # generative model, not in bolt-on kinematic score terms.
            if self.cfg.n_intero > 0:
                nov_hat = o_hat[:, -1]
                total_pragmatic -= self.cfg.novelty_pref_weight * \
                    (nov_hat - self.cfg.target_novelty).pow(2)

            # Safety-hold avoidance (see ActorConfig.target_hold): with two
            # intero channels the tail is [hold, novelty].
            if self.cfg.n_intero >= 2:
                hold_hat = o_hat[:, -2]
                total_pragmatic -= self.cfg.hold_pref_weight * \
                    (hold_hat - self.cfg.target_hold).pow(2)

            # Advance recurrent state for next step of rollout
            z = z_next

        # Normalize score components to [0,1]
        epi_min, epi_max = total_epistemic.min(), total_epistemic.max()
        epi_norm = (total_epistemic - epi_min) / (epi_max - epi_min + 1e-6)

        # Confidence gate (see ActorConfig.epi_floor).
        epi_gate = min(1.0, float(epi_max) / max(self.cfg.epi_floor, 1e-9))

        prag_min, prag_max = total_pragmatic.min(), total_pragmatic.max()
        prag_norm = (total_pragmatic - prag_min) / (prag_max - prag_min + 1e-6)

        beta = self.cfg.pragmatic_weight
        score = (1.0 - beta) * epi_gate * epi_norm + beta * prag_norm

        # Commitment bonus: prefer candidates near the action currently held,
        # scaled by distance in command space (max possible = 2*sqrt(2)).
        if prev_action is not None and self.cfg.smooth_weight > 0.0:
            prev = torch.as_tensor(np.asarray(prev_action), dtype=torch.float32,
                                   device=cands.device)
            dist = (cands[:, 0, :] - prev).norm(dim=1) / (2.0 * math.sqrt(2.0))
            score = score + self.cfg.smooth_weight * (1.0 - dist)

        # Policy prior from the slow layer (see ActorConfig.slow_prior_weight).
        if slow_action is not None and self.cfg.slow_prior_weight > 0.0:
            sa = torch.as_tensor(np.asarray(slow_action), dtype=torch.float32,
                                 device=cands.device)
            dist = (cands[:, 0, :] - sa).norm(dim=1) / (2.0 * math.sqrt(2.0))
            score = score + self.cfg.slow_prior_weight * (1.0 - dist)

        # Gate awareness (see ActorConfig.blocked_penalty): score components
        # are O(1), so the penalty dominates for any meaningfully-forward
        # candidate while leaving reverse/pivot ranking untouched.
        if forward_blocked and self.cfg.blocked_penalty > 0.0:
            fwd = (cands[:, 0, 0] + cands[:, 0, 1]) * 0.5
            score = score - self.cfg.blocked_penalty * torch.clamp(fwd, min=0.0)

        if self.cfg.deterministic:
            idx = int(torch.argmax(score))
        else:
            logits = score / max(self.cfg.temperature, 1e-6)
            logits = logits - logits.max()
            # Sample on CPU: the generator lives there, and N is tiny.
            probs = torch.softmax(logits, dim=0).cpu()
            idx = int(torch.multinomial(probs, 1, generator=self._g))

        # MPC: select the FIRST action of the chosen sequence
        action = cands[idx, 0, :]
        info = {
            "epistemic": float(total_epistemic[idx]),
            "epistemic_max": float(total_epistemic.max()),
            "epistemic_mean": float(total_epistemic.mean()),
            "pragmatic": float(total_pragmatic[idx]),
            "epi_gate": epi_gate,
        }
        return action, info
