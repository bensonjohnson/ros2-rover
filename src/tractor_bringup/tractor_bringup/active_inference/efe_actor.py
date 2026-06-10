"""Action selection by expected free energy — pure epistemic value.

Active inference picks actions that minimize expected free energy. With no
pragmatic (goal) term, that reduces to MAXIMIZING expected information gain:
do the thing you'd learn the most from. We estimate information gain as the
world model's transition-ensemble disagreement about the resulting latent.

We don't run a planner. Each control step we propose a batch of candidate
track commands (a structured set + random samples), score each by one-step
epistemic value, and softmax-sample one. Sampling (rather than hard argmax)
keeps early behavior from locking onto a single direction and lets the brain
keep probing. One forward pass over the ensemble — cheap enough for the CPU
control loop.
"""

from __future__ import annotations

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
    temperature: float = 0.5    # softmax temperature over the blended score
    deterministic: bool = False # if True, argmax instead of sampling
    forward_bias: float = 0.3   # (deprecated, kept for configuration compatibility)
    pragmatic_weight: float = 0.4 # beta weight: 0 = pure epistemic, 1 = pure pragmatic
    num_bins: int = 72          # lidar bins preceding the proprio channels in obs
    use_proprio: bool = True    # False = no proprio channels, pragmatic term disabled
    # Spatial novelty (transient visit-grid curiosity). The latent rollout is
    # myopic (~0.5 s), but differential-drive kinematics are nearly free, so
    # the novelty term looks several seconds ahead.
    novelty_weight: float = 0.0     # 0 disables the term
    novelty_horizon_s: float = 2.5  # kinematic lookahead
    novelty_steps: int = 6          # integration steps over that horizon
    action_scale: float = 0.6       # candidates are pre-scale; motors see cand*scale
    kin_v_max: float = 0.2          # m/s of one track at full (post-scale) command
    kin_track_width: float = 0.154  # m between track centers
    target_wl: float = 0.65     # target left wheel speed in [0,1] (0.5 is stationary)
    target_wr: float = 0.65     # target right wheel speed in [0,1]
    target_yaw: float = 0.5     # target yaw rate in [0,1] (0.5 is straight, no spin)
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

    def _spatial_novelty(self, cands: torch.Tensor, pose, novelty_fn) -> np.ndarray:
        """Mean visit-grid novelty along each candidate's kinematic rollout.

        Candidates hold one action over the horizon, so a simple constant-
        twist unicycle integration predicts where each one takes the rover.
        Returns [N] novelty in (0, 1].
        """
        cfg = self.cfg
        a_eff = cands[:, 0, :].numpy() * cfg.action_scale     # [N,2] track cmds
        v = 0.5 * (a_eff[:, 0] + a_eff[:, 1]) * cfg.kin_v_max
        w = (a_eff[:, 1] - a_eff[:, 0]) * cfg.kin_v_max / cfg.kin_track_width

        x0, y0, th0 = pose
        N = a_eff.shape[0]
        xs = np.full(N, x0, dtype=np.float64)
        ys = np.full(N, y0, dtype=np.float64)
        ths = np.full(N, th0, dtype=np.float64)
        dt = cfg.novelty_horizon_s / cfg.novelty_steps
        total = np.zeros(N, dtype=np.float64)
        for _ in range(cfg.novelty_steps):
            ths += w * dt
            xs += v * np.cos(ths) * dt
            ys += v * np.sin(ths) * dt
            total += novelty_fn(xs, ys)
        return total / cfg.novelty_steps

    @torch.no_grad()
    def select(self, model, z_from: torch.Tensor, forward_bias: float | None = None,
               pose=None, novelty_fn=None):
        """Return (action[2], info) for the current latent state.

        Performs multi-step trajectory rollouts over the world model ensemble,
        accumulating epistemic value (ensemble disagreement) and pragmatic
        value (proprioceptive prior preference matching: moving forward, no
        spin). If `pose` (x, y, theta) and `novelty_fn` are given, a spatial-
        novelty term steers toward recently-unvisited space.
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

            # Advance recurrent state for next step of rollout
            z = z_next

        # Normalize score components to [0,1]
        epi_min, epi_max = total_epistemic.min(), total_epistemic.max()
        epi_norm = (total_epistemic - epi_min) / (epi_max - epi_min + 1e-6)

        prag_min, prag_max = total_pragmatic.min(), total_pragmatic.max()
        prag_norm = (total_pragmatic - prag_min) / (prag_max - prag_min + 1e-6)

        # Fallback to config value, or blend with custom forward_bias if passed
        beta = self.cfg.pragmatic_weight
        if forward_bias is not None:
            beta = forward_bias

        score = (1.0 - beta) * epi_norm + beta * prag_norm

        # Spatial novelty: steer toward recently-unvisited space.
        nov = None
        if (self.cfg.novelty_weight > 0.0 and pose is not None
                and novelty_fn is not None):
            nov = self._spatial_novelty(cands, pose, novelty_fn)
            nov_t = torch.from_numpy(nov.astype(np.float32))
            nov_min, nov_max = nov_t.min(), nov_t.max()
            nov_norm = (nov_t - nov_min) / (nov_max - nov_min + 1e-6)
            score = score + self.cfg.novelty_weight * nov_norm

        if self.cfg.deterministic:
            idx = int(torch.argmax(score))
        else:
            logits = score / max(self.cfg.temperature, 1e-6)
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)
            idx = int(torch.multinomial(probs, 1, generator=self._g))

        # MPC: select the FIRST action of the chosen sequence
        action = cands[idx, 0, :]
        info = {
            "epistemic": float(total_epistemic[idx]),
            "epistemic_max": float(total_epistemic.max()),
            "epistemic_mean": float(total_epistemic.mean()),
            "pragmatic": float(total_pragmatic[idx]),
            "novelty": float(nov[idx]) if nov is not None else None,
        }
        return action, info
