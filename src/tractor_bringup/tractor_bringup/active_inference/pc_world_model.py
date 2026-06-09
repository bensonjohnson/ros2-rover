"""Temporal predictive-coding world model — the rover's 'cortex'.

This is a single-latent-layer generative model trained with PURE predictive
coding: latent state is found by iteratively settling prediction errors, and
weights are updated with LOCAL rules (an error signal at a unit's output times
the activity at its input). There is no global backpropagation anywhere.

Generative model
----------------
    latent state      z_t            (real vector, dim D)
    observation       o_t  ≈ ô_t = σ(W_o · tanh(z_t) + b_o)        (lidar, [0,1])
    temporal prior    z_t  ≈ ẑ_t = W_z · tanh([z_{t-1}; a_{t-1}]) + b_z

Two error populations:
    e_o = o_t - ô_t         (sensory prediction error)
    e_z = z_t - ẑ_t         (state / dynamics prediction error)

Free energy (precision-weighted squared error):
    F = ½ π_o‖e_o‖² + ½ π_z‖e_z‖²

Inference settles z_t by gradient descent on F (a few iterations per step).
Learning then nudges each weight by the local product (output error × input
activity) — the predictive-coding learning rule, equivalent to ∂F/∂W but
computed locally rather than via autograd.

Epistemic drive
---------------
An ENSEMBLE of transition predictors {W_z^(m)} is trained, each on a bootstrap
subset of steps so they diverge where data is sparse. The variance of their
predicted next latent given a candidate action is the expected information gain
(disagreement). High disagreement = "I don't yet understand what happens if I
do this" = where the agent wants to go. Disagreement collapses for genuinely
unpredictable noise once the ensemble agrees it's unlearnable — that is the
noisy-TV fix.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PCConfig:
    obs_dim: int = 72
    latent_dim: int = 64
    action_dim: int = 2
    ensemble_size: int = 5

    # Inference (state settling)
    n_infer_iters: int = 24
    infer_lr: float = 0.3

    # Learning (local weight updates)
    lr_obs: float = 0.05
    lr_trans: float = 0.02

    # Precisions (inverse variances) weighting the two error streams.
    precision_obs: float = 1.0
    precision_z: float = 1.0
    # Proprio channels sit at the tail of the observation vector. With only a
    # handful of proprio dims vs ~72 lidar dims, a shared precision lets lidar
    # drown out self-motion — so the tail channels get their own (higher) one.
    n_proprio: int = 0
    precision_proprio: float = 1.0

    # Per-member bootstrap keep-probability (ensemble diversity).
    bootstrap_prob: float = 0.8

    seed: int = 0


class PCWorldModel:
    def __init__(self, cfg: PCConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)

        D, O, A, M = cfg.latent_dim, cfg.obs_dim, cfg.action_dim, cfg.ensemble_size

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(self.device)

        # Observation (decoder) weights: o = σ(W_o tanh(z) + b_o)
        self.W_o = randn(O, D, scale=0.1)
        self.b_o = torch.zeros(O, device=self.device)

        # Transition ensemble: ẑ = W_z tanh([z_prev; a]) + b_z, one per member.
        self.W_z = [randn(D, D + A, scale=0.1) for _ in range(M)]
        self.b_z = [torch.zeros(D, device=self.device) for _ in range(M)]

        # Per-channel observation precision (getattr: checkpoints pickled
        # before these config fields existed must still load).
        n_prop = int(getattr(cfg, "n_proprio", 0))
        pi_prop = float(getattr(cfg, "precision_proprio", cfg.precision_obs))
        self.pi_o = torch.full((O,), float(cfg.precision_obs), device=self.device)
        if n_prop > 0:
            self.pi_o[O - n_prop:] = pi_prop
        self.pi_z = cfg.precision_z

        # Recurrent state carried across timesteps.
        self.z_prev = torch.zeros(D, device=self.device)
        self._g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    # ---- generative pieces -------------------------------------------------

    def _decode(self, z: torch.Tensor):
        s = torch.tanh(z)
        u = self.W_o @ s + self.b_o
        o_hat = torch.sigmoid(u)
        return o_hat, s

    def _trans_input(self, z_prev: torch.Tensor, action: torch.Tensor):
        return torch.tanh(torch.cat([z_prev, action]))

    def _predict_member(self, m: int, s_in: torch.Tensor):
        return self.W_z[m] @ s_in + self.b_z[m]

    def _prior_mean(self, s_in: torch.Tensor):
        preds = torch.stack([self._predict_member(m, s_in)
                             for m in range(self.cfg.ensemble_size)])
        return preds.mean(dim=0)

    @torch.no_grad()
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Predicted observation ô = σ(W_o tanh(z) + b_o) for a latent z."""
        o_hat, _ = self._decode(z)
        return o_hat

    # ---- inference: settle z_t given observation and prior -----------------

    @torch.no_grad()
    def infer(self, o_t: torch.Tensor, action_prev: torch.Tensor,
              z_prev: torch.Tensor | None = None,
              member: int | None = None,
              n_iters: int | None = None):
        """Settle the current latent state by minimizing free energy.

        `z_prev` overrides the recurrent context (used by sequence replay so it
        doesn't disturb the live state). `member` uses a single ensemble
        member's prediction as the temporal prior instead of the ensemble mean
        (used by REM dreaming so members stay decoupled and diversity — the
        epistemic signal — survives sleep). `n_iters` overrides the settle
        iteration count (replay uses fewer to fit the control-tick budget).
        Returns (z_settled, F, obs_err_norm).
        """
        zp = self.z_prev if z_prev is None else z_prev
        s_in = self._trans_input(zp, action_prev)
        z_hat = (self._prior_mean(s_in) if member is None
                 else self._predict_member(member, s_in))   # temporal prior
        z = z_hat.clone()

        iters = self.cfg.n_infer_iters if n_iters is None else n_iters
        for _ in range(iters):
            o_hat, s = self._decode(z)
            e_o = o_t - o_hat
            e_z = z - z_hat
            # ∂F/∂z via the chain through σ and tanh (computed by hand, locally).
            dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
            g_obs = (self.W_o.t() @ dF_du) * (1.0 - s * s)
            g_z = g_obs + self.pi_z * e_z
            z = z - self.cfg.infer_lr * g_z

        o_hat, _ = self._decode(z)
        e_o = o_t - o_hat
        e_z = z - z_hat
        F = 0.5 * float((self.pi_o * e_o) @ e_o) + 0.5 * self.pi_z * float(e_z @ e_z)
        return z, float(F), float(torch.linalg.norm(e_o))

    # ---- learning: local weight updates after settling ---------------------

    @torch.no_grad()
    def learn(self, z_settled: torch.Tensor, action_prev: torch.Tensor,
              o_t: torch.Tensor, z_prev: torch.Tensor | None = None,
              advance: bool = True):
        """Apply local predictive-coding updates.

        `z_prev` overrides the transition context; `advance=False` leaves the
        live recurrent state untouched (both used by replay).
        """
        cfg = self.cfg
        zp = self.z_prev if z_prev is None else z_prev

        # Observation weights: ΔW_o = -lr ∂F/∂W_o = -lr (dF_du) sᵀ
        o_hat, s = self._decode(z_settled)
        e_o = o_t - o_hat
        dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
        self.W_o -= cfg.lr_obs * torch.outer(dF_du, s)
        self.b_o -= cfg.lr_obs * dF_du

        # Transition ensemble: each member regresses toward the settled z_t,
        # on a bootstrap subset of steps for diversity.
        s_in = self._trans_input(zp, action_prev)
        for m in range(cfg.ensemble_size):
            if torch.rand((), generator=self._g).item() > cfg.bootstrap_prob:
                continue
            z_hat_m = self._predict_member(m, s_in)
            e_zm = z_settled - z_hat_m            # member dynamics error
            self.W_z[m] += cfg.lr_trans * self.pi_z * torch.outer(e_zm, s_in)
            self.b_z[m] += cfg.lr_trans * self.pi_z * e_zm

        # Advance recurrent state.
        if advance:
            self.z_prev = z_settled.detach().clone()

    # ---- epistemic value (expected information gain) -----------------------

    @torch.no_grad()
    def epistemic_value(self, z_from: torch.Tensor, actions: torch.Tensor):
        """Ensemble disagreement about next latent for each candidate action.

        actions: [N, action_dim]. Returns [N] disagreement (higher = more
        informative = more desirable under pure-epistemic active inference).
        """
        N = actions.shape[0]
        z_rep = z_from.unsqueeze(0).expand(N, -1)
        s_in = torch.tanh(torch.cat([z_rep, actions], dim=1))   # [N, D+A]
        # Stack ensemble predictions: [M, N, D]
        preds = torch.stack([s_in @ self.W_z[m].t() + self.b_z[m]
                             for m in range(self.cfg.ensemble_size)])
        # Variance across ensemble members, summed over latent dims, then
        # normalized by the mean prediction magnitude so curiosity reflects
        # relative ignorance rather than raw weight scale (which grows over
        # the brain's lifetime). The +1 keeps tiny early-life latents from
        # exploding the ratio.
        disagreement = preds.var(dim=0, unbiased=False).sum(dim=1)   # [N]
        scale = preds.mean(dim=0).pow(2).sum(dim=1) + 1.0            # [N]
        return disagreement / scale

    # ---- persistence -------------------------------------------------------

    def state_dict(self):
        return {
            "W_o": self.W_o, "b_o": self.b_o,
            "W_z": self.W_z, "b_z": self.b_z,
            "z_prev": self.z_prev, "cfg": self.cfg,
        }

    def load_state_dict(self, sd):
        self.W_o, self.b_o = sd["W_o"], sd["b_o"]
        self.W_z, self.b_z = sd["W_z"], sd["b_z"]
        # Start each session with a fresh recurrent state: the checkpointed
        # z_prev belongs to whatever moment the last save happened to catch.
        self.z_prev = torch.zeros_like(sd["z_prev"])
