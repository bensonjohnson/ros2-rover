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
    # Interoceptive channels (e.g. place novelty) follow the proprio block at
    # the very end of the observation vector. A single scalar against ~80
    # dims needs an even stronger precision to register in settling/learning.
    n_intero: int = 0
    precision_intero: float = 1.0

    # Learned precision: each observation channel keeps a slow EMA of its own
    # squared prediction error; its precision becomes prior × clamp(ref/ema),
    # ref being the cross-channel mean. Channels noisier than average get
    # down-weighted (the model stops being surprised by what it can't
    # predict), more-predictable channels get boosted — attention, learned.
    # The hand-set block precisions above remain as the PRIOR the learned
    # multiplier modulates; the clamp bounds how far data can move them.
    learn_precision: bool = True
    precision_tau_steps: float = 2000.0   # EMA horizon, in learn() calls
    precision_mult_min: float = 0.25
    precision_mult_max: float = 4.0

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
        n_int = int(getattr(cfg, "n_intero", 0))
        pi_int = float(getattr(cfg, "precision_intero", cfg.precision_obs))
        self.pi_o = torch.full((O,), float(cfg.precision_obs), device=self.device)
        if n_prop > 0:
            self.pi_o[O - n_int - n_prop:O - n_int] = pi_prop
        if n_int > 0:
            self.pi_o[O - n_int:] = pi_int
        # pi_o is the WORKING precision (prior × learned multiplier); the
        # prior is kept so the multiplier stays bounded relative to it.
        self.pi_prior = self.pi_o.clone()
        self._err_sq_ema: torch.Tensor | None = None
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
              n_iters: int | None = None,
              td_target: torch.Tensor | None = None,
              td_precision: float = 0.0):
        """Settle the current latent state by minimizing free energy.

        `z_prev` overrides the recurrent context (used by sequence replay so it
        doesn't disturb the live state). `member` uses a single ensemble
        member's prediction as the temporal prior instead of the ensemble mean
        (used by REM dreaming so members stay decoupled and diversity — the
        epistemic signal — survives sleep). `n_iters` overrides the settle
        iteration count (replay uses fewer to fit the control-tick budget).

        `td_target` is a TOP-DOWN prior from a deeper layer: its prediction of
        this layer's activation tanh(z), weighted by `td_precision`. In the
        hierarchy, the slow layer's prediction of the fast latent is an
        empirical prior the fast layer settles under — the deeper layer's
        expectation gently biasing what the shallower layer perceives.
        Returns (z_settled, F, obs_err_norm).
        """
        zp = self.z_prev if z_prev is None else z_prev
        s_in = self._trans_input(zp, action_prev)
        z_hat = (self._prior_mean(s_in) if member is None
                 else self._predict_member(member, s_in))   # temporal prior
        z = z_hat.clone()

        use_td = td_target is not None and td_precision > 0.0
        iters = self.cfg.n_infer_iters if n_iters is None else n_iters
        for _ in range(iters):
            o_hat, s = self._decode(z)
            e_o = o_t - o_hat
            e_z = z - z_hat
            # ∂F/∂z via the chain through σ and tanh (computed by hand, locally).
            dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
            g_obs = (self.W_o.t() @ dF_du) * (1.0 - s * s)
            g_z = g_obs + self.pi_z * e_z
            if use_td:
                g_z = g_z + td_precision * (s - td_target) * (1.0 - s * s)
            z = z - self.cfg.infer_lr * g_z

        o_hat, s = self._decode(z)
        e_o = o_t - o_hat
        e_z = z - z_hat
        F = 0.5 * float((self.pi_o * e_o) @ e_o) + 0.5 * self.pi_z * float(e_z @ e_z)
        if use_td:
            e_td = s - td_target
            F += 0.5 * td_precision * float(e_td @ e_td)
        return z, float(F), float(torch.linalg.norm(e_o))

    # ---- learning: local weight updates after settling ---------------------

    @torch.no_grad()
    def learn(self, z_settled: torch.Tensor, action_prev: torch.Tensor,
              o_t: torch.Tensor, z_prev: torch.Tensor | None = None,
              advance: bool = True, update_precision: bool = True):
        """Apply local predictive-coding updates.

        `z_prev` overrides the transition context; `advance=False` leaves the
        live recurrent state untouched (both used by replay).
        `update_precision=False` freezes the learned precision EMA — replay
        and sleep re-derive errors from cold contexts (and SWS injects
        artificial noise), neither of which belongs in the sensor-noise
        ledger; only the live stream writes it.
        """
        cfg = self.cfg
        zp = self.z_prev if z_prev is None else z_prev

        # Observation weights: ΔW_o = -lr ∂F/∂W_o = -lr (dF_du) sᵀ
        o_hat, s = self._decode(z_settled)
        e_o = o_t - o_hat
        if update_precision and getattr(cfg, "learn_precision", True):
            self._update_precision(e_o)
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

    @torch.no_grad()
    def _update_precision(self, e_o: torch.Tensor):
        """Fold one error sample into the learned per-channel precision.

        Optimal precision is the inverse error variance (the −log π term in F
        makes 1/⟨e²⟩ the stationary point), so the rule is just a slow EMA of
        e² per channel inverted RELATIVE to the cross-channel mean: the
        relative form keeps the average precision anchored at the prior, so
        overall F magnitude and effective learning rates stay stable.
        """
        cfg = self.cfg
        e2 = e_o * e_o
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

    def precision_mult(self) -> torch.Tensor:
        """Learned precision multiplier per channel (1.0 = at the prior)."""
        return self.pi_o / self.pi_prior

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
            "pi_o": self.pi_o, "err_sq_ema": self._err_sq_ema,
        }

    def load_state_dict(self, sd):
        self.W_o, self.b_o = sd["W_o"], sd["b_o"]
        self.W_z, self.b_z = sd["W_z"], sd["b_z"]
        # Learned precision (checkpoints from before it existed load with the
        # prior-initialized pi_o and start their EMA fresh).
        if sd.get("pi_o") is not None and sd["pi_o"].shape == self.pi_o.shape:
            self.pi_o = sd["pi_o"]
        if sd.get("err_sq_ema") is not None \
                and sd["err_sq_ema"].shape == self.pi_prior.shape:
            self._err_sq_ema = sd["err_sq_ema"]
        # Start each session with a fresh recurrent state: the checkpointed
        # z_prev belongs to whatever moment the last save happened to catch.
        self.z_prev = torch.zeros_like(sd["z_prev"])


def insert_obs_channel(sd: dict, index: int, b_init: float = -2.0,
                       pi_init: float | None = None) -> dict:
    """Widen a saved fast-brain state_dict by one observation channel.

    The decoder is one row per channel, so a checkpoint trained before a new
    sensory channel existed can keep every learned weight: insert a zero row
    (plus bias / precision entries) at `index` and let the new channel train
    up from scratch online. The transition ensemble lives entirely in latent
    space and is untouched. b_init defaults to a low bias (sigmoid(-2)≈0.12)
    for channels that are usually near zero.
    """
    W = sd["W_o"]
    sd["W_o"] = torch.cat([W[:index], torch.zeros(1, W.shape[1]), W[index:]])
    b = sd["b_o"]
    sd["b_o"] = torch.cat([b[:index], torch.tensor([b_init]), b[index:]])
    for key, fill in (("pi_o", pi_init), ("err_sq_ema", None)):
        v = sd.get(key)
        if v is None:
            continue
        val = float(v.mean()) if fill is None else float(fill)
        sd[key] = torch.cat([v[:index], torch.tensor([val]), v[index:]])
    return sd
