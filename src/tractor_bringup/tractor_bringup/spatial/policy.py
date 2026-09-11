"""Pure-PNN exploration policy — distilled frontier-seeking over the PC map.

The thesis of the redesign in one module. Pure-reactive control on the SCAN
failed: the scan has no memory, so a rover facing away from a doorway has no
signal it exists. But the PC spatial map (pc_map.PCSpatialMap) IS that memory —
so a reactive policy that reads the MAP instead of the scan can turn toward a
door it isn't currently facing. And the map is a predictive-coding net, so map
+ map-policy is pure PNN.

The policy reads an EGOCENTRIC crop of the map (robot-centred, heading-aligned)
and outputs a target bearing; the same pursuit+escape control law validated in
frontier.py executes it. So we distil only the intelligence — "which way is the
frontier" — into the net, and keep the trivial reactive controller. Training is
imitation of the frontier.py teacher (proven 1.4 rooms) by the LOCAL DELTA RULE
on a single linear readout (delta-rule on one layer == the local PC weight
update, no backprop) over fixed nonlinear features of the patch.

Egocentric patch channels (robot frame, forward = +x of the patch):
  0: free        (map says explored-free here)
  1: occupied    (explored-wall)
  2: unknown     (never sensed = where information lives)
So the readout can learn "steer toward reachable unknown" as a linear function
of where the unknown sits around me — frontier-seeking, with the map's memory.
"""

from __future__ import annotations

import math

import torch


def egocentric_patch(pc_map, pos, heading, half_m=4.0, P=32):
    """Sample a [B,3,P,P] robot-centred, heading-aligned patch of the PC map.

    Channels: free, occupied, unknown. Patch +row (forward) = robot heading,
    +col = robot-left. Uses grid_sample so it's batched + differentiable-free."""
    B = pc_map.B
    dev = pc_map.device
    n, res = pc_map.n, pc_map.res
    origin = pc_map.origin                                  # [B,2] world of cell 0

    # Local patch coords (metres): forward fx, left fy in [-half_m, half_m].
    lin = torch.linspace(-half_m, half_m, P, device=dev)
    fx, fy = torch.meshgrid(lin, lin, indexing="ij")        # [P,P]
    fx = fx.view(1, P, P); fy = fy.view(1, P, P)
    ch, sh = heading.view(B, 1, 1).cos(), heading.view(B, 1, 1).sin()
    wx = pos[:, 0].view(B, 1, 1) + fx * ch - fy * sh        # [B,P,P]
    wy = pos[:, 1].view(B, 1, 1) + fx * sh + fy * ch
    ci = (wx - origin[:, 0].view(B, 1, 1)) / res            # row (x)
    cj = (wy - origin[:, 1].view(B, 1, 1)) / res            # col (y)
    # grid_sample: grid x -> last dim (cols=cj), grid y -> dim -2 (rows=ci).
    gx = cj / (n - 1) * 2 - 1
    gy = ci / (n - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                    # [B,P,P,2]

    seen = pc_map.seen().float().unsqueeze(1)               # [B,1,n,n]
    occ = pc_map.occupied().float().unsqueeze(1)
    free = pc_map.free().float().unsqueeze(1)
    stack = torch.cat([free, occ, seen], dim=1)             # [B,3,n,n]
    samp = torch.nn.functional.grid_sample(
        stack, grid, mode="nearest", align_corners=True, padding_mode="zeros")
    free_s, occ_s, seen_s = samp[:, 0], samp[:, 1], samp[:, 2]
    unknown_s = 1.0 - seen_s
    return torch.stack([free_s, occ_s, unknown_s], dim=1)   # [B,3,P,P]


class PCPolicy:
    """Single linear readout over fixed random features of the patch, trained
    by the local delta rule to predict the teacher's target bearing as
    (cos, sin). Random nonlinear features (a fixed projection + cos) give it
    capacity while keeping the ONLY learned layer local — the PC update rule.

    CENTERING (added after the first version shipped a degenerate policy):
    the raw RFF features are dominated by a DC component — measured
    ‖mean feature‖ = 0.96 against unit-norm features, i.e. every patch maps to
    nearly the same vector, with only ~27% of that magnitude carrying which-way
    information. A delta rule on uncentered features spends essentially all of
    its capacity fitting the MEAN bearing: the shipped readout scored MAE 25.3
    deg with 5.2 deg of output spread against a 27.7 deg constant baseline —
    a constant turn wearing a policy's clothes. Least-squares on the SAME
    features reached 16.3 deg with 27.5 deg spread, which proves the features
    can express the mapping and the update rule was the bottleneck.

    So: subtract a running feature mean and re-normalise before the readout,
    and give the readout an explicit bias to hold the DC term. Both are still
    single-layer and local — a running mean is not backprop — and they leave
    the delta rule seeing the discriminative variance instead of the mean.
    """

    def __init__(self, in_dim, n_feat=1024, lr=0.02, device="cpu", seed=0,
                 center=True, mu_decay=0.99):
        g = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device(device)
        # Fixed random feature projection (reservoir): not learned.
        self.Wf = (torch.randn(n_feat, in_dim, generator=g) / math.sqrt(in_dim)
                   ).to(dev)
        self.bf = (torch.rand(n_feat, generator=g) * 2 * math.pi).to(dev)
        # Learned readout: centred features -> (cos, sin). Delta-rule trained.
        self.Wr = torch.zeros(2, n_feat, device=dev)
        self.br = torch.zeros(2, device=dev)
        # Running feature mean (the DC direction being removed).
        self.mu = torch.zeros(n_feat, device=dev)
        self._mu_init = False
        self.center = center
        self.mu_decay = mu_decay
        self.lr = lr
        self.device = dev

    def features(self, patch):
        """Raw unit-norm random features (uncentred)."""
        x = patch.reshape(patch.shape[0], -1)               # [B, 3*P*P]
        f = torch.cos(x @ self.Wf.t() + self.bf)            # [B, n_feat] RFF
        # Unit-norm so the delta rule is stable (lr·‖f‖² < 2): without this
        # ‖f‖²~n_feat and the readout diverges to NaN.
        return f / (f.norm(dim=1, keepdim=True) + 1e-6)

    def _centred(self, f):
        """Remove the DC direction and re-normalise, so the delta rule sees
        the part of the feature that actually varies with the map."""
        if not self.center:
            return f
        fc = f - self.mu
        return fc / (fc.norm(dim=1, keepdim=True) + 1e-6)

    def predict(self, patch):
        """Return (target bearing [B] rad robot-relative, raw features).

        The raw (uncentred) features are returned because `learn` needs them
        to update the running mean."""
        f = self.features(patch)
        out = self._centred(f) @ self.Wr.t() + self.br      # [B,2] = (cos,sin)
        return torch.atan2(out[:, 1], out[:, 0]), f

    def save(self, path):
        torch.save({"Wf": self.Wf.cpu(), "bf": self.bf.cpu(),
                    "Wr": self.Wr.cpu(), "br": self.br.cpu(),
                    "mu": self.mu.cpu(), "center": self.center,
                    "lr": self.lr, "schema": 2}, path)

    def load(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.Wf = sd["Wf"].to(self.device)
        self.bf = sd["bf"].to(self.device)
        self.Wr = sd["Wr"].to(self.device)
        if int(sd.get("schema", 1)) < 2:
            # Pre-centering checkpoint: reproduce its original behaviour
            # exactly (no centring, no bias) rather than silently changing it.
            self.center = False
            self.br = torch.zeros(2, device=self.device)
            self.mu = torch.zeros_like(self.mu)
        else:
            self.br = sd["br"].to(self.device)
            self.mu = sd["mu"].to(self.device)
            self.center = bool(sd["center"])
        self._mu_init = True

    @torch.no_grad()
    def learn(self, feats, target_bearing, valid):
        """Local delta-rule update toward (cos,sin) of the teacher bearing.

        ΔWr = lr * error * featuresᵀ — the single-layer PC/Hebbian-error rule,
        no backprop. `feats` are the RAW features from `predict`; the running
        mean is updated from them here. `valid` [B] masks envs with no teacher
        target."""
        nv = int(valid.sum())
        if nv == 0:
            return
        vmask = valid.bool()
        if self.center:
            batch_mu = feats[vmask].mean(0)
            if not self._mu_init:
                self.mu = batch_mu.clone()
                self._mu_init = True
            else:
                self.mu.mul_(self.mu_decay).add_(batch_mu, alpha=1 - self.mu_decay)
        fc = self._centred(feats)
        tgt = torch.stack([target_bearing.cos(), target_bearing.sin()], dim=1)
        pred = fc @ self.Wr.t() + self.br                   # [B,2]
        err = (tgt - pred) * valid.float().unsqueeze(1)     # [B,2]
        self.Wr += self.lr * (err.t() @ fc) / nv
        self.br += self.lr * err.sum(0) / nv


def pursuit_cmd(tb, ranges, bearings, blocked, escape_state, escape_ticks=14):
    """frontier.py's validated control: pursue bearing tb [B], committed
    escape-pivot when the gate blocks. escape_state = (escape[B], edir[B])
    mutated in place. Returns cmd [B,2]."""
    B = tb.shape[0]
    dev = tb.device
    escape, edir = escape_state
    new_esc = blocked & (escape <= 0)
    la = ((bearings - math.radians(55) + math.pi) % (2 * math.pi)
          - math.pi).abs() < math.radians(40)
    ra = ((bearings + math.radians(55) + math.pi) % (2 * math.pi)
          - math.pi).abs() < math.radians(40)
    lo = (ranges * la).sum(1) / la.sum().clamp(min=1)
    ro = (ranges * ra).sum(1) / ra.sum().clamp(min=1)
    edir_new = torch.where(lo >= ro, torch.ones(B, device=dev),
                           -torch.ones(B, device=dev))
    edir[:] = torch.where(new_esc, edir_new, edir)
    escape[:] = torch.where(new_esc, torch.full_like(escape, escape_ticks), escape)
    escaping = (escape > 0) | blocked
    escape[:] = torch.where(escaping, (escape - 1).clamp(min=0), escape)

    turn = (tb / 1.0).clamp(-1.0, 1.0)
    fwd = (1.0 - tb.abs() / 1.6).clamp(min=0.2) * 0.9
    pursuit = torch.stack([fwd - turn, fwd + turn], dim=1)
    pivot_s = torch.where(tb >= 0, torch.ones(B, device=dev),
                          -torch.ones(B, device=dev))
    pivot = torch.stack([-0.7 * pivot_s, 0.7 * pivot_s], dim=1)
    cmd = torch.where((tb.abs() > 1.3)[:, None], pivot, pursuit)
    esc = torch.stack([-0.75 * edir, 0.75 * edir], dim=1)
    cmd = torch.where(escaping[:, None], esc, cmd)
    return cmd.clamp(-1, 1)
