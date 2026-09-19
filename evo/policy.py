"""Genome: recurrent MLP policy (+ optional explicit memory), flat float32.

    o_t (scan bins + proprio + last action)
      -> pre_t = Wx o + Wh h_{t-1} + [Wm m_{t-1}] + bh
      -> h_t   = tanh(pre_t)
      -> [memory write, mem>0:
             q = [o, h_t]
             g = sigmoid(Wg q + bg)          # per-slot write gate, bg mean -3
             v = tanh(Wv q + bv)             # write candidate, Wv init x0.1
             m_t = (1 - g) * m_{t-1} + g * v # EMA-style write]
      -> a_t = tanh(Wo h + [Wom m] + bo)      (2 wheel commands)

Weights are evolved, never gradient-learned. The flat packing keeps the
whole population in one [P, N] tensor so mutation/crossover are elementwise
ops and the batched forward is a handful of einsums.

Memory (mem>0, run 25) is CAMEMBE-style: M fixed slots [P, G, M] the net
reads (into the hidden pre-activation and the action) and writes through a
learned gate every tick. It starts NEARLY OFF — write-gate bias inits at
mean -3 (g ~ 0.05) with small write values — so a mem>0 gen-0 rollout
behaves close to the mem=0 flat MLP and evolution only opens memory if it
pays (a fixed hidden recurrence has 24 runs of evidence it will not find
"that direction already failed" on its own; explicit slots make it
reachable). The `m` buffer is owned by the net (allocated in bind(),
zeroed by reset_memory()), so the step(o, h) -> (a, h) signature every
caller already uses is unchanged. Genomes without a `mem_slots` npz field
are mem=0 and pack byte-identically to the pre-memory layout.
"""

from __future__ import annotations

import numpy as np
import torch


def genome_size(n_in: int, hidden: int, mem: int = 0) -> int:
    n = (n_in * hidden + hidden          # Wx, bh
         + hidden * hidden               # Wh
         + hidden * 2 + 2)               # Wo, bo
    if mem:
        q = n_in + hidden                # write reads [o, h]
        n += q * mem + mem               # Wg, bg  (per-slot write gate)
        n += q * mem + mem               # Wv, bv  (write candidate)
        n += mem * hidden                # Wm  (read memory into hidden)
        n += mem * 2                     # Wom (read memory into action)
    return n


def _param_shapes(n_in: int, hidden: int, mem: int = 0):
    """(name, shape, init_std, init_mean) in packing order. Biases init
    hot-ish (std): a fully-zero policy never moves, and an ES on
    frozen-at-zero dynamics has no gradient of progress to follow.
    Memory params append only when mem>0, so the mem=0 packing is
    byte-identical to the historical layout (and sample_population's
    per-gene std for mem=0 is unchanged: init_mean is 0 for every legacy
    parameter). The write-gate bias bg is the one parameter with a nonzero
    init MEAN (-3): memory starts closed and must earn its way on; Wv
    init std is x0.1 so an early accidental write is a nudge, not a
    full-tanh shock."""
    out = [
        ("Wx", (n_in, hidden), 1.0 / np.sqrt(n_in), 0.0),
        ("bh", (hidden,), 0.5, 0.0),
        ("Wh", (hidden, hidden), 1.0 / np.sqrt(hidden), 0.0),
        ("bo", (2,), 0.5, 0.0),
        ("Wo", (hidden, 2), 1.0 / np.sqrt(hidden), 0.0),
    ]
    if mem:
        q = n_in + hidden
        out += [
            ("Wg", (q, mem), 1.0 / np.sqrt(q), 0.0),
            ("bg", (mem,), 1.0, -3.0),
            ("Wv", (q, mem), 0.1 / np.sqrt(q), 0.0),
            ("bv", (mem,), 0.0, 0.0),
            ("Wm", (mem, hidden), 1.0 / np.sqrt(mem), 0.0),
            ("Wom", (mem, 2), 1.0 / np.sqrt(mem), 0.0),
        ]
    return out


def per_gene_scale(n_in: int, hidden: int, mem: int = 0) -> np.ndarray:
    """Per-gene init std, reused as the mutation-size unit: a sigma of 1.0
    means one init-standard-deviation for that gene, so layers with tiny
    weights (deep, wide fan-in) are not frozen out of the search."""
    out = []
    for _, shape, std, _mean in _param_shapes(n_in, hidden, mem):
        out.append(np.full(int(np.prod(shape)), std, dtype=np.float32))
    return np.concatenate(out)


def sample_population(P: int, n_in: int, hidden: int,
                      rng: np.random.Generator, mem: int = 0) -> np.ndarray:
    """P fresh nets, each drawn from the init distribution — the natural
    'random greenfield rover' the first generation must not be worse than."""
    stds, means = [], []
    for _, shape, std, mean in _param_shapes(n_in, hidden, mem):
        stds.append(np.full(int(np.prod(shape)), std, dtype=np.float32))
        means.append(np.full(int(np.prod(shape)), mean, dtype=np.float32))
    std = np.concatenate(stds)
    mean = np.concatenate(means)
    return (mean[None, :]
            + rng.standard_normal((P, std.size)).astype(np.float32) * std)


OBS_MODES = ("v1", "v2")      # v2: gate-flag channels + inputs centred
ACTION_MODES = ("lr", "vw")   # vw: outputs (v, w) mixed to tracks


def genome_meta(obs: str = "v1", action: str = "lr", mem: int = 0) -> dict:
    """npz fields every saved genome/population carries (absent = v1/lr/0).
    mem_slots records the memory width so every scorer rebuilds the exact
    architecture from the file alone (no CLI flag to get wrong)."""
    assert obs in OBS_MODES and action in ACTION_MODES and mem >= 0
    return {"obs_mode": obs, "action_mode": action, "mem_slots": int(mem)}


def read_meta(d) -> tuple[str, str, int]:
    """(obs_mode, action_mode, mem_slots) of a loaded npz; files from runs
    <= 24 have no mem_slots field and are mem=0 (byte-identical packing)."""
    obs = str(d["obs_mode"]) if "obs_mode" in d else "v1"
    act = str(d["action_mode"]) if "action_mode" in d else "lr"
    mem = int(d["mem_slots"]) if "mem_slots" in d else 0
    return obs, act, mem


class PopulationNet:
    """[P individuals] x [G envs] batched forward. All tensors on device.

    obs="v2": the first n_in-2 inputs (scan bins + proprio, all in [0, 1])
    are centred to [-1, 1] before the net — raw [0, 1] scans sit near 1 in
    open space, a large DC drive into tanh units whose biases also init at
    0.5 (evolved controllers came out bang-bang, |a| ~0.97). last-action
    inputs are already in [-1, 1]. (The gate-flag channels themselves are
    produced by Arena(gate_obs=True).)
    action="vw": outputs are (forward, turn), tracks = clamp(v -/+ w): going
    straight / turning are single-output changes instead of a coordinated
    left/right pair (left_trim 0.8 makes 'straight' an asymmetric L/R).

    mem>0: the net owns memory m [P, G, mem] (allocated in bind, zeroed by
    reset_memory()). step() keeps its (o, h) -> (a, h) signature; callers
    never pass or see m. Correctness across games: zero memory between
    rollouts — GraphRunner installs reset_memory as an arena reset hook;
    one-shot tools call net.reset_memory() before each run_games call."""

    def __init__(self, thetas: torch.Tensor, n_in: int, hidden: int,
                 obs: str = "v1", action: str = "lr", mem: int = 0):
        """thetas [P, N] on device."""
        self.P, self.N = thetas.shape
        self.n_in, self.hidden, self.mem = n_in, hidden, int(mem)
        self.thetas = thetas
        self.obs_center = obs == "v2"
        self.action_vw = action == "vw"
        self._views = None
        self.m = None

    def bind(self, P: int, G: int):
        """(Re)bind views after thetas changed. P may differ from thetas'
        first dim only via expand — we always pass the exact population."""
        t = self.thetas
        off = 0
        views = {}
        for name, shape, _std, _mean in _param_shapes(self.n_in, self.hidden,
                                                      self.mem):
            numel = int(np.prod(shape))
            views[name] = t[:, off:off + numel].view(self.P, *shape)
            off += numel
        assert off == self.N, f"packing {off} != genome size {self.N}"
        self._views = views
        self._G = G
        if self.mem:
            want = (self.P, G, self.mem)
            if self.m is None or tuple(self.m.shape) != want:
                self.m = torch.zeros(want, device=t.device)

    def reset_memory(self):
        if self.m is not None:
            self.m.zero_()

    def step(self, o: torch.Tensor, h: torch.Tensor):
        """o [P, G, n_in], h [P, G, hidden] -> (action [P,G,2], h).
        Batched over individuals with bmm (row-vector convention, same
        maths as the old expand+einsum, without expanded weight views).
        The mem>0 branch is bmm/elementwise only and updates memory via
        copy_ — buffer addresses stay stable, so the whole rollout remains
        CUDA-graph capturable (same contract as GraphRunner's h.copy_)."""
        v = self._views
        if self.obs_center:
            k = self.n_in - 2
            o = torch.cat([o[..., :k] * 2.0 - 1.0, o[..., k:]], dim=-1)
        pre = (torch.bmm(o, v["Wx"]) + torch.bmm(h, v["Wh"])
               + v["bh"].unsqueeze(1))
        if self.mem:
            assert self.m is not None, "bind() not called"
            pre = pre + torch.bmm(self.m, v["Wm"])
        h_new = torch.tanh(pre)
        act_pre = torch.bmm(h_new, v["Wo"]) + v["bo"].unsqueeze(1)
        if self.mem:
            # WRITE after h_new (memory sees what the controller just
            # observed through h); the action reads the UPDATED memory the
            # same tick — a write observed at tick t can steer t itself.
            q = torch.cat([o, h_new], dim=-1)
            g = torch.sigmoid(torch.bmm(q, v["Wg"]) + v["bg"].unsqueeze(1))
            w = torch.tanh(torch.bmm(q, v["Wv"]) + v["bv"].unsqueeze(1))
            self.m.copy_(self.m + g * (w - self.m))   # EMA write, graph-safe
            act_pre = act_pre + torch.bmm(self.m, v["Wom"])
        act = torch.tanh(act_pre)
        if self.action_vw:
            fwd, turn = act[..., 0:1], act[..., 1:2]
            act = torch.cat([fwd - turn, fwd + turn], dim=-1).clamp(-1, 1)
        return act, h_new
