"""ES genome as a platform Policy (SIM_PLATFORM stage 3a).

`ESGenomePolicy` wraps a `PopulationNet` over a STATIC theta buffer owned by
the policy, with its own hidden state and (for mem>0) the net's memory. It is
the Policy-protocol twin of two legacy paths that must agree bit-for-bit:

  * `evo.evolve.evaluate` — the eager closure that REBINDS its hidden state
    each tick (`state["h"] = h_new`);
  * `evo.evolve.GraphRunner._step` — the capturable path that keeps the
    hidden-state ADDRESS stable (`self.h.copy_(h_new)`).

The adapter takes the GraphRunner form (in-place `copy_`): it is capture-safe
AND produces exactly the values the eager closure produces, because a rebind
and an in-place copy of the same tensor hold the same numbers tick to tick.
`set_thetas()` refreshes the static weights OUTSIDE capture; `bind()` sizes
the state for the engine's P x G*merged_houses layout; `reset()` zeroes the
hidden state first and then the net's memory — the exact order the legacy
hooks ran in.
"""

from __future__ import annotations

import torch

from rover_sim.policies.es_genome import PopulationNet, genome_size
from rover_sim.runner.obs import OBS_DIM, ObsSpec, flat
from rover_sim.scoring import fitness


class ESGenomePolicy:
    """ES genome as a Policy: PopulationNet + static theta buffer + owned
    hidden state. Capture-safe (all buffers static; h updated via copy_).
    merged_houses must equal the engine's G_sets. set_thetas() before each
    run, OUTSIDE capture."""

    def __init__(self, P, hidden, *, obs="v1", action="lr", mem=0,
                 merged_houses=1, device=None):
        self.P, self.hidden, self.mem = int(P), int(hidden), int(mem)
        self.merged_houses = int(merged_houses)
        dev = torch.device(device) if device is not None else torch.device("cpu")
        N = genome_size(OBS_DIM, self.hidden, self.mem)
        self.theta_buf = torch.zeros(self.P, N, device=dev)
        self.net = PopulationNet(self.theta_buf, OBS_DIM, self.hidden,
                                 obs=obs, action=action, mem=self.mem)
        self.h = None
        self._G_eff = None
        self.obs_spec = ObsSpec(proprio_version="v2" if obs == "v2" else "v1")
        self.graph_safe = True

    def set_thetas(self, thetas):
        """Copy fresh genome rows into the static buffer — call OUTSIDE
        capture (between graph replays)."""
        self.theta_buf.copy_(thetas)

    def bind(self, P, G):
        """Size state for P individuals x G*merged_houses games. Cheap and
        idempotent on repeat (the engine calls it once per run())."""
        assert int(P) == self.P, f"bind P {P} != policy P {self.P}"
        G_eff = int(G) * self.merged_houses
        self.net.bind(self.P, G_eff)
        want = (self.P, G_eff, self.hidden)
        if self.h is None or tuple(self.h.shape) != want:
            self.h = torch.zeros(want, device=self.theta_buf.device)
        self._G_eff = G_eff

    def reset(self):
        """Zero recurrent state between games: hidden first, then memory
        (the legacy hook registration order). Idempotent."""
        if self.h is not None:
            self.h.zero_()
        if self.mem:
            self.net.reset_memory()

    def step(self, obs, prev_act):
        """obs: channel_views dict (or flat tensor). [B= P*G_eff, OBS_DIM] ->
        raw cmd [B, 2]. h is updated IN PLACE (address-stable -> graph-legal);
        values match the legacy evaluate closure's rebind exactly."""
        P, G_eff = self.P, self._G_eff
        o = flat(obs).view(P, G_eff, OBS_DIM)
        a, h_new = self.net.step(o, self.h)
        self.h.copy_(h_new)
        return a.view(P * G_eff, 2)


class GraphRunner:
    """Legacy-compat graph evaluator — the pre-platform
    `evo.evolve.GraphRunner` (removed from evolve.py in stage 3b).

    Same constructor/run contract, so legacy importers (`from evo.evolve
    import GraphRunner` — e.g. evo/bisect5.py) keep working. Internals are
    the `ESGenomePolicy` adapter; NEW code should use the adapter +
    `engine.run(..., graph=True)` directly.
    """

    def __init__(self, arena, P: int, hidden: int, obs: str = "v1",
                 action: str = "lr", mem: int = 0):
        self.arena, self.P, self.hidden = arena, P, hidden
        self.G_eff = arena.G * arena.G_sets      # envs per individual
        self.policy = ESGenomePolicy(P, hidden, obs=obs, action=action,
                                     mem=mem, merged_houses=arena.G_sets,
                                     device=arena.device)

    def run(self, thetas: torch.Tensor, ticks: int,
            w_dist: float, w_coll: float, w_cov: float = 0.0,
            w_rev: float = 0.0, w_net: float = 0.0,
            w_spin: float = 0.0):
        self.policy.set_thetas(thetas)
        metrics = self.arena.run(self.policy, ticks, graph=True)
        fit = fitness(metrics, w_dist=w_dist, w_coll=w_coll,
                      w_cov=w_cov, w_rev=w_rev, w_net=w_net,
                      w_spin=w_spin).view(self.P, self.G_eff).mean(dim=1)
        return fit, metrics
