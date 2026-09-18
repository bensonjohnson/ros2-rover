"""Which forward-gate rule blocks forward exploration? Re-score genomes with
one gate rule relaxed at a time (eager, 32 L3 building-holdout houses)."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from evo.arena import Arena, OBS_DIM, NUM_BINS, rev_frac, fitness
from evo.policy import PopulationNet
from evo.worlds import build_pool, HOLDOUT_SEED
from pnn_sim.safety_gate import GateConfig
dev, G = "cuda", 32
worlds = build_pool(3, G, HOLDOUT_SEED)

def variant(ar, name):
    g = ar.gate
    base_ps = g.process_scan
    if name == "no_side_eq":
        def ps(*a, **k):
            base_ps(*a, **k); g._left.fill_(1e3); g._right.fill_(1e3)
        g.process_scan = ps
    elif name == "no_arc":
        def ps(*a, **k):
            saved = g._cmd_ang.clone(); g._cmd_ang.zero_()
            base_ps(*a, **k); g._cmd_ang.copy_(saved)
        g.process_scan = ps
    elif name == "no_front_gate":
        def ps(*a, **k):
            base_ps(*a, **k); g.front_blocked.zero_()
        g.process_scan = ps

CFG = {"baseline": GateConfig(), "no_hold": GateConfig(min_block_duration=0.0),
       "no_hysteresis": GateConfig(hysteresis=0.0),
       "no_side_eq": GateConfig(), "no_arc": GateConfig(),
       "no_front_gate": GateConfig()}

def run(path, mirror, vname):
    d = np.load(path); H = int(d["hidden"])
    th = torch.as_tensor(d["thetas"], device=dev).view(1, -1)
    ar = Arena(1, G, seed=1, device=dev, fp16=True, worlds=worlds, fused=True,
               gate_cfg=CFG[vname])
    variant(ar, vname)
    net = PopulationNet(th, OBS_DIM, H); net.bind(1, G)
    st = {"h": torch.zeros(1, G, H, device=dev)}
    def step(obs, prev):
        o = obs
        if mirror:
            o = obs.clone(); p, q = NUM_BINS, NUM_BINS + 8
            o[:, :p] = torch.roll(obs[:, :p], 36, dims=1)
            o[:, p] = 1 - obs[:, p + 1]; o[:, p + 1] = 1 - obs[:, p]
            o[:, p + 5] = 1 - obs[:, p + 5]; o[:, p + 6] = 1 - obs[:, p + 6]
            o[:, q] = -obs[:, q + 1]; o[:, q + 1] = -obs[:, q]
        a, st["h"] = net.step(o.view(1, G, OBS_DIM), st["h"])
        a = a.view(G, 2)
        return torch.stack([-a[:, 1], -a[:, 0]], 1) if mirror else a
    torch.manual_seed(3)
    m = ar.run_games(step, 5400)
    return m

print(f"{'genome':34s} {'gate variant':14s} {'fit':>6} {'rooms':>5} {'cross':>5} {'rev':>4} {'dist':>5} {'coll':>6} {'fstops':>6}")
for tag, path, mirror in (("15f champ (forward)", "evo_runs/greenfield15f_s8/val_best_genome.npz", False),
                          ("14s9 champ MIRRORED (forward)", "evo_runs/greenfield14_s9/val_best_genome.npz", True),
                          ("14s9 champ as-is (reverse)", "evo_runs/greenfield14_s9/val_best_genome.npz", False)):
    for v in CFG:
        m = run(path, mirror, v)
        print(f"{tag:34s} {v:14s} {float(fitness(m).mean()):6.3f} {float(m['rooms'].mean()):5.2f} "
              f"{float((m['rooms']>=2).float().mean()):5.3f} {float(rev_frac(m).mean()):4.2f} "
              f"{float(m['dist_m'].mean()):5.1f} {float(m['collisions'].mean()):6.1f} {float(m['stops'].mean()):6.1f}", flush=True)
