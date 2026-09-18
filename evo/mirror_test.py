"""Mirror test: run a genome as-is and rotated 180 deg in body frame.
Rotated policy sees scan bins shifted by 36 (180 deg), swapped+negated wheel
and last-action channels, negated accel x/y; its [L, R] output maps to the
real tracks as (-R, -L). A perfectly symmetric sim would score both equally;
the gap measures what the sim's front/rear asymmetry is worth."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from evo.arena import Arena, OBS_DIM, NUM_BINS, rev_frac, fitness
from evo.policy import PopulationNet
from evo.worlds import build_pool, HOLDOUT_SEED
from pnn_sim.rover import RoverConfig
dev = "cuda"
G = 32
worlds = build_pool(3, G, HOLDOUT_SEED)
res = {}
for tag, path in (("run14_s9_champ(reverse)", "evo_runs/greenfield14_s9/val_best_genome.npz"),
                  ("run15f_champ(forward)", "evo_runs/greenfield15f_s8/val_best_genome.npz")):
    d = np.load(path); H = int(d["hidden"])
    th = torch.as_tensor(d["thetas"], device=dev).view(1, -1)
    for mirror in (False, True):
        rc = RoverConfig(left_trim=1.0, right_trim=0.8) if mirror else None
        ar = Arena(1, G, seed=1, device=dev, fp16=True, worlds=worlds, fused=True, rover_cfg=rc)
        net = PopulationNet(th, OBS_DIM, H); net.bind(1, G)
        h = torch.zeros(1, G, H, device=dev)
        ar._reset_hooks.append(h.zero_)
        def step(obs, prev):
            global h
            o = obs
            if mirror:
                o = obs.clone()
                o[:, :NUM_BINS] = torch.roll(obs[:, :NUM_BINS], 36, dims=1)
                p = NUM_BINS
                o[:, p + 0] = 1 - obs[:, p + 1]; o[:, p + 1] = 1 - obs[:, p + 0]
                o[:, p + 5] = 1 - obs[:, p + 5]; o[:, p + 6] = 1 - obs[:, p + 6]
                q = NUM_BINS + 8
                o[:, q + 0] = -obs[:, q + 1]; o[:, q + 1] = -obs[:, q + 0]
            a, h = net.step(o.view(1, G, OBS_DIM), h)
            a = a.view(G, 2)
            if mirror:
                a = torch.stack([-a[:, 1], -a[:, 0]], 1)
            return a
        torch.manual_seed(3)
        m = ar.run_games(step, 5400)
        fit = fitness(m)
        k = f"{tag} {'MIRRORED' if mirror else 'as-is  '}"
        print(f"{k:40s} fit {float(fit.mean()):.3f} rooms {float(m['rooms'].mean()):.2f} "
              f"cross {float((m['rooms']>=2).float().mean()):.3f} rev {float(rev_frac(m).mean()):.2f} "
              f"dist {float(m['dist_m'].mean()):5.1f} coll {float(m['collisions'].mean()):5.1f} "
              f"gate_stops {float(m['stops'].mean()):6.1f}", flush=True)
