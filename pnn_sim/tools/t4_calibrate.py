#!/usr/bin/env python3
"""T4 calibration sweep: which curiosity knobs get a FRESH brain forming
multiple rotation-invariant places within the quick-run budget?

Runs configs through the real BatchedTrainer (same components the gate uses)
and reports places/displacement/novelty per config. Findings feed
BatchedTrainConfig defaults + the rover launch parameters.
"""
import sys, os, json, time
sys.path[:0] = ['.', 'src/tractor_bringup']
import numpy as np
import torch

from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

CONFIGS = {
    "A_defaults":      dict(),
    "B_epifloor":      dict(epi_floor=0.005),
    "C_epifloor_appet": dict(epi_floor=0.005, novelty_pref_weight=3.0,
                             hold_pref_weight=1.0),
    "D_early_babble_off": dict(epi_floor=0.005, novelty_pref_weight=3.0,
                               hold_pref_weight=1.0, babble_decay_ticks=500),
    "E_deep":          dict(epi_floor=0.005, novelty_pref_weight=3.0,
                           hold_pref_weight=1.0, ticks=4500),
}

def run(name, over, device="cpu", envs=8):
    ticks = over.pop("ticks", 1500)
    cfg = BatchedTrainConfig(envs=envs, device=device, seed=44,
                             out_dir=f"/tmp/pnn_sweep_{name}",
                             switch_world_every=0, snapshot_every=0,
                             save_interval_s=1e9, log_envs=0, **over)
    tr = BatchedTrainer(cfg)
    spawn = torch.stack([tr.env.x.clone(), tr.env.y.clone()], dim=1)
    t0 = time.time()
    for _ in range(ticks):
        tr.tick()
    disp = torch.linalg.norm(
        torch.stack([tr.env.x, tr.env.y], dim=1) - spawn, dim=1)
    r = {"config": name, "ticks": ticks,
         "wall_s": round(time.time() - t0, 1),
         "med_disp_m": round(float(disp.median()), 3),
         "med_places": round(float(tr.place.n_places().float().median()), 1),
         "mean_places": round(float(tr.place.n_places().float().mean()), 2),
         "novelty": round(float(tr.nov_ema.mean()), 4),
         "coll": tr._collisions, "stops": tr.gate.stops}
    tr.close()
    print(json.dumps(r), flush=True)
    return r

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="?", default="",
                    help="comma list of config names (default all)")
    ap.add_argument("--device", default="cuda" if __import__(
        "torch").cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=64)
    a = ap.parse_args()
    only = a.configs.split(",") if a.configs else list(CONFIGS)
    out = [run(k, dict(CONFIGS[k]), device=a.device, envs=a.envs)
           for k in only if k in CONFIGS]
    with open("/tmp/pnn_sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("sweep done")
