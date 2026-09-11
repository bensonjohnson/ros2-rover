"""Which EFE score terms actually DISCRIMINATE between candidate actions?

    python3 -m tractor_bringup.active_inference.diagnose_efe \
        --ckpt ~/.ros/pnn_brain.pt --log ~/.ros/pnn_experience.jsonl


The actor min-max normalises the pragmatic total across candidates, so a term
influences the choice ONLY through how much it varies from one candidate action
to another. A term with near-zero spread is dead weight regardless of its
absolute value or its configured weight — and that is invisible in any log that
reports the term's value rather than its spread.

Replays the rover's real logged observations through a trained PCN checkpoint,
mirroring EFEActor.select's rollout exactly, but keeping each score component
separate and reporting its across-candidate spread.
"""
import json
import os
import sys

_REPO = "/home/benson/Documents/ros2-rover"
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from tractor_bringup.active_inference.pc_world_model import PCWorldModel
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--ckpt", default=os.path.expanduser("~/.ros/pnn_brain.pt"))
_ap.add_argument("--log", default=os.path.expanduser("~/.ros/pnn_experience.jsonl"),
                 help="a pnn_experience*.jsonl written by the runner")
_ap.add_argument("--target-nov", type=float, default=0.8)
_ap.add_argument("--horizon", type=int, default=8)
_ap.add_argument("--steps", type=int, default=400)
_a = _ap.parse_args()
TARGET_NOV, HORIZON, N_STEPS = _a.target_nov, _a.horizon, _a.steps

sd = torch.load(_a.ckpt, map_location="cpu", weights_only=False)
cfg = sd["cfg"]
model = PCWorldModel(cfg)
model.load_state_dict(sd)          # W_z/b_z are per-member lists
print(f"obs_dim={cfg.obs_dim} latent={cfg.latent_dim} "
      f"ensemble={cfg.ensemble_size} n_intero={getattr(cfg,'n_intero',0)}")

obs = []
with open(_a.log) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("obs"):
            obs.append(d["obs"])
O = torch.tensor(np.array(obs, dtype=np.float32))
n = min(N_STEPS, len(O))
print(f"replaying {n} of {len(O)} real steps | target_novelty={TARGET_NOV} "
      f"horizon={HORIZON} ({HORIZON/15:.2f}s)\n")

acfg = ActorConfig(num_bins=72, n_intero=2, target_novelty=TARGET_NOV,
                   horizon=HORIZON)
actor = EFEActor(acfg)
cands = actor._candidates(HORIZON)
N = cands.shape[0]
nb = acfg.num_bins
t_wl, t_wr, t_yaw = acfg.target_wl, acfg.target_wr, acfg.target_yaw

sp = {k: [] for k in ("epistemic", "proprio", "novelty", "hold")}
choices = {"full": [], "prag_only": [], "epi_only": []}
nov_hat_spread = []
z_live = None
a_prev = torch.zeros(2)   # model is single-stream: 1-D tensors

with torch.no_grad():
    for t in range(n):
        o = O[t]
        z_live = model.infer(o, a_prev, z_prev=z_live)
        if isinstance(z_live, tuple):
            z_live = z_live[0]
        z = z_live.reshape(-1).unsqueeze(0).expand(N, -1)
        epi = torch.zeros(N); prop = torch.zeros(N)
        nov = torch.zeros(N); hold = torch.zeros(N)
        novs = []
        for h in range(HORIZON):
            a_h = cands[:, h, :]
            s_in = torch.tanh(torch.cat([z, a_h], dim=1))
            preds = torch.stack([s_in @ model.W_z[m].t() + model.b_z[m]
                                 for m in range(model.cfg.ensemble_size)])
            z_next = preds.mean(dim=0)
            epi += preds.var(dim=0, unbiased=False).sum(dim=1) / \
                (s_in.pow(2).sum(dim=1) + 1.0)
            o_hat = torch.sigmoid(torch.tanh(z_next) @ model.W_o.t() + model.b_o)
            prop -= ((o_hat[:, nb] - t_wl).pow(2)
                     + (o_hat[:, nb + 1] - t_wr).pow(2)
                     + (o_hat[:, nb + 4] - t_yaw).pow(2))
            nov -= acfg.novelty_pref_weight * (o_hat[:, -1] - TARGET_NOV).pow(2)
            hold -= acfg.hold_pref_weight * (o_hat[:, -2] - acfg.target_hold).pow(2)
            novs.append(o_hat[:, -1])
            z = z_next
        for k, v in (("epistemic", epi), ("proprio", prop),
                     ("novelty", nov), ("hold", hold)):
            sp[k].append(float(v.std()))
        # Which term actually DECIDES? Compare the argmax of the real blended
        # score against the score with the epistemic term removed, and against
        # epistemic alone. The actor min-max normalises each block to [0,1]
        # INDEPENDENTLY, so a term with a tiny raw spread is stretched to full
        # range and can dominate regardless of how little real signal it has.
        prag_t = prop + nov + hold
        def nrm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-6)
        beta = acfg.pragmatic_weight
        gate = min(1.0, float(epi.std()) / max(acfg.epi_spread_floor, 1e-9))
        full = (1 - beta) * gate * nrm(epi) + beta * nrm(prag_t)
        choices["full"].append(int(full.argmax()))
        choices["prag_only"].append(int(nrm(prag_t).argmax()))
        choices["epi_only"].append(int(nrm(epi).argmax()))
        nov_hat_spread.append(float(torch.stack(novs).mean(0).std()))


print("SPREAD ACROSS CANDIDATE ACTIONS  (std over the %d candidates)" % N)
print("a term can only steer the decision through this number\n")
for k in ("epistemic", "proprio", "novelty", "hold"):
    v = np.array(sp[k])
    print(f"  {k:10s}  mean {v.mean():.6f}   median {np.median(v):.6f}")

prag = {k: np.array(sp[k]).mean() for k in ("proprio", "novelty", "hold")}
tot = sum(prag.values())
print("\nshare of the PRAGMATIC spread (what survives min-max normalisation):")
for k, v in prag.items():
    print(f"  {k:10s}  {v/max(tot,1e-9)*100:5.1f}%")

print(f"\npredicted-novelty spread across candidates: {np.mean(nov_hat_spread):.6f}")
print(f"  i.e. every candidate action predicts essentially the SAME novelty"
      if np.mean(nov_hat_spread) < 1e-3 else "")
es = np.array(sp["epistemic"])
print(f"\nepistemic gate (spread-based): across-candidate spread mean {es.mean():.5f} "
      f"vs epi_spread_floor {acfg.epi_spread_floor}")
print(f"  gate fully open on {100*(es>=acfg.epi_spread_floor).mean():.1f}% of steps")
print(f"  mean gate value: {np.minimum(1.0, es/acfg.epi_spread_floor).mean():.3f}")
f = np.array(choices["full"]); pr = np.array(choices["prag_only"])
eo = np.array(choices["epi_only"])
print("\nWHO ACTUALLY DECIDES (argmax agreement with the real blended score):")
print(f"  pragmatic-only picks the same action as the full score: {100*(f==pr).mean():5.1f}%")
print(f"  epistemic-only picks the same action as the full score: {100*(f==eo).mean():5.1f}%")
print(f"  distinct actions chosen over {len(f)} steps: full={len(set(f.tolist()))}"
      f"  prag={len(set(pr.tolist()))}  epi={len(set(eo.tolist()))} of {N} candidates")
