#!/usr/bin/env python3
"""Behavior-cloned explorer seeding for run 8 (evo.bc_seed).

Runs 1-7a: ES from random init never leaves the cruise attractor
(train_cells ~0.066, rooms 1.01) even with novelty priced above cruising
comfort (w_cov 0.6). explore_probe.py showed scripted personalities cover
0.110-0.144 — 2x the evolved population — so exploration IS representable
from scan72+proprio; what is missing is a mutation path INTO it. Classic
fix: seed the population with behavior-cloned experts, let ES improve them.
(Gradients are used for INITIALIZATION ONLY — evolution itself stays
gradient-free.)

Pipeline:
  1. candidate round — N scripted personalities (explore_probe's per-env
     bias/commitment-timer modes, verbatim) in the TRAIN arena; rank by
     per-personality coverage (drop near-standstills).
  2. keep top K — replay each over G houses x S sets, recording the
     (obs, expert cmd) stream (pre-gate: the student learns the pre-gate
     map; the arena re-gates it at deploy, as for evolved members).
  3. distill each keeper into a recurrent-MLP genome with Adam
     (segment-BPTT over the expert stream), packed in evo.policy layout.
  4. verify — replay the distilled MLP (NOT the script) in the train
     houses and the honest holdout 777000. Only members whose REPLAY
     coverage beats --min-cov are kept: scripted coverage the genome
     cannot reproduce is worthless.

    python3 -m evo.bc_seed --device cuda --out evo_runs/bc8/bc_thetas.npz
then
    python3 -m evo.evolve ... --bc-from evo_runs/bc8/bc_thetas.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .arena import Arena, NUM_BINS, OBS_DIM
from .policy import _param_shapes, genome_size, per_gene_scale
from .evolve import evaluate   # the EXACT eval path evolution uses


def make_scripted(per: dict, reps: int, device: str):
    """Parameterized version of explore_probe.make_scripted: personality
    params are caller-provided [N] tensors so keepers can be replayed for
    data collection; each personality is replicated over `reps` envs
    (same personality, different houses). Behaviour is verbatim from the
    probe (the expert whose coverage we must reproduce)."""
    dev = torch.device(device)
    N = int(per["bias"].shape[0])
    B = N * reps

    def rep(x):
        return torch.as_tensor(x, device=dev).repeat_interleave(reps)

    bias, commit_T, mode_i = rep(per["bias"]), rep(per["commit_T"]), rep(per["mode_i"])
    timer = torch.zeros(B, device=dev)
    hold = torch.stack([rep(per["cmd0L"]), rep(per["cmd0R"])], dim=1).clone()
    zero = torch.zeros(B, device=dev)

    def step(obs, prev):
        s = obs[:, :NUM_BINS]
        front = s[:, :8].min(dim=1).values
        left = s[:, 9:36].mean(dim=1)
        right = s[:, 36:63].mean(dim=1)
        timer2 = timer + 1
        fire = timer2 >= commit_T
        newL = torch.rand(B, device=dev) * 2 - 1
        newR = torch.rand(B, device=dev) * 2 - 1
        open_diff = (left - right).clamp(-1, 1)
        steer = 0.5 * open_diff + 0.5 * bias
        f = 0.4 + 0.6 * front.clamp(0, 1)
        m0 = torch.stack([f - steer, f + steer], dim=1)                 # cruise
        m1 = torch.stack([torch.full_like(bias, 0.15), torch.ones_like(bias)], dim=1)
        m1b = torch.stack([torch.ones_like(bias), torch.full_like(bias, 0.15)], dim=1)
        m1s = torch.where((bias > 0).unsqueeze(1), m1, m1b)              # arc
        m2 = torch.stack([newL, newR], dim=1)                            # random
        pick = torch.where(mode_i.unsqueeze(1) == 0, m0,
                           torch.where(mode_i.unsqueeze(1) == 1, m1s, m2))
        holdNew = torch.where(fire.unsqueeze(1), pick, hold)
        # wedged override: front closed AND both sides closed -> back up
        wedged = (front < 0.06) & (torch.maximum(left, right) < 0.12)
        back = torch.stack([torch.full_like(bias, -0.5), torch.full_like(bias, 0.5)], dim=1)
        back2 = torch.stack([torch.full_like(bias, 0.5), torch.full_like(bias, -0.5)], dim=1)
        holdNew = torch.where(wedged.unsqueeze(1),
                              torch.where((bias > 0).unsqueeze(1), back, back2),
                              holdNew)
        hold.copy_(holdNew)
        timer.copy_(torch.where(fire, zero, timer2))
        return hold.clamp(-1, 1)

    return step


def pack_theta(named: dict, n_in: int, hidden: int) -> np.ndarray:
    return np.concatenate([named[name].detach().cpu().numpy().ravel()
                           for name, _, _ in _param_shapes(n_in, hidden)]
                          ).astype(np.float32)


def distill(o: torch.Tensor, a: torch.Tensor, hidden: int,
            rng: np.random.Generator, epochs: int = 60, seg: int = 180,
            bs: int = 256, lr: float = 3e-3) -> tuple[np.ndarray, float]:
    """o [T,B,n_in], a [T,B,2] expert stream -> packed genome + final MSE.
    Student rolls its OWN hidden state over the expert observation stream
    (open-loop BC); obs already carries last action, so between command
    changes the student just needs to echo prev_act through recurrence."""
    T, B, _ = o.shape
    nseg_t = T // seg
    o_s = (o[:nseg_t * seg].view(nseg_t, seg, B, -1)
           .permute(2, 0, 1, 3).reshape(B * nseg_t, seg, -1))
    a_s = (a[:nseg_t * seg].view(nseg_t, seg, B, -1)
           .permute(2, 0, 1, 3).reshape(B * nseg_t, seg, -1))
    M = B * nseg_t

    scales = torch.as_tensor(per_gene_scale(OBS_DIM, hidden),
                             device=o.device)
    named = {}
    for name, shape, scale in _param_shapes(OBS_DIM, hidden):
        init = torch.as_tensor(
            rng.standard_normal(shape).astype(np.float32), device=o.device)
        named[name] = (init * float(scale) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam(list(named.values()), lr=lr)

    idx = np.arange(M)
    for ep in range(epochs):
        rng.shuffle(idx)
        tot = 0.0
        for s0 in range(0, M, bs):
            rows = torch.as_tensor(idx[s0:s0 + bs], device=o.device)
            ob = o_s.index_select(0, rows)
            ab = a_s.index_select(0, rows)
            pre = ob @ named["Wx"]
            h = torch.zeros(ob.shape[0], hidden, device=o.device)
            loss = 0.0
            for t in range(seg):
                h = torch.tanh(pre[:, t] + h @ named["Wh"] + named["bh"])
                act = torch.tanh(h @ named["Wo"] + named["bo"])
                loss = loss + torch.nn.functional.mse_loss(act, ab[:, t])
            opt.zero_grad()
            (loss / seg).backward()
            opt.step()
            tot += float(loss) / seg
    mse = tot / max(1, M // bs)
    return pack_theta(named, OBS_DIM, hidden), mse


def replay_cov(thetas: torch.Tensor, arena: Arena, hidden: int,
               ticks: int) -> tuple[float, float]:
    """Mean coverage + rooms of [P,N] genomes in `arena`, through the
    EXACT evolution eval path (PopulationNet, not the script)."""
    _, m = evaluate(arena, thetas, hidden, ticks, w_dist=0.0,
                    w_coll=0.0, w_cov=0.0)
    return (float((m["cells"] / m["cells_total"]).mean()),
            float(m["rooms"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--keepers", type=int, default=8)
    ap.add_argument("--cand-games", type=int, default=8)
    ap.add_argument("--data-games", type=int, default=32)
    ap.add_argument("--data-sets", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seg", type=int, default=180)
    ap.add_argument("--min-cov", type=float, default=0.09,
                    help="KEEP only if MLP replay coverage >= this in train "
                         "houses (cruise band is 0.063-0.066, scripted "
                         "ceiling 0.144)")
    ap.add_argument("--train-seed", type=int, default=65_000,
                    help="must match the run-8 train seed's first sets")
    ap.add_argument("--holdout-seed", type=int, default=777_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--out", default="evo_runs/bc8/bc_thetas.npz")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # personalities — same distributions as explore_probe (mix over modes)
    N = args.candidates
    per = {
        "bias": rng.uniform(-1, 1, N).astype(np.float32),
        "commit_T": rng.integers(15, 150, N).astype(np.float32),
        "mode_i": rng.integers(0, 3, N).astype(np.int64),
        "cmd0L": rng.uniform(-1, 1, N).astype(np.float32),
        "cmd0R": rng.uniform(-1, 1, N).astype(np.float32),
    }

    # ---- 1. candidate round: rank personalities by coverage -------------
    cand = Arena(N, args.cand_games, seed=args.train_seed, device=dev,
                 fp16=args.fp16, merged_houses=1)
    step = make_scripted(per, args.cand_games, dev)
    torch.manual_seed(3)
    m = cand.run_games(step, args.ticks)
    reps = args.cand_games
    cov = (m["cells"] / m["cells_total"]).view(N, reps).mean(1).cpu().numpy()
    dist = m["dist_m"].view(N, reps).mean(1).cpu().numpy()
    ok = [i for i in np.argsort(-cov) if dist[i] >= 3.0]   # drop standstills
    keep = ok[:args.keepers]
    print(f"[cand] {N} personalities in {time.time()-t0:.0f}s; "
          f"cov top={np.round(cov[keep], 3)}", flush=True)
    if len(keep) < 4:
        print("[cand] WARNING: <4 viable personalities — probe ceiling not "
              "reached; abort (fix candidates/modes, not distiller)",
              flush=True)
        raise SystemExit(2)

    # ---- 2. record expert streams over train houses ----------------------
    data_arena = Arena(len(keep), args.data_games, seed=args.train_seed,
                       device=dev, fp16=args.fp16,
                       merged_houses=args.data_sets)
    rows = args.data_games * args.data_sets
    keep_per = {k: per[k][keep] for k in per}
    step = make_scripted(keep_per, rows, dev)
    torch.manual_seed(4)
    rec = {"obs": [], "cmd": []}
    data_arena.run_games(step, args.ticks, rec=rec)
    O = torch.stack(rec["obs"])                       # [T, B, n_in]
    A = torch.stack(rec["cmd"])                       # [T, B, 2]
    print(f"[data] expert streams {tuple(O.shape)} in "
          f"{time.time()-t0:.0f}s", flush=True)
    del rec
    import gc; gc.collect()

    # ---- 3. distill each keeper, 4. verify replay ------------------------
    ho = Arena(len(keep), args.data_games, seed=args.holdout_seed,
               device=dev, fp16=args.fp16)
    thetas, meta = [], []
    for j, kp in enumerate(keep):
        o_j = O[:, j * rows:(j + 1) * rows].contiguous()   # [T, rows, n_in]
        a_j = A[:, j * rows:(j + 1) * rows].contiguous()
        th, mse = distill(o_j, a_j, args.hidden, rng, epochs=args.epochs,
                          seg=args.seg)
        thetas.append(th)
        meta.append({"script_cov": float(cov[kp]), "mse": mse})
        print(f"[distill] keeper {j} (script cov {cov[kp]:.3f}) "
              f"mse={mse:.4f} ({time.time()-t0:.0f}s)", flush=True)

    TH = torch.as_tensor(np.stack(thetas), device=dev)
    tr_cov, tr_rooms = None, None
    tr_a = Arena(len(keep), args.data_games, seed=args.train_seed,
                 device=dev, fp16=args.fp16, merged_houses=args.data_sets)
    tr_cov, tr_rooms = replay_cov(TH, tr_a, args.hidden, args.ticks)
    ho_cov, ho_rooms = replay_cov(TH, ho, args.hidden, args.ticks)
    # per-member table
    _, mt = evaluate(tr_a, TH, args.hidden, args.ticks, w_dist=0.0,
                     w_coll=0.0, w_cov=0.0)
    pcov = (mt["cells"] / mt["cells_total"]).view(len(keep), rows).mean(1).cpu().numpy()
    _, mh = evaluate(ho, TH, args.hidden, args.ticks, w_dist=0.0,
                     w_coll=0.0, w_cov=0.0)
    phov = (mh["cells"] / mh["cells_total"]).view(len(keep), args.data_games).mean(1).cpu().numpy()
    print("\nmember  script_cov  bc_train_cov  bc_hold_cov   mse")
    for j, kp in enumerate(keep):
        print(f"  {j:>3}     {cov[kp]:.3f}        {pcov[j]:.3f}         "
              f"{phov[j]:.3f}        {meta[j]['mse']:.4f}")

    sel = [j for j in np.argsort(-pcov) if pcov[j] >= args.min_cov]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             thetas=np.stack([thetas[j] for j in sel]) if sel else np.zeros((0, genome_size(OBS_DIM, args.hidden)), np.float32),
             hidden=args.hidden, train_seed=args.train_seed,
             script_cov=np.array([meta[j]["script_cov"] for j in sel]),
             bc_train_cov=pcov[sel] if len(sel) else np.zeros(0),
             bc_hold_cov=phov[sel] if len(sel) else np.zeros(0))
    print(f"\n[bc_seed] KEPT {len(sel)}/{len(keep)} members "
          f"(min replay train-cov {args.min_cov}) -> {args.out}", flush=True)
    print(f"[bc_seed] population-level: train cov {tr_cov:.3f} rooms "
          f"{tr_rooms:.2f} | holdout cov {ho_cov:.3f} rooms {ho_rooms:.2f}")
    print("VERDICT:", "GO — MLP CAN hold explorer behaviour (>=4 keepers)"
          if len(sel) >= 4 else
          "BC-REPLAY-WEAK — MLP init cannot reproduce scripted coverage; "
          "try --hidden 128 before concluding memory-genome", flush=True)


if __name__ == "__main__":
    main()
