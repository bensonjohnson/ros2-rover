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
from .policy import (_param_shapes, genome_size, per_gene_scale,
                     PopulationNet)
from .evolve import evaluate   # the EXACT eval path evolution uses


# fixed projection for the scan-hash "RNG": given obs, the draw is
# DETERMINISTIC — a clock-timer/RNG-driven expert is unimitable (keeper-0
# lesson: mse 0.0022 yet replay froze; the MLP cannot see the expert's
# internal timer, so it averages its commits to ~0 = standstill).
_HASH_W = None


def make_scripted(per: dict, reps: int, device: str):
    """Parameterized MARKOVIAN expert: cmd = f(current scan72; personality
    constants). No timers, no RNG, no prev-scan state — everything the
    labels depend on is in the current observation, so a feedforward-back-
    boned student CAN represent it exactly (keeper-0 lesson: a clock-timer
    expert is unimitable, mse 0.0022 yet replay froze — the MLP averages
    unobservable commits to zero = standstill). Commitment is environmental:
    while a wall is near, 'blocked' stays true, so pivoting persists until
    the scan opens (the world clocks the policy, not hidden state).
    Personality (bias, mode) is constant per individual -> the distilled
    genome encodes it in its own weights.
    Modes: 0 cruise (openness+bias steering), 1 seek-openness (turn toward
    the more open side hard), 2 biased-orbit (constant differential)."""
    dev = torch.device(device)
    N = int(per["bias"].shape[0])
    B = N * reps

    def rep(x):
        return torch.as_tensor(x, device=dev).repeat_interleave(reps)

    bias, mode_i = rep(per["bias"]), rep(per["mode_i"])

    def step(obs, prev):
        s = obs[:, :NUM_BINS]
        front = s[:, :8].min(dim=1).values
        left = s[:, 9:36].mean(dim=1)
        right = s[:, 36:63].mean(dim=1)
        # low-noise deterministic hash of the observable geometry: flips
        # sign with room structure, stable while standing still
        hsh = torch.sin(37.13 * front) + 0.5 * torch.sin(11.7 * (left - right))
        piv_dir = torch.sign(hsh + 0.05 * bias)
        open_diff = (left - right).clamp(-1, 1)
        steer = 0.5 * open_diff + 0.5 * bias
        f = 0.4 + 0.6 * front.clamp(0, 1)
        m0 = torch.stack([f - steer, f + steer], dim=1)                  # cruise
        m1 = torch.stack([f + 0.8 * open_diff.clamp(min=0),
                          f + 0.8 * (-open_diff).clamp(min=0)], dim=1)    # seek open
        m2 = torch.stack([f + 0.35 * bias.clamp(min=0),
                          f + 0.35 * (-bias).clamp(min=0)], dim=1)        # orbit
        cmd = torch.where(mode_i.unsqueeze(1) == 0, m0,
                          torch.where(mode_i.unsqueeze(1) == 1, m1, m2))
        # blocked ahead -> pivot in place (persists while scan says so)
        blocked = front < 0.12
        pivot = torch.stack([0.15 * (piv_dir > 0) + (-0.15) * (piv_dir < 0),
                             -0.15 * (piv_dir > 0) + 0.15 * (piv_dir < 0)], dim=1)
        cmd = torch.where(blocked.unsqueeze(1), pivot, cmd)
        # wedged: front closed AND both sides closed -> back up
        wedged = (front < 0.06) & (torch.maximum(left, right) < 0.12)
        back = torch.stack([torch.full_like(bias, -0.5), torch.full_like(bias, 0.5)], dim=1)
        back2 = torch.stack([torch.full_like(bias, 0.5), torch.full_like(bias, -0.5)], dim=1)
        cmd = torch.where(wedged.unsqueeze(1),
                          torch.where((bias > 0).unsqueeze(1), back, back2),
                          cmd)
        return cmd.clamp(-1, 1)

    return step


def pack_theta(named: dict, n_in: int, hidden: int) -> np.ndarray:
    return np.concatenate([named[name].detach().cpu().numpy().ravel()
                           for name, _, _ in _param_shapes(n_in, hidden)]
                          ).astype(np.float32)


def distill(o: torch.Tensor, a: torch.Tensor, hidden: int,
            rng: np.random.Generator, epochs: int = 60, seg: int = 180,
            bs: int = 256, lr: float = 3e-3,
            init_theta: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """o [T,B,n_in], a [T,B,2] expert stream -> packed genome + final MSE.
    Student rolls its OWN hidden state over the expert observation stream
    (open-loop BC); obs already carries last action, so between command
    changes the student just needs to echo prev_act through recurrence.
    init_theta: packed genome to warm-start from (DAgger rounds retrain)."""
    T, B, _ = o.shape
    nseg_t = T // seg
    o_s = (o[:nseg_t * seg].view(nseg_t, seg, B, -1)
           .permute(2, 0, 1, 3).reshape(B * nseg_t, seg, -1))
    a_s = (a[:nseg_t * seg].view(nseg_t, seg, B, -1)
           .permute(2, 0, 1, 3).reshape(B * nseg_t, seg, -1))
    M = B * nseg_t

    named = {}
    if init_theta is not None:
        off = 0
        for name, shape, _ in _param_shapes(OBS_DIM, hidden):
            numel = int(np.prod(shape))
            named[name] = torch.as_tensor(
                init_theta[off:off + numel].reshape(shape).copy(),
                device=o.device).requires_grad_(True)
            off += numel
    else:
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
            tot += loss.detach().item() / seg
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
    ap.add_argument("--dagger", type=int, default=2,
                    help="DAgger rounds: roll the student, relabel with the "
                         "expert, retrain (fixes covariate shift; keeper-0 "
                         "froze despite mse 0.0022)")
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

    # ---- 3. distill each keeper (BC -> DAgger), 4. verify replay --------
    ho = Arena(len(keep), args.data_games, seed=args.holdout_seed,
               device=dev, fp16=args.fp16)
    tr_a = Arena(len(keep), args.data_games, seed=args.train_seed,
                 device=dev, fp16=args.fp16, merged_houses=args.data_sets)
    # single-individual roll arena: same houses/layout as one keeper's block
    roll_a = Arena(1, args.data_games, seed=args.train_seed, device=dev,
                   fp16=args.fp16, merged_houses=args.data_sets)

    def student_roll(th_j: np.ndarray, expert_j):
        """roll the current student (its OWN commands) through the train
        houses; return (student obs stream, expert labels on those states)."""
        TH1 = torch.as_tensor(th_j[None, :], device=dev)
        net = PopulationNet(TH1, OBS_DIM, args.hidden)
        net.bind(1, rows)
        hbuf = torch.zeros(1, rows, args.hidden, device=dev)

        def s_step(obs, prev):
            a, h_new = net.step(obs.view(1, rows, OBS_DIM), hbuf)
            hbuf.copy_(h_new)
            return a.view(rows, 2)
        r2 = {"obs": [], "cmd": []}
        roll_a.run_games(s_step, args.ticks, rec=r2)
        O2 = torch.stack(r2["obs"])              # [T, rows, n_in]
        del r2
        with torch.no_grad():
            # label one tick at a time: expert personality is per-env
            # [rows]; obs rows == envs at a fixed tick (Markovian: no prev)
            L2 = torch.stack([expert_j(O2[t], None)
                              for t in range(O2.shape[0])])
        return O2, L2

    thetas, meta = [], []
    for j, kp in enumerate(keep):
        single = {k: keep_per[k][j:j + 1] for k in keep_per}
        expert_j = make_scripted(single, rows, dev)
        o_j = O[:, j * rows:(j + 1) * rows].contiguous()   # [T, rows, n_in]
        a_j = A[:, j * rows:(j + 1) * rows].contiguous()
        th, mse = distill(o_j, a_j, args.hidden, rng, epochs=args.epochs,
                          seg=args.seg)
        # KEEP-BEST across stages: closed-loop coverage is cliffy (keeper-0:
        # mse 0.0036, script 0.170, replay 0.048 — one flipped pivot at a
        # choke tick sinks the run; a DAgger retrain can sink it further).
        # Verify by replay after every stage; ship the best genome seen.
        best_th, best_cov = th, replay_cov(
            torch.as_tensor(th[None, :], device=dev), roll_a,
            args.hidden, args.ticks)[0]
        # DAgger rounds: student visits states, expert labels them, retrain
        for dround in range(args.dagger):
            O2, L2 = student_roll(th, expert_j)
            o_mix = torch.cat([o_j, O2], dim=1)
            a_mix = torch.cat([a_j, L2], dim=1)
            th, mse = distill(o_mix, a_mix, args.hidden, rng,
                              epochs=max(15, args.epochs // 2), seg=args.seg,
                              init_theta=th)
            del O2, L2, o_mix, a_mix
            c2 = replay_cov(torch.as_tensor(th[None, :], device=dev),
                            roll_a, args.hidden, args.ticks)[0]
            if c2 > best_cov:
                best_th, best_cov = th, c2
        th = best_th
        thetas.append(th)
        meta.append({"script_cov": float(cov[kp]), "mse": mse})
        print(f"[distill] keeper {j} (script cov {cov[kp]:.3f}) "
              f"mse={mse:.4f} best-roll-cov={best_cov:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

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
