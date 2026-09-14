#!/usr/bin/env python3
"""Greenfield evolutionary run: evolve MLP policies from random noise to
room-explorers.

ES recipe: elitist (mu+lambda)-style loop — tournament selection, uniform
crossover, per-gene Gaussian mutation with per-individual log-normal
self-adapted sigma (in init-scale units), plus random immigrants each
generation to fight premature convergence. Fitness = mean over G paired
houses of ground-truth room coverage (+dist, -collisions); all individuals
see the SAME houses and noise streams within a generation (paired test),
train houses rotate among K seeds across generations, and a fixed holdout
arena scores the final champion — plus scripted baselines for reference.

    python3 -m evo.evolve --pop 64 --games 32 --gens 40 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .arena import Arena, OBS_DIM, fitness
from .policy import (PopulationNet, genome_size, per_gene_scale,
                     sample_population)


def evaluate(arena: Arena, thetas: torch.Tensor, hidden: int,
             ticks: int, w_dist: float = 0.03,
             w_coll: float = 0.25) -> tuple[torch.Tensor, dict]:
    """Run every individual in every house; returns per-individual fitness
    (mu) and the raw per-env metrics for the best individual."""
    P = arena.P
    G_eff = arena.G * arena.G_sets
    net = PopulationNet(thetas, OBS_DIM, hidden)
    net.bind(P, G_eff)
    state = {"h": torch.zeros(P, G_eff, hidden, device=thetas.device)}

    def policy_step(obs, prev_act):
        o = obs.view(P, G_eff, OBS_DIM)
        a, state["h"] = net.step(o, state["h"])
        return a.view(P * G_eff, 2)

    metrics = arena.run_games(policy_step, ticks)
    fit = fitness(metrics, w_dist=w_dist, w_coll=w_coll).view(P, G_eff).mean(dim=1)
    return fit, metrics


class GraphRunner:
    """CUDA-graph evaluator for one arena: the whole rollout — physics,
    gate, policy — is captured ONCE and replayed per generation. The
    launch-bound regime on the GB10 is thousands of tiny kernels per game
    (raycast is only ~30 segments wide); replay replaces them with one
    graphLaunch. Weights flow through a STATIC buffer (copy_ outside the
    graph), which is the only way a parameter-changing graph is legal."""

    def __init__(self, arena: Arena, P: int, hidden: int):
        from .policy import genome_size
        self.arena, self.P, self.hidden = arena, P, hidden
        self.G_eff = arena.G * arena.G_sets      # envs per individual
        N = genome_size(OBS_DIM, hidden)
        self.theta_buf = torch.zeros(P, N, device=arena.device)
        self.net = PopulationNet(self.theta_buf, OBS_DIM, hidden)
        self.net.bind(P, self.G_eff)
        self.h = torch.zeros(P, self.G_eff, hidden, device=arena.device)
        arena._reset_hooks.append(self.h.zero_)
        # NOTE: env randomness goes through the DEFAULT CUDA generator
        # (see _DefaultRNGMixin) — natively graph-safe; on replay each game
        # sees the same captured noise stream, which is what we want
        # (paired comparison across individuals).
        self._built = False

    def _step(self, obs, prev_act):
        a, h_new = self.net.step(
            obs.view(self.P, self.G_eff, OBS_DIM), self.h)
        self.h.copy_(h_new)   # in-place: keeps the hidden-state address
                              # stable across graph replays (a rebind would
                              # leak state between games and orphan the
                              # reset hook)
        return a.view(self.P * self.G_eff, 2)

    def run(self, thetas: torch.Tensor, ticks: int,
            w_dist: float, w_coll: float):
        self.theta_buf.copy_(thetas)
        metrics = self.arena.run_games(self._step, ticks, graph=True)
        fit = fitness(metrics, w_dist=w_dist,
                      w_coll=w_coll).view(self.P, self.G_eff).mean(dim=1)
        return fit, metrics


def evolve(args):
    dev = args.device
    hidden, P, G = args.hidden, args.pop, args.games
    N = genome_size(OBS_DIM, hidden)
    scale = torch.as_tensor(per_gene_scale(OBS_DIM, hidden), device=dev)

    rng = np.random.default_rng(args.seed)
    thetas = torch.as_tensor(
        sample_population(P, OBS_DIM, hidden, rng), device=dev)
    sigma = torch.full((P, 1), args.sigma0, device=dev)

    # ONE merged train arena (P x [K sets x G] houses) + graph: this stack
    # faults with >1 live CUDA graph (gmbisect T3-T5), so house rotation
    # happens INSIDE the graph as extra env columns, not as separate arenas.
    train = Arena(P, G, seed=args.train_seed, device=dev, fp16=args.fp16,
                  merged_houses=args.train_rotations)
    G_eff = G * args.train_rotations
    # holdout stays eager (scored 1-in-N gens — cheap) on its own worlds
    holdout = Arena(P, G, seed=args.holdout_seed, device=dev,
                    fp16=args.fp16)

    if args.graph:
        runner = GraphRunner(train, P, hidden)

        def score(is_train, th):
            if is_train:
                return runner.run(th, args.ticks, args.w_dist, args.w_coll)
            return evaluate(holdout, th, hidden, args.ticks,
                            w_dist=args.w_dist, w_coll=args.w_coll)
    else:
        def score(is_train, th):
            a = train if is_train else holdout
            return evaluate(a, th, hidden, args.ticks, w_dist=args.w_dist,
                            w_coll=args.w_coll)

    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "evolution.jsonl"), "a")

    tau = 0.4
    t0 = time.time()
    best_fit_ever, best_theta = -np.inf, thetas[0].cpu().clone()

    for gen in range(args.gens):
        fit, metrics = score(True, thetas)
        fit_np = fit.cpu().numpy()

        order = np.argsort(-fit_np)
        if fit_np[order[0]] > best_fit_ever:
            best_fit_ever = float(fit_np[order[0]])
            best_theta = thetas[order[0]].cpu().clone()
            np.savez(os.path.join(args.out_dir, "best_genome.npz"),
                     thetas=best_theta.numpy(), hidden=hidden,
                     fitness=best_fit_ever, gen=gen)

        # --- report every --report-every gens (holdout score included) ----
        if gen % args.report_every == 0 or gen == args.gens - 1:
            hf, hm = score(False, thetas)
            b = int(hf.argmax())
            sl = slice(b * G, (b + 1) * G)          # holdout: G_sets=1
            rec = {
                "gen": gen, "elapsed_s": round(time.time() - t0, 1),
                "fit_best": round(float(fit_np[order[0]]), 4),
                "fit_med": round(float(np.median(fit_np)), 4),
                "sigma_mean": round(float(sigma.mean()), 5),
                "holdout_best": round(float(hf[b]), 4),
                "holdout_rooms": round(float(hm["rooms"][sl].mean()), 2),
                "holdout_dist": round(float(hm["dist_m"][sl].mean()), 1),
                "holdout_coll": round(float(hm["collisions"][sl].mean()), 1),
            }
            print(f"gen {gen:>4d}  best={rec['fit_best']:.4f} "
                  f"med={rec['fit_med']:.4f}  "
                  f"HOLDOUT best={rec['holdout_best']:.4f} "
                  f"rooms={rec['holdout_rooms']:.2f} "
                  f"dist={rec['holdout_dist']:.1f}m "
                  f"coll={rec['holdout_coll']:.1f}  "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        # ---------------- reproduction -----------------------------------
        n_elite = max(2, P // 4)
        elite_idx = order[:n_elite]
        mu = max(2, P // 4)                        # recombination parents
        survivors = torch.as_tensor(elite_idx, device=dev)

        n_immigrant = max(1, P // 10)
        n_child = P - n_elite - n_immigrant

        # tournament selection from the whole scored pop (elitism already
        # guarantees the top mu survive untouched)
        def tournament(k=3):
            cand = rng.integers(0, P, size=(n_child, k))
            return cand[np.arange(n_child),
                        fit_np[cand].argmax(axis=1)]

        p1 = torch.as_tensor(tournament(), device=dev)
        p2 = torch.as_tensor(tournament(), device=dev)
        mix = (torch.rand(n_child, 1, device=dev) < 0.5).float()
        child = p1.unsqueeze(1).expand(-1, N) * mix \
            + p2.unsqueeze(1).expand(-1, N) * (1 - mix)

        # self-adapted mutation: sigma' = sigma * exp(tau*N + tauN*N_i)
        s_p = sigma[p1].maximum(sigma[p2])
        s_child = s_p * torch.exp(tau * torch.randn(n_child, 1, device=dev)
                                  + 0.1 * torch.randn(n_child, 1,
                                                       device=dev))
        s_child = s_child.clamp(1e-4, 1.0)
        child = child + s_child * scale.unsqueeze(0) \
            * torch.randn(n_child, N, device=dev)

        immigrants = torch.as_tensor(
            sample_population(n_immigrant, OBS_DIM, hidden, rng),
            device=dev)
        sig_imm = torch.full((n_immigrant, 1), args.sigma0, device=dev)

        thetas = torch.cat([thetas[survivors], child, immigrants], dim=0)
        sigma = torch.cat([sigma[survivors], s_child, sig_imm], dim=0)

    np.savez(os.path.join(args.out_dir, "final_population.npz"),
             thetas=thetas.cpu().numpy(), sigma=sigma.cpu().numpy(),
             hidden=hidden)
    log_f.close()
    print(f"\nbest fitness seen: {best_fit_ever:.4f} -> "
          f"{args.out_dir}/best_genome.npz", flush=True)
    return best_theta, best_fit_ever


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--games", type=int, default=32,
                    help="houses per individual (paired across pop)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=3600,
                    help="sim ticks per game (3600 = 4 sim-min)")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sigma0", type=float, default=0.25,
                    help="initial mutation size in init-sigma units")
    ap.add_argument("--train-rotations", type=int, default=4,
                    help="train-house seed rotation length")
    ap.add_argument("--train-seed", type=int, default=40_000)
    ap.add_argument("--holdout-seed", type=int, default=777_000,
                    help="never used for selection; match eval_checkpoints")
    ap.add_argument("--report-every", type=int, default=5)
    ap.add_argument("--w-dist", type=float, default=0.03,
                    help="distance term weight; LOWER (e.g. 0.005) if the "
                    "run plateaus as a fast wall-hugger — rooms dominates then")
    ap.add_argument("--w-coll", type=float, default=0.25)
    ap.add_argument("--fp16", action="store_true",
                    help="fp16 raycast (GB10/Spark fast path; ~2x on the "
                    "bandwidth-bound scan, sensing-only)")
    ap.add_argument("--graph", action="store_true",
                    help="capture the full rollout as a CUDA graph (one "
                    "replay per generation; removes kernel-launch overhead)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="evo_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
