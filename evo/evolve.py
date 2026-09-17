#!/usr/bin/env python3
"""Greenfield evolutionary run: evolve MLP policies from random noise to
room-explorers.

ES recipe: elitist (mu+lambda)-style loop — tournament selection (the child
copies ONE parent; there is no crossover), per-gene Gaussian mutation with
per-individual log-normal self-adapted sigma (in init-scale units), plus
random immigrants each generation to fight premature convergence.

Mutation size (run 9 fix): runs 1-8 used sigma0 0.25, tau 0.4, sigma'
inherited as max(parent sigmas), clamp 1.0 — an upward-ratcheting step
that reached a median 0.36 (quartile 0.81) init-sigmas on EVERY gene. The
evolved controllers are bang-bang (|a| ~0.97) and cliff between sigma 0.1
(open-loop action change 0.03) and 0.25 (0.27; 0.5 -> 0.62), so most
children were behaviourally unrelated to their parent = random search
around untouched elites. Defaults now: sigma0 0.03, sigma <= 0.15,
tau = 1/sqrt(N), sigma inherited from the copied parent. --legacy-mutation
restores the runs 1-8 step-size rule (NOT the parent-gather bug below).

Inheritance (run 9 fix): runs 1-8 built children from the parent INDEX
instead of the parent genome, so nothing was ever inherited — see the
reproduction block. Fitness = mean over G paired
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
from .policy import (PopulationNet, genome_meta, genome_size,
                     per_gene_scale, read_meta, sample_population)


def evaluate(arena: Arena, thetas: torch.Tensor, hidden: int,
             ticks: int, w_dist: float = 0.03,
             w_coll: float = 0.25,
             w_cov: float = 0.0, obs: str = "v1",
             action: str = "lr") -> tuple[torch.Tensor, dict]:
    """Run every individual in every house; returns per-individual fitness
    (mu) and the raw per-env metrics for the best individual."""
    P = arena.P
    G_eff = arena.G * arena.G_sets
    net = PopulationNet(thetas, OBS_DIM, hidden, obs=obs, action=action)
    net.bind(P, G_eff)
    state = {"h": torch.zeros(P, G_eff, hidden, device=thetas.device)}

    def policy_step(obs, prev_act):
        o = obs.view(P, G_eff, OBS_DIM)
        a, state["h"] = net.step(o, state["h"])
        return a.view(P * G_eff, 2)

    metrics = arena.run_games(policy_step, ticks)
    fit = fitness(metrics, w_dist=w_dist, w_coll=w_coll,
                  w_cov=w_cov).view(P, G_eff).mean(dim=1)
    return fit, metrics


class GraphRunner:
    """CUDA-graph evaluator for one arena: the whole rollout — physics,
    gate, policy — is captured ONCE and replayed per generation. The
    launch-bound regime on the GB10 is thousands of tiny kernels per game
    (raycast is only ~30 segments wide); replay replaces them with one
    graphLaunch. Weights flow through a STATIC buffer (copy_ outside the
    graph), which is the only way a parameter-changing graph is legal."""

    def __init__(self, arena: Arena, P: int, hidden: int, obs: str = "v1",
                 action: str = "lr"):
        from .policy import genome_size
        self.arena, self.P, self.hidden = arena, P, hidden
        self.G_eff = arena.G * arena.G_sets      # envs per individual
        N = genome_size(OBS_DIM, hidden)
        self.theta_buf = torch.zeros(P, N, device=arena.device)
        self.net = PopulationNet(self.theta_buf, OBS_DIM, hidden, obs=obs,
                                 action=action)
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
            w_dist: float, w_coll: float, w_cov: float = 0.0):
        self.theta_buf.copy_(thetas)
        metrics = self.arena.run_games(self._step, ticks, graph=True)
        fit = fitness(metrics, w_dist=w_dist,
                      w_coll=w_coll,
                      w_cov=w_cov).view(self.P, self.G_eff).mean(dim=1)
        return fit, metrics


def reproduce(thetas: torch.Tensor, sigma: torch.Tensor, fit_np: np.ndarray,
              order: np.ndarray, scale: torch.Tensor,
              rng: np.random.Generator, *, n_elite: int, n_immigrant: int,
              tau: float, sigma_max: float, sigma0: float, hidden: int,
              legacy: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Next population, laid out [survivors | children | immigrants].

    Survivors = top n_elite untouched. Each child copies ONE tournament
    parent's genome and gets per-gene Gaussian mutation in init-scale
    units with a log-normal self-adapted per-individual sigma.

    Runs 1-8 built children as `p1.unsqueeze(1).expand(-1, N) * mix`, i.e.
    the parent INDEX (0..P-1) broadcast as every gene: children were
    constant vectors + noise, nothing was inherited, and the ES reduced to
    elitism over gen-0 samples and random immigrants (evo.test_reproduce).
    """
    P, N = thetas.shape
    dev = thetas.device
    n_child = P - n_elite - n_immigrant
    survivors = torch.as_tensor(order[:n_elite], device=dev)

    def tournament(k=3):
        cand = rng.integers(0, P, size=(n_child, k))
        return cand[np.arange(n_child), fit_np[cand].argmax(axis=1)]

    par = torch.as_tensor(tournament(), device=dev)
    if legacy:
        # runs 1-8 step-size rule (max of two parents, tau 0.4, <= 1.0)
        par2 = torch.as_tensor(tournament(), device=dev)
        s_child = sigma[par].maximum(sigma[par2]) * torch.exp(
            tau * torch.randn(n_child, 1, device=dev)
            + 0.1 * torch.randn(n_child, 1, device=dev))
    else:
        s_child = sigma[par] * torch.exp(
            tau * torch.randn(n_child, 1, device=dev))
    s_child = s_child.clamp(1e-4, sigma_max)
    child = thetas[par] + s_child * scale.unsqueeze(0) \
        * torch.randn(n_child, N, device=dev)

    immigrants = torch.as_tensor(
        sample_population(n_immigrant, OBS_DIM, hidden, rng), device=dev)
    sig_imm = torch.full((n_immigrant, 1), sigma0, device=dev)
    return (torch.cat([thetas[survivors], child, immigrants], dim=0),
            torch.cat([sigma[survivors], s_child, sig_imm], dim=0))


class OES:
    """OpenAI-ES (Salimans et al. 2017) over the flat genome, as an
    alternative to the truncation GA (--algo oes).

    Population slot 0 = the centre mu; slots 1..2n = mirrored pairs
    mu +/- sigma*scale*eps (antithetic sampling cancels the fitness
    baseline); a leftover odd slot re-scores the best genome seen. All 2n
    perturbed scores become centred ranks in [-0.5, 0.5] (fitness shaping:
    invariant to the w_cov/w_dist scale, robust to collision outliers), and
    the search-gradient estimate g = sum_i (r+_i - r-_i) eps_i / (2 n sigma)
    drives Adam on z = mu / scale (per-gene init-scale units, like the GA's
    sigma) with L2 decay. Every one of the P rollouts feeds one gradient —
    the GA keeps the top quarter and discards the rest."""

    def __init__(self, mu: torch.Tensor, scale: torch.Tensor, P: int,
                 sigma: float, lr: float, l2: float = 0.005):
        self.z = (mu / scale).clone()
        self.scale, self.P = scale, P
        self.sigma, self.lr, self.l2 = sigma, lr, l2
        self.n = (P - 1) // 2
        self.m = torch.zeros_like(self.z)
        self.v = torch.zeros_like(self.z)
        self.t = 0
        self.eps = None

    @property
    def mu(self) -> torch.Tensor:
        return self.z * self.scale

    def ask(self, best: torch.Tensor | None) -> torch.Tensor:
        N = self.z.numel()
        self.eps = torch.randn(self.n, N, device=self.z.device)
        d = self.sigma * self.eps * self.scale
        rows = [self.mu[None], self.mu + d, self.mu - d]
        if self.P - 1 - 2 * self.n:
            rows.append((best if best is not None else self.mu)[None])
        return torch.cat(rows, dim=0)

    def tell(self, fit_np: np.ndarray) -> float:
        n = self.n
        f = fit_np[1:1 + 2 * n]
        ranks = np.empty(2 * n, dtype=np.float32)
        ranks[np.argsort(f)] = np.arange(2 * n, dtype=np.float32)
        ranks = ranks / (2 * n - 1) - 0.5
        w = torch.as_tensor(ranks[:n] - ranks[n:], device=self.z.device)
        g = (w[:, None] * self.eps).sum(0) / (2 * n * self.sigma)
        g = g - self.l2 * self.z                 # ascent + decay
        self.t += 1
        b1, b2 = 0.9, 0.999
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        step = self.lr * mh / (vh.sqrt() + 1e-8)
        self.z.add_(step)
        return float(step.norm() / (self.z.norm() + 1e-12))


def evolve(args):
    dev = args.device
    hidden, P, G = args.hidden, args.pop, args.games
    N = genome_size(OBS_DIM, hidden)
    scale = torch.as_tensor(per_gene_scale(OBS_DIM, hidden), device=dev)
    meta = genome_meta(args.obs, args.action)

    rng = np.random.default_rng(args.seed)
    thetas = torch.as_tensor(
        sample_population(P, OBS_DIM, hidden, rng), device=dev)
    sigma = torch.full((P, 1), args.sigma0, device=dev)
    if args.legacy_mutation:
        tau, sigma_max = 0.4, 1.0
    else:
        tau = args.tau if args.tau > 0 else 1.0 / np.sqrt(N)
        sigma_max = args.sigma_max
    print(f"[mutation] {'legacy' if args.legacy_mutation else 'v9'}: "
          f"N={N} sigma0={args.sigma0} tau={tau:.4f} sigma_max={sigma_max}",
          flush=True)

    if args.seed_from:
        # Curriculum/warm-restart hookup: final_population.npz (thetas
        # [P,N], sigma) or best_genome.npz (single row) from an earlier run.
        d = np.load(args.seed_from)
        assert int(d["hidden"]) == hidden, \
            f"--hidden {hidden} != file hidden {int(d['hidden'])}"
        assert read_meta(d) == (args.obs, args.action), \
            f"--seed-from genome modes {read_meta(d)} != run modes"
        seed_th = torch.as_tensor(d["thetas"], device=dev)
        if seed_th.ndim == 1:
            seed_th = seed_th.unsqueeze(0)
        n_seed = min(P, seed_th.shape[0])
        if seed_th.shape[0] > P:
            raise SystemExit(f"--seed-from has {seed_th.shape[0]} rows "
                             f"(>pop {P}); match --pop to the source run")
        thetas[:n_seed] = seed_th[:n_seed]
        carried = (args.keep_seed_sigma and "sigma" in d
                   and d["sigma"].shape[0] >= n_seed)
        if carried:
            sigma[:n_seed] = torch.as_tensor(
                d["sigma"][:n_seed], device=dev).view(n_seed, 1) \
                .clamp(max=sigma_max)
        print(f"[warm-start] {n_seed}/{P} from {args.seed_from} "
              f"(rest fresh, sigma "
              f"{'carried over' if carried else f'reset to {args.sigma0}'})",
              flush=True)

    if getattr(args, "bc_from", ""):
        # BC seeding (run 8): scripted-explorer distilled genomes occupy the
        # FIRST slots of the initial population — fresh samples fill the
        # rest. Unlike --seed_from this REPLACES random rows (the point is
        # to seed the search inside the explorer basin, not extend an old
        # run), and sigma gets a modest floor so the seeds do not freeze.
        d = np.load(args.bc_from)
        assert int(d["hidden"]) == hidden, \
            f"--hidden {hidden} != bc file hidden {int(d['hidden'])}"
        assert read_meta(d) == (args.obs, args.action), \
            f"--bc-from genome modes {read_meta(d)} != run modes"
        assert int(d.get("train_seed", args.train_seed)) == args.train_seed, \
            "bc file distilled on a different train seed — match --train-seed"
        bc = torch.as_tensor(d["thetas"], device=dev)
        n_bc = min(P, bc.shape[0])
        if bc.shape[0] == 0:
            raise SystemExit("bc file has 0 kept members (min-cov gate "
                             "rejected all — inspect bc_seed output)")
        thetas[:n_bc] = bc[:n_bc]
        # runs 1-8 floored seeds at 0.15 — past the behaviour cliff
        sigma[:n_bc] = (max(args.sigma0, 0.15) if args.legacy_mutation
                        else args.sigma0)
        print(f"[bc-seed] {n_bc}/{P} distilled explorers in slot 0..{n_bc-1} "
              f"(rest fresh random)", flush=True)

    # ---- building mode: per-level house pools + curriculum state --------
    HG = G * args.train_rotations
    level = args.level
    pools, caps, ho_worlds, val_worlds = {}, None, None, None
    best_val, best_val_theta = -np.inf, None
    world_rng = np.random.default_rng(args.seed + 99)
    if args.world == "buildings":
        from pnn_sim.buildings import LEVELS
        from .arena import world_caps
        from .worlds import (HOLDOUT_SEED, VAL_SEED, build_pool, level_seed,
                             summarize)
        top = max(LEVELS) if args.curriculum else level
        for lv in range(level, top + 1):
            pools[lv] = build_pool(lv, args.pool, level_seed(args.train_seed,
                                                             lv),
                                   workers=args.pool_workers)
            print(f"[worlds] L{lv} pool: {summarize(pools[lv])}", flush=True)
        ho_worlds = build_pool(max(LEVELS), G, HOLDOUT_SEED,
                               workers=args.pool_workers)
        # VALIDATION set (level 3, own seed): picks the champion. Train
        # fitness on a fresh 64-house draw per gen is too noisy to pick one
        # (run 11: train argmax deep-eval'd at 0.14 with 90.8 collisions)
        # and the holdout must stay unselected.
        val_worlds = build_pool(max(LEVELS), G, VAL_SEED,
                                workers=args.pool_workers)
        allw = [w for p_ in pools.values() for w in p_]
        caps = world_caps(allw)
        print(f"[worlds] caps (segments, rects, doors) = {caps}; building "
              f"holdout: {summarize(ho_worlds)}", flush=True)

    def sample_houses(lv):
        idx = world_rng.choice(len(pools[lv]), size=HG,
                               replace=len(pools[lv]) < HG)
        return [pools[lv][i] for i in idx]

    def build_arenas(door_w, holdout_door=(0.7, 1.0)):
        # ONE merged train arena (P x [K sets x G] houses) + graph: this stack
        # faults with >1 live CUDA graph (gmbisect T3-T5), so house rotation
        # happens INSIDE the graph as extra env columns, not as separate
        # arenas. Caller must drop the previous arena/runner BEFORE calling.
        bkw = {}
        if args.world == "buildings":
            bkw = dict(worlds=sample_houses(level), caps=caps)
        tr = Arena(P, G, seed=args.train_seed, device=dev, fp16=args.fp16,
                   merged_houses=args.train_rotations,
                   door_w_range=(door_w, door_w) if door_w else (0.7, 1.0),
                   cells=not args.no_cells, every_cover=args.every_cover,
                   fused=args.fused, compile=args.compile,
                   gate_obs=args.obs == "v2", **bkw)
        ho = Arena(P, G, seed=args.holdout_seed, device=dev,
                   fp16=args.fp16, door_w_range=holdout_door,
                   cells=not args.no_cells, every_cover=args.every_cover,
                   fused=args.fused, compile=args.compile,
                   gate_obs=args.obs == "v2")
        if ho_worlds is not None:
            # building holdout (level 3, never selected on): eager side eval
            hob = Arena(P, G, seed=args.holdout_seed, device=dev,
                        fp16=args.fp16, cells=not args.no_cells,
                        every_cover=args.every_cover, fused=args.fused,
                        gate_obs=args.obs == "v2", worlds=ho_worlds)
            hval = Arena(P, G, seed=args.holdout_seed, device=dev,
                         fp16=args.fp16, cells=not args.no_cells,
                         every_cover=args.every_cover, fused=args.fused,
                         gate_obs=args.obs == "v2", worlds=val_worlds)
        if not args.graph:
            def score(is_train, th):
                a = tr if is_train is True else (
                    hob if is_train == "hob" else
                    hval if is_train == "val" else ho)
                return evaluate(a, th, hidden, args.ticks,
                                w_dist=args.w_dist, w_coll=args.w_coll,
                                w_cov=args.w_cov, obs=args.obs,
                                action=args.action)
            return tr, ho, score

        rn = GraphRunner(tr, P, hidden, obs=args.obs, action=args.action)

        def score(is_train, th):
            if is_train is True:
                return rn.run(th, args.ticks, args.w_dist, args.w_coll,
                              args.w_cov)
            return evaluate(hob if is_train == "hob" else
                            hval if is_train == "val" else ho, th, hidden,
                            args.ticks,
                            w_dist=args.w_dist, w_coll=args.w_coll,
                            w_cov=args.w_cov, obs=args.obs,
                            action=args.action)
        return tr, ho, score

    train, holdout, score = build_arenas(args.door_w)
    G_eff = G * args.train_rotations

    oes = None
    if args.algo == "oes":
        # centre = slot 0 of the (possibly warm-started) population
        oes = OES(thetas[0].clone(), scale, P, sigma=args.oes_sigma,
                  lr=args.oes_lr, l2=args.oes_l2)
        thetas = oes.ask(None)
        sigma.fill_(args.oes_sigma)
        print(f"[oes] pairs={oes.n} sigma={args.oes_sigma} lr={args.oes_lr}"
              f" l2={args.oes_l2}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "evolution.jsonl"), "a")
    gen_f = open(os.path.join(args.out_dir, "gens.jsonl"), "a")

    t0 = time.time()
    best_fit_ever, best_theta = -np.inf, thetas[0].cpu().clone()

    # population layout after reproduction: [survivors | children | immigrants]
    n_elite = max(2, P // 4)
    n_immigrant = max(1, P // 10)
    n_child = P - n_elite - n_immigrant
    child_stats = []        # per gen since last report: (win frac, d median)
    streak = 0              # curriculum: consecutive gens over advance_at

    for gen in range(args.gens):
        # --- doorway curriculum: rebuild arenas + graph at the phase cut --
        if (args.anneal_gen and gen == args.anneal_gen
                and args.door_w != args.anneal_door_w):
            np.savez(os.path.join(args.out_dir, "phase1_best_genome.npz"),
                     thetas=best_theta.numpy(), hidden=hidden, **meta,
                     fitness=best_fit_ever, gen=gen - 1, door_w=args.door_w)
            print(f"[curriculum] gen {gen}: door {args.door_w} -> "
                  f"{args.anneal_door_w} m; recapturing graph", flush=True)
            # Release the live graph EXPLICITLY first: the default CUDA
            # generator stays graph-registered until CUDAGraph.reset(), or
            # the next eager randn (warmup) dies with "Offset increment
            # outside graph capture". Then drop all refs (the graph is also
            # reachable via score's closure) so the pool is reclaimable.
            train.drop_graphs()
            del train, holdout, score
            del fit, metrics, fit_np, order
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            args.door_w = args.anneal_door_w
            train, holdout, score = build_arenas(args.door_w)
            # fitness across door widths is not comparable: restart tracker
            best_fit_ever = -np.inf

        if (args.world == "buildings" and gen > 0
                and gen % args.resample_every == 0):
            train.load_worlds(sample_houses(level))   # same buffers: graph ok
        fit, metrics = score(True, thetas)
        fit_np = fit.cpu().numpy()

        order = np.argsort(-fit_np)
        if gen > 0 and oes is None:
            # offspring quality on the SAME houses/noise as the re-scored
            # survivors: are mutants neighbours of their parents, or noise?
            f_surv = fit_np[:n_elite]
            f_child = fit_np[n_elite:n_elite + n_child]
            child_stats.append((
                float((f_child > np.median(f_surv)).mean()),
                float(np.median(f_child) - np.median(f_surv))))
        if fit_np[order[0]] > best_fit_ever:
            best_fit_ever = float(fit_np[order[0]])
            best_theta = thetas[order[0]].cpu().clone()
            np.savez(os.path.join(args.out_dir, "best_genome.npz"),
                     thetas=best_theta.numpy(), hidden=hidden, **meta,
                     fitness=best_fit_ever, gen=gen)

        # --- honest per-gen train readout (free: metrics already on hand) --
        # The ELITE (top n_elite by train fitness) is what selection keeps;
        # train_rooms/train_cells below average the WHOLE pop (immigrants +
        # mutants) and hide elite movement.
        def per_ind(m, n_env):
            """per-individual means [P] of rooms/cells-frac/dist/coll"""
            def mean(v):
                return v.view(P, n_env).mean(dim=1).cpu().numpy()
            return {"rooms": mean(m["rooms"]),
                    "cells": mean(m["cells"] / m["cells_total"]),
                    "dist": mean(m["dist_m"]),
                    "coll": mean(m["collisions"])}
        el = order[:n_elite]
        champ = int(order[0])
        tr = per_ind(metrics, G_eff)
        trec = {"gen": gen, "elapsed_s": round(time.time() - t0, 1),
                "fit_best": round(float(fit_np[champ]), 4),
                "fit_elite": round(float(fit_np[el].mean()), 4),
                "fit_med": round(float(np.median(fit_np)), 4),
                "elite_cells": round(float(tr["cells"][el].mean()), 4),
                "elite_rooms": round(float(tr["rooms"][el].mean()), 3),
                "elite_dist": round(float(tr["dist"][el].mean()), 1),
                "elite_coll": round(float(tr["coll"][el].mean()), 2),
                "champ_cells": round(float(tr["cells"][champ]), 4),
                "sigma_med": round(float(sigma.median()), 4)}
        if oes is not None:
            trec["center_fit"] = round(float(fit_np[0]), 4)
            trec["center_cells"] = round(float(tr["cells"][0]), 4)
        elif gen > 0:
            trec["child_win"] = round(child_stats[-1][0], 3)
            trec["child_dmed"] = round(child_stats[-1][1], 4)
        if args.world == "buildings":
            rooms_env = metrics["rooms"].view(P, G_eff)[el]
            cross_rate = float((rooms_env >= 2).float().mean())
            dc = metrics["door_crossings"].view(P, G_eff)[el]
            du = (metrics["doors_used"] / metrics["doors_total"].clamp(min=1)
                  ).view(P, G_eff)[el]
            rf = (metrics["rooms"] / metrics["rooms_total"]).view(
                P, G_eff)[el]
            trec.update({"level": level,
                         "elite_cross_rate": round(cross_rate, 3),
                         "elite_rooms_frac": round(float(rf.mean()), 4),
                         "elite_door_crossings": round(float(dc.mean()), 2),
                         "elite_doors_used_frac": round(float(du.mean()), 4)})
            print(f"      L{level} elite cross_rate={cross_rate:.3f} "
                  f"rooms_frac={float(rf.mean()):.3f} "
                  f"door_x={float(dc.mean()):.2f} "
                  f"doors_used={float(du.mean()):.3f}", flush=True)
            if args.curriculum and level < max(pools):
                streak = streak + 1 if cross_rate >= args.advance_at else 0
                if streak >= args.advance_patience:
                    np.savez(os.path.join(args.out_dir,
                                          f"level{level}_best_genome.npz"),
                             thetas=best_theta.numpy(), hidden=hidden,
                             **meta, fitness=best_fit_ever, gen=gen)
                    print(f"[curriculum] gen {gen}: L{level} cross_rate >= "
                          f"{args.advance_at} for {streak} gens -> "
                          f"L{level + 1}", flush=True)
                    trec["advance_to"] = level + 1
                    level += 1
                    streak = 0
                    best_fit_ever = -np.inf     # fitness not comparable
                    train.load_worlds(sample_houses(level))
        gen_f.write(json.dumps(trec) + "\n")
        gen_f.flush()
        print(f"  g{gen:>3d} elite fit={trec['fit_elite']:.4f} "
              f"cells={trec['elite_cells']:.4f} rooms={trec['elite_rooms']:.3f} "
              f"dist={trec['elite_dist']:.1f} coll={trec['elite_coll']:.2f} "
              f"sigma~{trec['sigma_med']:.4f} "
              + (f"center fit={trec['center_fit']:.4f} "
                 f"cells={trec['center_cells']:.4f}" if oes is not None else
                 f"child_win={trec.get('child_win', float('nan')):.3f}"),
              flush=True)

        # --- report every --report-every gens (holdout score included) ----
        if gen % args.report_every == 0 or gen == args.gens - 1:
            hf, hm = score(False, thetas)
            b = int(hf.argmax())
            sl = slice(b * G, (b + 1) * G)          # holdout: G_sets=1
            # holdout_* keys pick the argmax ON holdout (selection on test,
            # kept for comparability with runs 1-8); ho_champ_* is the
            # train champion, ho_elite_* the train elite — the honest ones.
            ho = per_ind(hm, G)
            cs = np.array(child_stats) if child_stats else None
            child_stats.clear()
            rec = {
                "gen": gen, "elapsed_s": round(time.time() - t0, 1),
                "door_w": args.door_w,
                "fit_best": round(float(fit_np[order[0]]), 4),
                "fit_med": round(float(np.median(fit_np)), 4),
                "train_rooms": round(float(metrics["rooms"].mean()), 2),
                "train_cells": round(float(
                    (metrics["cells"] / metrics["cells_total"]).mean()), 3),
                "sigma_mean": round(float(sigma.mean()), 5),
                "holdout_best": round(float(hf[b]), 4),
                "holdout_rooms": round(float(hm["rooms"][sl].mean()), 2),
                "holdout_cells": round(float(
                    (hm["cells"] / hm["cells_total"])[sl].mean()), 3),
                "holdout_dist": round(float(hm["dist_m"][sl].mean()), 1),
                "holdout_coll": round(float(hm["collisions"][sl].mean()), 1),
                "elite_cells": trec["elite_cells"],
                "elite_rooms": trec["elite_rooms"],
                "ho_champ_fit": round(float(hf[champ]), 4),
                "ho_champ_rooms": round(float(ho["rooms"][champ]), 2),
                "ho_champ_cells": round(float(ho["cells"][champ]), 3),
                "ho_champ_dist": round(float(ho["dist"][champ]), 1),
                "ho_champ_coll": round(float(ho["coll"][champ]), 1),
                "ho_elite_fit": round(float(hf[el].mean()), 4),
                "ho_elite_rooms": round(float(ho["rooms"][el].mean()), 3),
                "ho_elite_cells": round(float(ho["cells"][el].mean()), 4),
            }
            if cs is not None:
                rec["child_win"] = round(float(cs[:, 0].mean()), 3)
                rec["child_dmed"] = round(float(cs[:, 1].mean()), 4)
            if val_worlds is not None:
                vf, _ = score("val", thetas)
                vb = int(vf.argmax())
                rec["val_best_fit"] = round(float(vf[vb]), 4)
                rec["val_champ_fit"] = round(float(vf[champ]), 4)
                if float(vf[vb]) > best_val:
                    best_val = float(vf[vb])
                    best_val_theta = thetas[vb].cpu().clone()
                    np.savez(os.path.join(args.out_dir, "val_best_genome.npz"),
                             thetas=best_val_theta.numpy(), hidden=hidden,
                             **meta, val_fitness=best_val, gen=gen)
                print(f"      VALIDATION best={rec['val_best_fit']:.4f} "
                      f"(train champ {rec['val_champ_fit']:.4f}; best ever "
                      f"{best_val:.4f})", flush=True)
            if ho_worlds is not None:
                bf, bm = score("hob", thetas)
                bi = per_ind(bm, G)
                brc = (bm["rooms"].view(P, G) >= 2).float().mean(1) \
                    .cpu().numpy()
                bdx = bm["door_crossings"].view(P, G).mean(1).cpu().numpy()
                rec.update({
                    "level": level,
                    "hob_champ_fit": round(float(bf[champ]), 4),
                    "hob_champ_rooms": round(float(bi["rooms"][champ]), 2),
                    "hob_champ_cross_rate": round(float(brc[champ]), 3),
                    "hob_champ_cells": round(float(bi["cells"][champ]), 3),
                    "hob_champ_coll": round(float(bi["coll"][champ]), 1),
                    "hob_elite_fit": round(float(bf[el].mean()), 4),
                    "hob_elite_rooms": round(float(bi["rooms"][el].mean()), 3),
                    "hob_elite_cross_rate": round(float(brc[el].mean()), 3),
                    "hob_elite_door_crossings": round(float(bdx[el].mean()), 2),
                    "hob_elite_coll": round(float(bi["coll"][el].mean()), 2)})
                print(f"      BUILDING HOLDOUT (L3 mix) champ fit="
                      f"{rec['hob_champ_fit']:.4f} rooms="
                      f"{rec['hob_champ_rooms']:.2f} cross="
                      f"{rec['hob_champ_cross_rate']:.2f} coll="
                      f"{rec['hob_champ_coll']:.1f} | elite cross="
                      f"{rec['hob_elite_cross_rate']:.3f} door_x="
                      f"{rec['hob_elite_door_crossings']:.2f}", flush=True)
            print(f"gen {gen:>4d}  best={rec['fit_best']:.4f} "
                  f"med={rec['fit_med']:.4f}  "
                  f"HOLDOUT champ={rec['ho_champ_fit']:.4f} "
                  f"rooms={rec['ho_champ_rooms']:.2f} "
                  f"cells={rec['ho_champ_cells']:.3f} "
                  f"dist={rec['ho_champ_dist']:.1f}m "
                  f"coll={rec['ho_champ_coll']:.1f} | elite "
                  f"fit={rec['ho_elite_fit']:.4f} "
                  f"cells={rec['ho_elite_cells']:.4f}  "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        # ---------------- reproduction -----------------------------------
        if oes is not None:
            rel = oes.tell(fit_np)
            gen_f.write(json.dumps({"gen": gen, "oes_rel_step":
                                    round(rel, 6)}) + "\n")
            thetas = oes.ask(best_theta.to(dev) if gen > 0 else None)
            continue
        thetas, sigma = reproduce(
            thetas, sigma, fit_np, order, scale, rng,
            n_elite=n_elite, n_immigrant=n_immigrant, tau=tau,
            sigma_max=sigma_max, sigma0=args.sigma0, hidden=hidden,
            legacy=args.legacy_mutation)

    if oes is not None:
        np.savez(os.path.join(args.out_dir, "oes_center.npz"),
                 thetas=oes.mu.cpu().numpy(), hidden=hidden, **meta)
    np.savez(os.path.join(args.out_dir, "final_population.npz"),
             thetas=thetas.cpu().numpy(), sigma=sigma.cpu().numpy(),
             hidden=hidden, **meta)
    log_f.close()
    gen_f.close()
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
    ap.add_argument("--sigma0", type=float, default=0.03,
                    help="initial mutation size in init-sigma units "
                    "(runs 1-8: 0.25 — past the behaviour cliff ~0.1)")
    ap.add_argument("--sigma-max", type=float, default=0.15,
                    help="upper clamp on self-adapted sigma (legacy: 1.0)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="log-normal self-adaptation rate; 0 = 1/sqrt(N)")
    ap.add_argument("--legacy-mutation", action="store_true",
                    help="runs 1-8 step-size rule: sigma' from max of two "
                    "parents, tau 0.4, clamp 1.0 (pass --sigma0 0.25). The "
                    "parent-gather bug fix applies either way")
    ap.add_argument("--keep-seed-sigma", action="store_true",
                    help="--seed-from: carry the file's sigma (clamped to "
                    "--sigma-max) instead of resetting to --sigma0")
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
    ap.add_argument("--w-cov", type=float, default=0.0,
                    help="cell-coverage novelty weight (run 5): pays for "
                    "distinct 0.8 m cells visited, saturating. 0 = off "
                    "(identical to runs 1-4). ~0.3 = coverage buys one "
                    "room crossing")
    ap.add_argument("--no-cells", action="store_true",
                    help="do not even allocate/accumulate the coverage "
                    "buffer (fault bisect: byte-proven runs 1-4 kernels)")
    ap.add_argument("--every-cover", type=int, default=24,
                    help="coverage sample stride in ticks (0.32 m at "
                    "v_max < half a cell -> no cell skippable; smaller "
                    "graphs: the run-5 Xid-43 fault class)")
    ap.add_argument("--fp16", action="store_true",
                    help="fp16 raycast (GB10/Spark fast path; ~2x on the "
                    "bandwidth-bound scan, sensing-only)")
    ap.add_argument("--fused", action="store_true",
                    help="fused Triton raycast kernel (implies --fp16): "
                    "single kernel, zero [B,360,M] intermediates — "
                    "~20x faster scan on GB10; parity-gated (99.8% beams "
                    "<0.05m), deterministic, graph-capture-safe (static "
                    "out buffer)")
    ap.add_argument("--world", choices=("legacy", "buildings"),
                    default="legacy",
                    help="buildings: pnn_sim.buildings pools (rooms/door "
                    "thresholds, resampled train houses, L3 building "
                    "holdout). legacy: runs 1-9 make_house arena")
    ap.add_argument("--level", type=int, default=3,
                    help="building curriculum level to start at (0 easy: "
                    "2-3 rooms, wide doors, start facing a door .. 3 full "
                    "mix incl. halls + mazes)")
    ap.add_argument("--curriculum", action="store_true",
                    help="auto-advance levels when the elite's cross rate "
                    "(games with >= 2 rooms) >= --advance-at for "
                    "--advance-patience consecutive gens")
    ap.add_argument("--advance-at", type=float, default=0.5)
    ap.add_argument("--advance-patience", type=int, default=2)
    ap.add_argument("--pool", type=int, default=1024,
                    help="buildings per level pool")
    ap.add_argument("--pool-workers", type=int, default=0,
                    help="pool generation processes (0 = cpu_count - 1)")
    ap.add_argument("--resample-every", type=int, default=1,
                    help="draw a fresh train house set from the pool every "
                    "N gens (elites are re-scored on the same new set)")
    ap.add_argument("--algo", choices=("ga", "oes"), default="ga",
                    help="ga: truncation GA (runs 1-9); oes: OpenAI-ES with "
                    "mirrored sampling + rank shaping + Adam (class OES)")
    ap.add_argument("--oes-sigma", type=float, default=0.03,
                    help="OES perturbation std in init-scale units")
    ap.add_argument("--oes-lr", type=float, default=0.01,
                    help="OES Adam step (init-scale units)")
    ap.add_argument("--oes-l2", type=float, default=0.005)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the per-tick [B,360] chains (noise/"
                    "dropout, preprocess, gate, physics) — see "
                    "Arena._compile_tick; validate with a 3-gen smoke")
    ap.add_argument("--obs", choices=("v1", "v2"), default="v1",
                    help="v2: gate-flag channels replace the two pure-noise "
                    "proprio channels, inputs centred to [-1, 1]")
    ap.add_argument("--action", choices=("lr", "vw"), default="lr",
                    help="vw: net outputs (forward, turn) mixed to tracks")
    ap.add_argument("--graph", action="store_true",
                    help="capture the full rollout as a CUDA graph (one "
                    "replay per generation; removes kernel-launch overhead)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="evo_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-from", default="",
                    help="final_population.npz / best_genome.npz from an"
                         " earlier run: warm-start the population (+sigma)")
    ap.add_argument("--bc-from", default="",
                    help="bc_thetas.npz from evo.bc_seed: distilled "
                         "scripted-explorer genomes occupy the first pop "
                         "slots (run-8 behavior-cloned seeding)")
    ap.add_argument("--door-w", type=float, default=0.0,
                    help="override doorway width (m) for TRAIN houses;"
                         " 0 = natural (0.7-1.0). Holdout is always natural.")
    ap.add_argument("--anneal-gen", type=int, default=0,
                    help="DEPRECATED (torch 2.11 allocator assert on graph "
                         "recapture): prefer chaining two runs via "
                         "--seed-from. If used, run WITHOUT --graph.")
    ap.add_argument("--anneal-door-w", type=float, default=0.94,
                    help="door width (m) after --anneal-gen (real-ish 0.94)")
    args = ap.parse_args()
    if args.fused:
        args.fp16 = True       # fused kernel lives on the fp16 env class
    evolve(args)


if __name__ == "__main__":
    main()
