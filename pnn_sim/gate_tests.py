#!/usr/bin/env python3
"""Curiosity-gate acceptance suite — the pre-hardware green/red bar.

Four objective assertions the curiosity signal must satisfy on the batched
sim components (BatchedEnv, BatchedPlaceMemory, BatchedTrainer) before any
on-rover field session:

  T1 static_decay     In a STATIC world sitting still, place novelty must
                      trend to ~0. If it never decays, gating is broken and
                      the rover wanders forever for no reason.
  T2 revisit_discrim  After building familiarity in a house, teleporting back
                      to a VISITED place must read low novelty; spawning in a
                      NEVER-SEEN house must read high. A signal that merely
                      decays is useless — it must discriminate known/unknown.
  T3 noise_indifference  Inject unlearnable sensor ghosts (random beams
                      flipped to random ranges every tick, like real lidar
                      speckle / dangling cables). Mean novelty must not
                      inflate vs the clean baseline, and the place memory
                      must not churn slots on the flicker. This is the
                      boring-trap test: what hypnotizes the rover kills the
                      pet.
  T4 no_collapse      A short full-trainer run (actor + world model + slow
                      layer) must not park an agent in a corner: median
                      displacement and per-env place counts stay alive.
                      (Dark-room collapse variant — see commit f3a8d8c.)

The novelty measurement in T1–T3 mirrors BatchedTrainer.tick()'s novelty
block verbatim (batched_preprocess -> place.update -> nov_ema EMA) and drives
the rover with a structured random policy through the real BatchedGate, so
the shipping components — not a reimplementation — are what pass or fail.

Usage:
    python3 -m pnn_sim.gate_tests [--device cpu] [--quick] [--json out.json]

Exit code 0 = all gates green, 1 = at least one red.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pnn_sim.batched.env import BatchedEnv, BatchedGate, batched_preprocess
from pnn_sim.batched.place import BatchedPlaceMemory
from pnn_sim.rover import RoverConfig
from pnn_sim.safety_gate import GateConfig

TICK = 1.0 / 15.0          # control rate the trainer uses
NOV_TAU = 1.0              # nov_ema tau, matches BatchedTrainConfig
ACTION_SCALE = 0.6         # matches BatchedTrainConfig.action_scale


class NoveltyPath:
    """Drives B envs with a structured random policy and measures the
    EXACT novelty pipeline BatchedTrainer.tick() uses. One class, three
    tests' worth of scenarios — the only variable is what we do to the
    envs and what we inject."""

    def __init__(self, batch: int, device: str, seed: int = 0,
                 rover_cfg: RoverConfig | None = None):
        self.B = batch
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.env = BatchedEnv(batch, rover_cfg or RoverConfig(seed=seed),
                              seed=seed, device=device)
        self.gate = BatchedGate(GateConfig(), batch,
                                time_fn=lambda: self._t, device=device)
        self.place = BatchedPlaceMemory(batch, device=device)
        self.nov_ema = torch.ones(batch, device=self.device)
        self._t = 0.0
        self._held = np.zeros((batch, 2), dtype=np.float32)
        self._persist = np.zeros(batch, dtype=int)

    def _random_cmd(self) -> np.ndarray:
        f = self.rng.uniform(0.4, 1.0, self.B)
        d = self.rng.uniform(-0.6, 0.6, self.B)
        fwd = np.stack([f + d, f - d], axis=1)
        spin = self.rng.uniform(-1, 1, self.B)
        spin = np.stack([spin, -spin], axis=1)
        return np.where(self.gate.front_blocked.cpu().numpy()[:, None],
                        spin, fwd)

    def step(self, flicker_beams: int = 0):
        ranges = self.env.scan()
        self.gate.process_scan(ranges, self.env.angle_min,
                               self.env.angle_increment)
        if flicker_beams > 0:
            # Unlearnable sensor ghosts: flip random beams to random ranges
            # (lidar speckle / swaying cable) BEFORE preprocessing, exactly
            # as noise arrives before binning on the real rover.
            ranges = ranges.clone()
            idx = torch.randint(0, ranges.shape[1],
                                (self.B, flicker_beams),
                                device=self.device)
            vals = torch.rand(self.B, flicker_beams, device=self.device) \
                * 3.0 + 0.1
            ranges.scatter_(1, idx, vals)

        scan72 = batched_preprocess(ranges, self.env.angle_min,
                                    self.env.angle_increment)
        # --- mirror of BatchedTrainer.tick() novelty block ---
        nov_raw = self.place.update(scan72, TICK)
        alpha = min(1.0, TICK / max(NOV_TAU, TICK))
        self.nov_ema += alpha * (nov_raw - self.nov_ema)
        # ------------------------------------------------------

        self._persist -= 1
        redec = self._persist <= 0
        self._held = np.where(redec[:, None], self._random_cmd(), self._held)
        self._persist = np.where(redec, 5 - 1, self._persist - 1)
        out = np.clip(self._held * ACTION_SCALE, -1.0, 1.0)
        gated = self.gate.gate(
            torch.from_numpy(out.astype(np.float32)).to(self.device))
        self.env.step(gated, TICK)
        self._t += TICK

    def teleport(self, env_idx: int, pose: tuple[float, float, float]):
        i = torch.tensor([env_idx], device=self.device, dtype=torch.long)
        self.env.x[i], self.env.y[i], self.env.theta[i] = (
            torch.tensor(pose[0], device=self.device),
            torch.tensor(pose[1], device=self.device),
            torch.tensor(pose[2], device=self.device))
        self.env.v_left[i] = 0.0
        self.env.v_right[i] = 0.0

    def idle_steps(self, n: int):
        """Sit perfectly still for n ticks (still sensing)."""
        zero = np.zeros((self.B, 2), dtype=np.float32)
        for _ in range(n):
            ranges = self.env.scan()
            self.gate.process_scan(ranges, self.env.angle_min,
                                   self.env.angle_increment)
            scan72 = batched_preprocess(ranges, self.env.angle_min,
                                        self.env.angle_increment)
            nov_raw = self.place.update(scan72, TICK)
            alpha = min(1.0, TICK / max(NOV_TAU, TICK))
            self.nov_ema += alpha * (nov_raw - self.nov_ema)
            self.env.step(torch.from_numpy(zero).to(self.device), TICK)
            self._t += TICK

    def mean_nov(self, tail: int = 0) -> float:
        return float(self.nov_ema.mean())


# ---------------------------------------------------------------- tests

def t1_static_decay(device: str, quick: bool) -> dict:
    """Idle in one static house: novelty must decay to ~0.

    NOTE: place memory grants every new place fam_scale_s=20 s of sustained
    novelty credit by design (the pet is allowed to stay fascinated for a
    while), so the idle budget must exceed 20 s of continuous presence or we
    are only measuring that mechanism. Budgets below guarantee >= 25 s.
    """
    B = 8
    ticks = 450 if not quick else 300          # total idle >= 26.7 s > 20 s
    np_ = NoveltyPath(B, device, seed=11)      # real (noisy) sensor profile
    np_.idle_steps(ticks // 3)                  # settle: EMA + first place
    early = np_.mean_nov()
    np_.idle_steps(ticks)
    late = np_.mean_nov()
    ok = late < 0.05 and early > late
    return {"test": "T1_static_decay", "pass": ok,
            "novelty_early": round(early, 4), "novelty_late": round(late, 4),
            "criterion": "novelty_late < 0.05 after > fam_scale_s idle"}


def t2_revisit_discrimination(device: str, quick: bool) -> dict:
    """Familiar place reads LOW; never-seen house reads HIGH."""
    B = 8
    warm = 900 if not quick else 300            # 60s driving to build places
    np_ = NoveltyPath(B, device, seed=22)
    # build familiarity in home A (env 0) — drive a while
    for _ in range(warm):
        np_.step()
    spawn = (float(np_.env.x[0]), float(np_.env.y[0]),
             float(np_.env.theta[0]))
    # revisit: teleport env 0 back to its spawn (a visited place),
    # idle so only that fingerprint is fed
    np_.teleport(0, spawn)
    np_.idle_steps(45)
    # measure on env 0 only after EMA settles (~1s)
    known = float(np_.nov_ema[0])
    # novel: give env 0 a brand-new house, spawn pose, fresh place memory
    idx = torch.tensor([0], device=np_.device, dtype=torch.long)
    mask = np.zeros(B, dtype=bool)
    mask[0] = True
    np_.env.switch_world(mask)
    pose = np_.env._worlds[0].start_pose
    np_.place.clear(idx)
    np_.nov_ema[idx] = 1.0
    np_.teleport(0, (float(pose[0]), float(pose[1]), float(pose[2])))
    np_.idle_steps(30)
    fresh = float(np_.nov_ema[0])
    ok = known < 0.30 and fresh > 0.80
    return {"test": "T2_revisit_discrimination", "pass": ok,
            "novelty_known_place": round(known, 4),
            "novelty_never_seen_house": round(fresh, 4),
            "criterion": "known < 0.30 and novel > 0.80"}


def t3_noise_indifference(device: str, quick: bool) -> dict:
    """Unlearnable flicker must not inflate novelty or churn place slots."""
    B = 8
    ticks = 600 if not quick else 250
    flicker = 12                                 # beams / 360 flipped per tick
    base = NoveltyPath(B, device, seed=33)
    flick = NoveltyPath(B, device, seed=33)      # same worlds + same policy
    for _ in range(ticks):
        base.step(flicker_beams=0)
        flick.step(flicker_beams=flicker)
    nov_base, nov_flick = base.mean_nov(), flick.mean_nov()
    slots_base = float(base.place.n_places().float().mean())
    slots_flick = float(flick.place.n_places().float().mean())
    ok = (nov_flick <= nov_base + 0.15
          and slots_flick <= max(slots_base * 2.0, slots_base + 2))
    return {"test": "T3_noise_indifference", "pass": ok,
            "novelty_clean": round(nov_base, 4),
            "novelty_flicker": round(nov_flick, 4),
            "slots_clean": round(slots_base, 2),
            "slots_flicker": round(slots_flick, 2),
            "criterion": "flicker nov <= clean+0.15; slot churn <= 2x"}


def t4_no_collapse(device: str, quick: bool) -> dict:
    """Full trainer stack for a short run: no parked-in-corner agents."""
    from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig
    B = 8 if quick else 16
    ticks = 1500 if quick else 4500            # 100 s / 300 s sim per env —
    # enough time to cross doorways and form >= 3 rotation-invariant places
    out_dir = "/tmp/pnn_gate_tests_T4"
    cfg = BatchedTrainConfig(envs=B, device=device, seed=44,
                             out_dir=out_dir, switch_world_every=0,
                             snapshot_every=0, save_interval_s=1e9,
                             log_envs=0,
                             # Validated temperament (t4_place_probe/rules):
                             # frozen place refs (slot_blend 0 — 0.02 chase
                             # made dmin equilibrate ~0.09 << thresh so a
                             # continuous walk matched one place forever),
                             # calibrated fingerprint match (thresh 0.20 +
                             # shape 2.0 resolves 27-31 places/house, fp_grid),
                             # frontier steering from the PC map (3x disp).
                             place_slot_blend=0.0,
                             place_match_thresh=0.20,
                             place_shape_weight=2.0,
                             frontier_weight=1.0)
    tr = BatchedTrainer(cfg)
    spawn = torch.stack([tr.env.x.clone(), tr.env.y.clone()], dim=1)
    t0 = time.time()
    for _ in range(ticks):
        tr.tick()
    wall = time.time() - t0
    disp = torch.linalg.norm(
        torch.stack([tr.env.x, tr.env.y], dim=1) - spawn, dim=1)
    places = tr.place.n_places().float()
    nov = float(tr.nov_ema.mean())
    med_disp = float(disp.median())
    med_pl = float(places.median())
    tr.close()
    ok = med_disp > 0.30 and med_pl >= 3.0 and 0.05 < nov < 1.0
    return {"test": "T4_no_collapse", "pass": ok,
            "median_displacement_m": round(med_disp, 3),
            "median_places_per_env": round(med_pl, 1),
            "novelty": round(nov, 4),
            "wall_s": round(wall, 1),
            "criterion": "displacement > 0.3 m; places >= 3; novelty alive"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true",
                    help="short tick budgets (smoke level)")
    ap.add_argument("--json", default="", help="write results json here")
    ap.add_argument("--only", default="",
                    help="comma list: t1,t2,t3,t4 (default all)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable; falling back to cpu")
        args.device = "cpu"

    tests = {"t1": t1_static_decay, "t2": t2_revisit_discrimination,
             "t3": t3_noise_indifference, "t4": t4_no_collapse}
    run = [k for k in tests if not args.only or k in args.only.split(",")]

    results, all_ok = [], True
    for k in run:
        name = tests[k].__name__
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            r = tests[k](args.device, args.quick)
        except Exception as e:              # noqa: BLE001 — suite must not die
            import traceback
            traceback.print_exc()
            r = {"test": name, "pass": False, "error": f"{type(e).__name__}: {e}"}
        r["wall_s"] = r.get("wall_s", round(time.time() - t0, 1))
        all_ok &= bool(r["pass"])
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)

    print(f"\n{'=' * 46}\nGATE: {'GREEN — cleared for on-rover field work' if all_ok else 'RED — curiosity gating not ready for hardware'}\n{'=' * 46}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"all_pass": all_ok, "results": results}, f, indent=2)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
