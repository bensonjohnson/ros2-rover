#!/usr/bin/env python3
"""Building-mode arena checks (pnn_sim.buildings + Arena(worlds=...)).

    python3 -m evo.test_buildings --device cpu      (or cuda)

  B1  GPU rect room code == Building.room_id at every sampled pose
  B2  door_crossings / doors_used == numpy reference on the sampled poses;
      rooms >= 2 in a game implies >= 1 threshold crossing
  B3  load_worlds(new) replays == a fresh Arena built on `new`
  B4  generator invariants per family/level: segments <= cap, start in a
      room with clearance, all rooms reachable, thresholds join two rooms
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from pnn_sim.buildings import FAMILIES, LEVELS, make_building, make_level
from .arena import Arena, NUM_BINS, world_caps


def cruiser(B, device, seed=0):
    """Wandering scripted driver: cruise, steer to the open side, pivot
    when blocked, with a per-env bias so envs diverge."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bias = (torch.rand(B, generator=g) * 2 - 1).to(device)

    def step(obs, prev):
        s = obs[:, :NUM_BINS]
        front = torch.minimum(s[:, :6].min(1).values, s[:, -6:].min(1).values)
        left, right = s[:, 9:27].mean(1), s[:, 45:63].mean(1)
        steer = 0.6 * (left - right) + 0.25 * bias
        f = 0.3 + 0.7 * front.clamp(0, 1)
        cmd = torch.stack([f - steer, f + steer], 1)
        piv = torch.stack([-0.6 * torch.sign(bias), 0.6 * torch.sign(bias)], 1)
        return torch.where((front < 0.1).unsqueeze(1), piv, cmd).clamp(-1, 1)
    return step


def houses(n, seed, level=None, family="mix"):
    return [make_level(np.random.default_rng(seed + i), level) if level is not None
            else make_building(np.random.default_rng(seed + i), family)
            for i in range(n)]


def record(arena):
    poses_room, poses_door = [], []
    orig_room, orig_door = arena._room_code_rects, arena._doors_step

    def room():
        poses_room.append((arena.env.x.clone(), arena.env.y.clone()))
        return orig_room()

    def door():
        poses_door.append((arena.env.x.clone(), arena.env.y.clone()))
        return orig_door()
    arena._room_code_rects, arena._doors_step = room, door
    return poses_room, poses_door


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ticks", type=int, default=1500)
    args = ap.parse_args()
    dev = args.device
    P, G = 3, 4
    H = houses(G, 500, level=0)
    arena = Arena(P, G, seed=1, device=dev, worlds=H)
    rec_room, rec_door = record(arena)
    torch.manual_seed(0)
    m = arena.run_games(cruiser(arena.B, dev), args.ticks)
    B = arena.B

    # B1: room codes at every sampled pose
    bad = 0
    seen = [set() for _ in range(B)]
    for x, y in rec_room:
        xs, ys = x.cpu().numpy(), y.cpu().numpy()
        for b in range(B):
            r = H[b % G].room_id(float(xs[b]), float(ys[b]))
            if r >= 0:
                seen[b].add(r)
    ref_rooms = np.array([len(s_) for s_ in seen])
    assert np.array_equal(ref_rooms, m["rooms"].cpu().numpy().astype(int)), \
        (ref_rooms, m["rooms"])
    print(f"B1 ok: rooms per env {ref_rooms.tolist()} match room_id replay")

    # B2: door crossings reference
    cross = np.zeros(B)
    used = [set() for _ in range(B)]
    prev = [dict() for _ in range(B)]
    for x, y in rec_door:
        xs, ys = x.cpu().numpy(), y.cpu().numpy()
        for b in range(B):
            d = H[b % G].doors
            d = d[(d[:, 4] >= 0) & (d[:, 5] >= 0)]
            for k, (x1, y1, x2, y2, _, _) in enumerate(d):
                dx, dy = x2 - x1, y2 - y1
                L = max(np.hypot(dx, dy), 1e-6)
                qx, qy = xs[b] - x1, ys[b] - y1
                side = np.sign(dx * qy - dy * qx)
                t = (qx * dx + qy * dy) / (L * L)
                p = prev[b].get(k, 0.0)
                if side * p < 0 and -0.12 / L < t < 1 + 0.12 / L:
                    cross[b] += 1
                    used[b].add(k)
                if side != 0:
                    prev[b][k] = side
    gpu_cross = m["door_crossings"].cpu().numpy()
    assert np.allclose(cross, gpu_cross), (cross, gpu_cross)
    assert np.array_equal([len(u) for u in used],
                          m["doors_used"].cpu().numpy().astype(int))
    multi = ref_rooms >= 2
    assert (gpu_cross[multi] >= 1).all(), "room change without a crossing"
    print(f"B2 ok: crossings {gpu_cross.astype(int).tolist()} "
          f"(envs with >=2 rooms all crossed a threshold)")

    # B3: load_worlds == fresh arena
    H2 = houses(G, 900, level=1)
    caps = tuple(max(a, b) for a, b in zip(world_caps(H), world_caps(H2)))
    a1 = Arena(P, G, seed=1, device=dev, worlds=H, caps=caps)
    a1.load_worlds(H2)
    a2 = Arena(P, G, seed=1, device=dev, worlds=H2, caps=caps)
    out = []
    for a in (a1, a2):
        torch.manual_seed(5)
        out.append(a.run_games(cruiser(a.B, dev, seed=3), 600))
    for k in ("rooms", "rooms_total", "cells", "cells_total", "collisions",
              "door_crossings", "doors_total"):
        assert torch.equal(out[0][k], out[1][k]), k
    assert torch.allclose(out[0]["dist_m"], out[1]["dist_m"], atol=1e-3)
    print("B3 ok: load_worlds replay == fresh arena")

    # B4: generator invariants
    for tag, gen in ([(f, lambda r, f=f: make_building(r, f)) for f in FAMILIES]
                     + [(f"level{l}", lambda r, l=l: make_level(r, l))
                        for l in LEVELS]):
        for s in range(6):
            w = gen(np.random.default_rng(7000 + s))
            x, y, _ = w.start_pose
            assert w.segments.shape[0] <= 128, tag
            assert w.room_id(x, y) >= 0 and w.clearance(x, y) > 0.3, tag
            assert w.reachable(x, y), tag
            if w.family != "house":
                assert len(w.rooms_reachable) == w.n_rooms, tag
            d = w.doors[(w.doors[:, 4] >= 0) & (w.doors[:, 5] >= 0)]
            assert (d[:, 4] != d[:, 5]).all(), tag
    print("B4 ok: generator invariants over families + levels")
    print("ALL BUILDING CHECKS PASSED")


if __name__ == "__main__":
    main()
