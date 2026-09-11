"""Fusion-arithmetic checks for PlaceMemory.update(scan, vis_fp).

Temperament note: frozen refs (blend 0) + create gate are the FIELD
temperament tuned for a walking rover; for these unit scenarios (stationary
dwell clusters) we use the blend-0.02 regime the chase analysis validated:
blend >> drift/thresh so refs track the cluster center and the drift gate is
inert. What we assert here is ONLY the fusion arithmetic: w=0 == lidar-only,
w>0 can split/merge by vision, missing vis is inert.

Run from repo root: python3 pnn_sim/tools/fusion_check.py
"""
import sys
import numpy as np
sys.path.insert(0, '/home/benson/projects/ros2-rover/src/tractor_bringup')
from tractor_bringup.active_inference.place_memory import PlaceMemory


def scan(seed, level):
    r = np.random.default_rng(seed)
    return np.clip(level + 0.02 * r.standard_normal(72), 0, 1).astype(np.float32)


def one_vis(bit):
    v = np.zeros(23, np.float32); v[bit] = 1.0
    return v / np.linalg.norm(v)


seqA = [scan(1, 0.5) for _ in range(40)]
seqB = [scan(2, 0.25) for _ in range(40)]          # lidar-different room
seqB_same = [s.copy() for s in seqA]               # lidar-IDENTICAL room
visA, visB = one_vis(0), one_vis(8)


def replay(w, seq, vfs, blend=0.02, thresh=0.15):
    t = [0.0]
    pm = PlaceMemory(match_thresh=thresh, shape_weight=2.0, fp_ema_tau_s=1.0,
                     slot_blend=blend, create_drift_gate=0.0, vis_weight=w,
                     time_fn=lambda: t[0])
    for i, (s, v) in enumerate(zip(seq, vfs)):
        t[0] = i * 0.1
        pm.update(s, vis_fp=v)
    return pm.n_places()


vfs_none = [None] * 80
vfs_vis = [visA] * 40 + [visB] * 40

# 1) w=0 ignores vis entirely: equals lidar-only legacy behavior, and two
#    genuinely different lidar rooms DO separate in this regime
a = replay(0.0, seqA + seqB, vfs_none)
b = replay(0.0, seqA + seqB, vfs_vis)
print("1. lidar-only:", a, "| w=0 with vis fed:", b,
      "| equal:", a == b, "(want equal, ==2: two real rooms)")
assert a == b == 2

# 2) lidar-identical rooms: merged at w=0, SPLIT by vision at w=1
p0 = replay(0.0, seqA + seqB_same, vfs_vis)
p1 = replay(1.0, seqA + seqB_same, vfs_vis)
print("2. lidar-twins places @w=0:", p0, " @w=1:", p1, "(want 1 then 2)")
assert (p0, p1) == (1, 2), f"fusion not doing its job: {p0} {p1}"

# 3) vis steers only when BOTH query and slot have a visual ref:
#    frames present in room A only, w=1 -> the twin does NOT split (no crash)
vfs_half = [visA] * 40 + [None] * 40
p2 = replay(1.0, seqA + seqB_same, vfs_half)
print("3. vis refs only in room A @w=1:", p2, "(want 1: vision absent => lidar rule)")
assert p2 == 1

# 4) stationary + jittered vis below thresh must NOT create places
jit = [visA + 0.01 * np.random.default_rng(i).standard_normal(23).astype(np.float32)
       for i in range(40)]
t = [0.0]
pm = PlaceMemory(match_thresh=0.15, shape_weight=2.0, fp_ema_tau_s=1.0,
                 slot_blend=0.02, create_drift_gate=0.0, vis_weight=1.0,
                 time_fn=lambda: t[0])
for i, s in enumerate(seqA):
    t[0] = i * 0.1
    pm.update(s, vis_fp=jit[i] / np.linalg.norm(jit[i]))
print("4. stationary + jittered vis @w=1 places:", pm.n_places(), "(want 1)")
assert pm.n_places() == 1

# 5) the other direction: lidar twin rooms the vision MERGES (vis identical)
#    must stay merged even while lidar would split them at w=1... lidar here
#    is identical too, so trivially 1; real merge test = lidar B different
#    but vision identical, w large enough to matter only if it flipped
#    a match: assert it does NOT split beyond lidar-only.
p3 = replay(1.0, seqA + seqB, [visA] * 80)          # vision says same room
print("5. lidar-different rooms, vision identical @w=1:", p3, "(want 2: lidar still splits)")
assert p3 == 2

print("ALL FUSION CHECKS PASS")
