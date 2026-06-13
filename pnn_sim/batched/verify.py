#!/usr/bin/env python3
"""Equivalence tests: batched implementation vs the reference modules.

Each test copies weights from a reference object into the batched one,
feeds identical inputs, and asserts the outputs match to float tolerance.
Stochastic pieces (bootstrap, candidate sampling) are pinned for the
comparison. Run:

    python3 -m pnn_sim.batched.verify
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from tractor_bringup.active_inference.pc_world_model import PCWorldModel, PCConfig
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig
from tractor_bringup.active_inference.scan_preprocess import preprocess_scan

from pnn_sim.batched.model import BatchedPCWorldModel, BatchedEFEActor
from pnn_sim.batched.env import batched_preprocess
from pnn_sim.rover import SimRover, RoverConfig
from pnn_sim.world import make_house
from pnn_sim.batched.env import BatchedEnv, BatchedGate
from pnn_sim.safety_gate import SimSafetyGate, GateConfig

OBS, D, B = 82, 64, 5
CFG = dict(obs_dim=OBS, latent_dim=D, ensemble_size=5, n_infer_iters=24,
           n_proprio=8, precision_proprio=4.0, n_intero=2,
           precision_intero=6.0, seed=3)


def _pair():
    ref = PCWorldModel(PCConfig(**CFG))
    bat = BatchedPCWorldModel(PCConfig(**CFG), batch=B)
    bat.load_state_dict({k: v for k, v in ref.state_dict().items()})
    return ref, bat


def test_init_identical():
    ref = PCWorldModel(PCConfig(**CFG))
    bat = BatchedPCWorldModel(PCConfig(**CFG), batch=B)
    assert torch.equal(ref.W_o, bat.W_o), "W_o init differs"
    for m in range(5):
        assert torch.equal(ref.W_z[m], bat.W_z[m]), f"W_z[{m}] init differs"
    print("ok  init draws identical weights")


def test_infer():
    ref, bat = _pair()
    g = torch.Generator().manual_seed(7)
    obs = torch.rand(B, OBS, generator=g)
    act = torch.rand(B, 2, generator=g) * 2 - 1
    zp = torch.randn(B, D, generator=g) * 0.3
    td = torch.rand(B, D, generator=g) * 2 - 1

    bz, bF, berr = bat.infer(obs, act, z_prev=zp, td_target=td,
                             td_precision=0.2)
    for b in range(B):
        rz, rF, rerr = ref.infer(obs[b], act[b], z_prev=zp[b],
                                 td_target=td[b], td_precision=0.2)
        assert torch.allclose(rz, bz[b], atol=1e-5), f"z mismatch env {b}"
        assert abs(rF - float(bF[b])) < 1e-3, f"F mismatch env {b}"
        assert abs(rerr - float(berr[b])) < 1e-4, f"err mismatch env {b}"
    print("ok  infer matches per-row")


def test_learn_b1():
    """With B=1 and bootstrap pinned to keep-all, one batched learn must
    equal one reference learn exactly (mean over batch of 1 = the sample)."""
    ref = PCWorldModel(PCConfig(**{**CFG, "bootstrap_prob": 1.0}))
    bat = BatchedPCWorldModel(PCConfig(**{**CFG, "bootstrap_prob": 1.0}),
                              batch=1)
    bat.load_state_dict({k: v for k, v in ref.state_dict().items()})

    g = torch.Generator().manual_seed(11)
    obs = torch.rand(OBS, generator=g)
    act = torch.rand(2, generator=g) * 2 - 1
    zp = torch.randn(D, generator=g) * 0.3

    # Identical settled z into both learns — this isolates the update rule
    # (infer equivalence is test_infer's job; its 1e-6 float noise would be
    # amplified by the precision ratio here).
    rz, _, _ = ref.infer(obs, act, z_prev=zp)
    ref.learn(rz, act, obs, z_prev=zp)
    bat.learn(rz.unsqueeze(0), act.unsqueeze(0), obs.unsqueeze(0),
              z_prev=zp.unsqueeze(0))

    assert torch.allclose(ref.W_o, bat.W_o, atol=1e-6), "W_o update differs"
    assert torch.allclose(ref.b_o, bat.b_o, atol=1e-6), "b_o update differs"
    for m in range(5):
        assert torch.allclose(ref.W_z[m], bat.W_z[m], atol=1e-6), \
            f"W_z[{m}] update differs"
    assert torch.allclose(ref.pi_o, bat.pi_o, atol=1e-5), "precision differs"
    print("ok  learn (B=1, full bootstrap) matches exactly")


def test_actor():
    """Structured candidates only (n_random=0), deterministic argmax —
    batched scores must pick the reference's action per env."""
    ref_m, bat_m = _pair()
    acfg = dict(action_dim=2, n_random=0, deterministic=True,
                pragmatic_weight=0.3, num_bins=72, use_proprio=True,
                n_intero=2, target_novelty=0.8, novelty_pref_weight=2.0,
                hold_pref_weight=2.0, horizon=8, seed=5)
    ref_a = EFEActor(ActorConfig(**acfg))
    bat_a = BatchedEFEActor(ActorConfig(**acfg), batch=B)

    g = torch.Generator().manual_seed(13)
    z = torch.randn(B, D, generator=g) * 0.5
    prev = torch.rand(B, 2, generator=g) * 2 - 1
    slow = torch.rand(B, 2, generator=g) * 2 - 1
    blocked = torch.tensor([True, False, True, False, False])

    ba, _ = bat_a.select(bat_m, z, prev_action=prev, slow_action=slow,
                         slow_warm=torch.ones(B, dtype=torch.bool),
                         forward_blocked=blocked)
    for b in range(B):
        ra, _ = ref_a.select(ref_m, z[b], prev_action=prev[b].numpy(),
                             slow_action=slow[b].numpy(),
                             forward_blocked=bool(blocked[b]))
        assert torch.allclose(ra, ba[b], atol=1e-5), \
            f"actor choice differs env {b}: {ra} vs {ba[b]}"
    print("ok  actor (structured, argmax) matches per-env")


def test_preprocess():
    rng = np.random.default_rng(17)
    ranges = rng.uniform(0.03, 12.0, size=(B, 360)).astype(np.float32)
    ranges[rng.random((B, 360)) < 0.05] = np.inf
    out = batched_preprocess(torch.from_numpy(ranges), 0.0,
                             2 * np.pi / 360).numpy()
    for b in range(B):
        ref = preprocess_scan(ranges[b], 0.0, 2 * np.pi / 360)
        assert np.allclose(ref, out[b], atol=1e-6), f"preprocess differs env {b}"
    print("ok  batched scan preprocessing matches")


def test_rover_dynamics():
    """Noise-free configs: batched env row 0 must track the scalar rover."""
    quiet = dict(slip_std=0.0, lidar_noise_std=0.0, lidar_dropout_p=0.0,
                 gyro_noise_std=0.0, accel_noise_std=0.0)
    rng = np.random.default_rng(19)
    world = make_house(np.random.default_rng(23))

    ref = SimRover(world, RoverConfig(**quiet))
    bat = BatchedEnv(3, RoverConfig(**quiet), seed=23)
    # Pin env 0 to the reference's world and pose.
    bat._worlds[0] = world
    bat._build_segments()
    bat.x[0], bat.y[0], bat.theta[0] = world.start_pose
    bat.v_left[0] = bat.v_right[0] = bat._prev_v[0] = 0.0

    # The torch env is float32 (GPU-friendly; fp64 is crippled on most
    # GPUs), the scalar sim float64 — tolerances reflect that.
    for _ in range(50):
        cmd = rng.uniform(-1, 1, size=2)
        ref.step(cmd[0], cmd[1], 1 / 15)
        cmds = torch.from_numpy(np.tile(cmd, (3, 1)).astype(np.float32))
        bat.step(cmds, 1 / 15)
        assert abs(ref.x - float(bat.x[0])) < 1e-4 \
            and abs(ref.y - float(bat.y[0])) < 1e-4, "pose diverged"
        assert abs(ref.yaw_rate - float(bat.yaw_rate[0])) < 1e-4, \
            "gyro diverged"
    r_ref, _, _ = ref.scan()
    r_bat = bat.scan().numpy()
    assert np.allclose(r_ref, r_bat[0], atol=1e-2), "lidar diverged"
    print("ok  rover dynamics + lidar match the scalar sim")


def test_gate():
    cfg = GateConfig()
    t = {"v": 0.0}
    ref = SimSafetyGate(cfg, time_fn=lambda: t["v"])
    bat = BatchedGate(cfg, 2, time_fn=lambda: t["v"])
    rng = np.random.default_rng(29)
    n = 360
    inc = 2 * np.pi / n
    for i in range(300):
        # Random-ish scene with an obstacle sweeping through the front.
        ranges = rng.uniform(0.5, 4.0, size=n).astype(np.float32)
        if (i // 30) % 2 == 0:
            front = slice(0, 8)
            ranges[front] = rng.uniform(0.08, 0.18, size=8)
            ranges[-8:] = rng.uniform(0.08, 0.18, size=8)
        ref.process_scan(ranges, 0.0, inc)
        bat.process_scan(torch.from_numpy(np.tile(ranges, (2, 1))), 0.0, inc)
        assert ref.front_blocked == bool(bat.front_blocked[0]), \
            f"front_blocked diverged at scan {i}"
        cmd = rng.uniform(-1, 1, size=2)
        gl, gr = ref.gate(cmd[0], cmd[1])
        gb = bat.gate(torch.from_numpy(
            np.tile(cmd, (2, 1)).astype(np.float32))).numpy()
        assert abs(gl - gb[0, 0]) < 1e-6 and abs(gr - gb[0, 1]) < 1e-6, \
            f"gated cmd diverged at scan {i}: ({gl},{gr}) vs {gb[0]}"
        t["v"] += 1 / 15
    print("ok  safety gate state machine matches (300 scans)")


def test_place_memory():
    """Batched slot-array place memory vs the reference list version,
    driven by the same scan stream at the sim's fixed dt."""
    from tractor_bringup.active_inference.place_memory import PlaceMemory
    from pnn_sim.batched.place import BatchedPlaceMemory

    dt = 1.0 / 15.0
    t = {"v": 0.0}
    refs = [PlaceMemory(time_fn=lambda: t["v"]) for _ in range(3)]
    bat = BatchedPlaceMemory(3, max_places=64)

    rng = np.random.default_rng(31)
    # Scan stream that revisits a few "rooms" so matching/reinforce/new
    # paths all fire: blend between room prototypes with noise.
    rooms = rng.uniform(0.2, 1.0, size=(5, 72)).astype(np.float32)
    for i in range(400):
        scans = np.stack([rooms[(i // 40 + b) % 5]
                          + rng.normal(0, 0.01, 72).astype(np.float32)
                          for b in range(3)])
        t["v"] += dt
        nov_ref = np.array([refs[b].update(scans[b]) for b in range(3)])
        nov_bat = bat.update(torch.from_numpy(scans), dt).numpy()
        assert np.allclose(nov_ref, nov_bat, atol=1e-5), \
            f"novelty diverged at step {i}: {nov_ref} vs {nov_bat}"
    for b in range(3):
        assert refs[b].n_places() == int(bat.n_places()[b]), \
            f"place count diverged env {b}"
    print("ok  place memory matches (400 steps, revisits + new rooms)")


def main():
    torch.manual_seed(0)
    test_init_identical()
    test_infer()
    test_learn_b1()
    test_actor()
    test_preprocess()
    test_rover_dynamics()
    test_gate()
    test_place_memory()
    print("\nall equivalence tests passed")


if __name__ == "__main__":
    main()
