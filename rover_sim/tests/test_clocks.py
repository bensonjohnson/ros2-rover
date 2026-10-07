"""Multi-rate clock tests (docs/SIM_PLATFORM.md §8 fidelity item 1).

    python3 -m rover_sim.tests.test_clocks [--device cpu]

The legacy sim ticks one clock: every control tick carries a fresh lidar
revolution, a gate update, a policy tick and a motor step. The rover does the
opposite (LD19 ~10 Hz, brain 15 Hz, gate in the /scan callback, driver
rate-limited), so a policy trained on the single-clock sim learns a causally
wrong map. These tests pin the clock MODEL — the parts that must be right for
training to mean anything:

C1  Inert when off: `clocks=None` and an all-None `ClocksConfig()` both build
    no clock and produce bit-identical rollouts (an inactive config is not a
    silently different sim).
C2  Lidar hold: at control 15 Hz / lidar 10 Hz the obs scan channel changes on
    exactly the env's due ticks — never between them — and the delivered
    fraction is ~10/15, i.e. the policy really is acting on a stale revolution
    a third of the time.
C3  Rate semantics (unit): due-flag schedule, phase offsets, revolution
    counting, and the driver latch (a 5 Hz driver accepts one command in 3).
C4  Phase stagger: off = every env in lockstep; on = the phases are spread
    (deterministic golden-ratio sequence, no RNG), so no single phase
    relationship can become the trained convention.
C5  Engine integration: a rollout under clocks stays finite and sane, matches
    the legacy engine exactly under a zero command (no scan-dependent output),
    and differs once the policy reads the scan (staleness is observable).
C6  (CUDA only) capture: a clocks-enabled engine captures and replays a CUDA
    graph and reports finite metrics. Parent runs this on the Spark.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    # torch built with CUDA support but no usable driver (Hermes SBC): any
    # torch.cuda.* call aborts. On CPU rollouts it is semantically False.
    torch.cuda.is_current_stream_capturing = lambda: False

from ..runner.clocks import ClocksConfig, MultiRateClock
from ..runner.engine import CONTROL_HZ, RolloutEngine

SEED = 777_000
NOISE = 12_345
DT = 1.0 / CONTROL_HZ


def _engine(P=1, G=4, *, device="cpu", **kw):
    return RolloutEngine(P, G, seed=SEED, noise_seed=NOISE, device=device,
                         cells=False, **kw)


def _zero_policy(obs, prev):
    return torch.zeros(obs.shape[0], 2, device=obs.device)


def _scan_reading_policy(obs, prev):
    """Bang-bang scan follower: a stale reading flips the turn decision, so
    staleness reaches the trajectory and the metrics."""
    left = obs[:, :8].mean(dim=1)
    right = obs[:, 64:72].mean(dim=1)
    s = torch.where(left > right, torch.ones_like(left), -torch.ones_like(left))
    return torch.stack([0.6 - 0.6 * s, 0.6 + 0.6 * s], dim=1)


def _m(eng, policy, ticks=200, device="cpu"):
    return eng.run_games(policy, ticks)


def c1_inert_when_off(device):
    a = _engine(device=device)
    b = _engine(device=device, clocks=ClocksConfig())
    assert a._mr is None, "clocks=None built a clock"
    assert b._mr is None, "an all-None ClocksConfig is not inert"
    ma = a.run_games(_scan_reading_policy, 120)
    mb = b.run_games(_scan_reading_policy, 120)
    for k in ("dist_m", "collisions", "rooms", "net_m"):
        assert torch.equal(ma[k], mb[k]), f"{k}: {ma[k]} != {mb[k]}"
    print("C1 ok: clocks=None and ClocksConfig() are both inert and "
          f"bit-identical (dist {float(ma['dist_m'].mean()):.3f} m)")


def c2_lidar_hold(device):
    """The obs scan channel may only change on a due tick, and 10 Hz against
    a 15 Hz brain delivers ~2 of every 3 ticks."""
    B, ticks = 8, 300
    eng = _engine(1, B, device=device, clocks=ClocksConfig(lidar_hz=10.0))
    assert eng._mr is not None and eng._mr.lidar_on
    seen = []

    def probe(obs, prev):
        seen.append(obs[:, :72].clone())
        return torch.zeros(obs.shape[0], 2, device=obs.device)

    eng.run_games(probe, ticks)
    # unresolved: each tick's scan block per env
    blocks = torch.stack(seen)                      # [ticks, B, 72]
    changed = (blocks[1:] != blocks[:-1]).any(dim=2)   # [ticks-1, B]
    # ground truth: replay the SAME schedule from reset on a fresh clock
    # (the engine's own clock has advanced past the rollout).
    mc = MultiRateClock(ClocksConfig(lidar_hz=10.0), DT, B, 360, device)
    due = []
    for _ in range(ticks):
        mc.tick()
        due.append(mc.lidar_due.clone())
        mc.scan_taken()
    due = torch.stack(due)                          # [ticks, B]

    # a delivered revolution at tick t is visible in obs[t]; the clock's due
    # flag is computed at the START of tick t, so obs[t] fresh == due[t]
    assert not bool(changed[~due[1:]].any()), \
        "a scan block changed on a tick that was not due"
    frac = float(changed.float().mean())
    assert abs(frac - 10.0 / CONTROL_HZ) < 0.05, f"delivery fraction {frac}"
    # never fewer than ~10 Hz of revolutions over a long window
    assert float(changed[:200].float().sum(dim=0).min()) >= 100, "too few"
    ages = []
    last = np.zeros(B, dtype=int)
    for t in range(1, 201):
        for b in range(B):
            if bool(changed[t - 1, b]):
                ages.append(t - last[b])
                last[b] = t
    assert set(ages) <= {1, 2}, f"unexpected revolution gaps {sorted(set(ages))}"
    # how stale is the observation the brain actually acts on?
    age_sum, age_n = 0.0, 0
    for b in range(B):
        last_del = 0
        for t in range(ticks):
            if t > 0 and bool(changed[t - 1, b]):
                last_del = t
            age_sum += (t - last_del) * DT
            age_n += 1
    mean_age_ms = 1000.0 * age_sum / age_n
    # Content age AT the decision tick, counted from the tick that took the
    # revolution. With a locked 10:15 ratio the ages cycle {0, 0, dt} -> dt/3.
    # (Wall-clock age at use is up to one delivery tick larger: the tick grid
    # quantizes away the 0-67 ms between the sensor publishing and the brain
    # reading it, which averages ~50 ms on hardware timing.)
    assert 15.0 < mean_age_ms < 70.0, f"mean observation age {mean_age_ms:.1f} ms"
    reused = 1.0 - frac
    print(f"C2 ok: lidar 10 Hz vs brain 15 Hz — {frac:.3f} of ticks carry a "
          f"new revolution (expect {10 / CONTROL_HZ:.3f}), so {reused:.3f} of "
          f"decisions re-use content already acted on; gaps "
          f"{sorted(set(ages))}, mean content age {mean_age_ms:.1f} ms, "
          f"no phantom updates")


def c3_rate_semantics(device):
    # --- lidar only: 10 Hz, no stagger, 300 ticks of 1/15 s
    mc = MultiRateClock(ClocksConfig(lidar_hz=10.0, phase_stagger=False),
                        DT, 1, 360, device)
    n = 0
    gaps, last = [], -1
    for t in range(300):
        mc.tick()
        if bool(mc.lidar_due[0]):
            n += 1
            if last >= 0:
                gaps.append(t - last)
            last = t
            mc.scan_taken()
    assert set(gaps) == {1, 2}, gaps
    assert abs(n - 300 * 10 / CONTROL_HZ) <= 2, n
    lidar_gaps = sorted(set(gaps))

    # --- driver only: 5 Hz accepts one command in three (2 at the start:
    # the motors have no command yet, so the first one lands immediately).
    mc = MultiRateClock(ClocksConfig(driver_hz=5.0, phase_stagger=False),
                        DT, 1, 360, device)
    applied, due_ticks = [], []
    for i in range(30):
        mc.tick()
        # device matters: MultiRateClock lives on --device, and a CPU command
        # tensor against a CUDA clock is the bug this test caught on the Spark
        cmd = torch.full((1, 2), float(i + 1), device=mc.device)
        a = float(mc.latch_cmd(cmd)[0, 0])
        if bool(mc.driver_due[0]):
            due_ticks.append(i)
            mc.cmd_taken()
        applied.append(a)
    gaps = [b - a for a, b in zip(due_ticks, due_ticks[1:])]
    assert due_ticks == [0] + list(range(2, 30, 3)), due_ticks
    assert gaps == [2] + [3] * 9, f"acceptance gaps {gaps}"
    assert due_ticks[0] == 0, "the first command must land immediately"
    # held: every tick between acceptances repeats the last accepted command
    for a, b in zip(due_ticks, due_ticks[1:]):
        assert applied[a:b] == [float(a + 1)] * (b - a), applied[a:b]
    assert applied[due_ticks[-1]:] == [float(due_ticks[-1] + 1)] * (
        30 - due_ticks[-1]), applied[due_ticks[-1]:]
    assert len(set(applied)) == len(due_ticks), "a dropped command landed"
    print(f"C3 ok: lidar {n} revolutions in 300 ticks (gaps {lidar_gaps});"
          f" 5 Hz driver accepted {len(due_ticks)} of 30 commands at ticks "
          f"{due_ticks[:5]}... (latched: {len(set(applied))} distinct applied "
          f"values)")


def c4_phase_stagger(device):
    lock = MultiRateClock(ClocksConfig(lidar_hz=10.0, phase_stagger=False),
                          DT, 16, 360, device)
    stag = MultiRateClock(ClocksConfig(lidar_hz=10.0, phase_stagger=True),
                          DT, 16, 360, device)
    lock.tick()
    assert int(lock.lidar_due.sum()) == 16, "unstaggered clocks must agree"
    stag.tick()
    k = int(stag.lidar_due.sum())
    assert 0 < k < 16, f"staggered batch should split: {k} due"
    # deterministic: same config -> same phase vector, and it covers both
    # possible initial phases at 10/15 Hz
    again = MultiRateClock(ClocksConfig(lidar_hz=10.0, phase_stagger=True),
                           DT, 16, 360, device)
    assert torch.equal(stag._lidar_next, again._lidar_next)
    first = []
    for b in range(16):
        nxt = float(again._lidar_next[b])
        first.append(1 if nxt <= DT else 2)     # first due tick index + 1
    assert set(first) == {1, 2}, first
    print(f"C4 ok: lockstep {int(lock.lidar_due.sum())}/16 due, staggered "
          f"{k}/16 (first due tick split {sorted(set(first))}, "
          f"deterministic)")


def c5_engine_integration(device):
    legacy = _engine(device=device)
    clocked = _engine(device=device,
                      clocks=ClocksConfig(lidar_hz=10.0, driver_hz=25.0))
    m_leg = legacy.run_games(_scan_reading_policy, 600)
    m_clk = clocked.run_games(_scan_reading_policy, 600)
    for k, v in m_clk.items():
        assert bool(torch.isfinite(v).all()), f"{k} not finite: {v}"
    # a policy that ignores the scan cannot see the clock at all
    z1 = _engine(device=device).run_games(_zero_policy, 200)
    z2 = _engine(device=device,
                 clocks=ClocksConfig(lidar_hz=10.0)).run_games(_zero_policy, 200)
    assert torch.equal(z1["dist_m"], z2["dist_m"])
    assert float(z1["dist_m"].max()) == 0.0, "zero command moved the rover"
    # ...and a scan-reading policy must notice: staleness changes the rollout
    assert not torch.equal(m_leg["dist_m"], m_clk["dist_m"]), \
        "clocks made no difference to a scan-driven rollout"
    dmax = float((m_leg["dist_m"] - m_clk["dist_m"]).abs().max())
    dmean = float((m_leg["dist_m"] - m_clk["dist_m"]).abs().mean())
    assert dmax > 0.05, f"staleness barely reached the metrics ({dmax:.4f} m)"
    print(f"C5 ok: clocks rollout finite; zero-policy identical "
          f"(dist {float(z1['dist_m'].mean()):.4f}); scan-driven rollout "
          f"differs (|Δdist| mean {dmean:.3f} m, max {dmax:.3f} m)")


def c6_capture(device):
    if not str(device).startswith("cuda"):
        print("C6 skip: capture path is CUDA-only (--device cuda)")
        return
    eng = _engine(1, 8, device=device, fp16=True,
                  clocks=ClocksConfig(lidar_hz=10.0, driver_hz=25.0))
    m = eng.run_games(_scan_reading_policy, 200, graph=True)
    assert not getattr(eng, "_graph_broken", False), "graph capture failed"
    for k, v in m.items():
        assert bool(torch.isfinite(v).all()), f"{k} not finite"
    m2 = eng.run_games(_scan_reading_policy, 200, graph=True)
    assert torch.equal(m["rooms"], m2["rooms"]), "replay not repeatable"
    print(f"C6 ok: clocks engine captured + replayed a CUDA graph "
          f"(dist {float(m['dist_m'].mean()):.3f} m, "
          f"collisions {int(m['collisions'].sum())})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    c1_inert_when_off(args.device)
    c2_lidar_hold(args.device)
    c3_rate_semantics(args.device)
    c4_phase_stagger(args.device)
    c5_engine_integration(args.device)
    c6_capture(args.device)
    print("ALL CLOCK CHECKS PASSED")


if __name__ == "__main__":
    main()
