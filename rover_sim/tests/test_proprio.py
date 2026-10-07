"""Proprio truth + track-stall tests (docs/SIM_PLATFORM.md §8 items 2-3).

    python3 -m rover_sim.tests.test_proprio [--device cpu]

Two sim->real gaps that the platform must be able to express, because both
were paid for in the field:

* **Wheel proprio (item 2).** The sim feeds the lagged track speeds plus
  encoder noise. The DEPLOY runner cannot: the rover's right encoder reads 0
  while the track turns (live runs 5-8), so `evo/deploy/evo_runner.py
  --wheel-source model` substitutes the motor model. A policy trained on one
  and deployed on the other sees a different obs channel — the silent
  train/deploy skew class REVIVAL_PLAN lists as failure #4. `RoverConfig
  .wheel_source` now names all three realities.
* **Track stall (item 3).** A loaded track whose commanded speed is below the
  static-friction break-away does not move. This is why champ20m_idx115 —
  sim-best, 61% pivot ticks — sat PHYSICALLY STILL 56% of its ticks on the
  rover: pivot commands of +-0.15 are +-0.024 m/s of track speed, below the
  break-away. `RoverConfig.track_stall_speed` models it, and diff-drive
  kinematics then give the two observed behaviours for free: still, or an
  orbit about the stalled track.

W1  wheel_source="model"    -> wheel channels are exactly v/wheel_radius
W2  wheel_source="dead_right" -> the right channel is exactly 0
W3  wheel_source="true"     -> both channels carry encoder noise (legacy)
W4  engine integration      -> the three modes are distinguishable through the
                               obs, and a wheel-reading policy's rollout differs
S1  stall: both tracks below break-away -> the rover does not move at all
S2  stall: one track below, one above -> it orbits the stalled track at
    radius track_width / 2 (the live-run "orbits the dead wheel")
S3  stall off (legacy) -> the same command is an in-place pivot
S4  numpy SimRover and BatchedEnv agree under stall (identical pose trace)
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

try:
    torch.cuda.is_current_stream_capturing()
except Exception:
    torch.cuda.is_current_stream_capturing = lambda: False

from ..core.rover import RoverConfig, SimRover
from ..core.batched import BatchedEnv
from ..core.world import World
from ..runner.engine import RolloutEngine

DT = 1.0 / 15.0
SEG = np.array([[20.0, -20.0, 20.0, 20.0]], dtype=np.float64)   # far wall


def _world(start=(0.0, 0.0, 0.0)):
    return World(SEG, (-25.0, -25.0, 25.0, 25.0), start)


def _quiet(**kw) -> RoverConfig:
    """Rover with all noise off: the tests are about mechanism, not noise."""
    return replace(RoverConfig(), slip_std=0.0, lidar_noise_std=0.0,
                   gyro_noise_std=0.0, accel_noise_std=0.0, **kw)


def _numpy_trace(cfg, cmds, ticks=None, start=(0.0, 0.0, 0.0)):
    ticks = len(cmds) if ticks is None else ticks
    r = SimRover(_world(start), cfg)
    tr = []
    for i in range(ticks):
        r.step(cmds[i][0], cmds[i][1], DT)
        tr.append((r.x, r.y, r.theta, r.wheel_l, r.wheel_r,
                   r.v_left, r.v_right))
    return tr


def _batched_trace(cfg, cmds, ticks=None, start=(0.0, 0.0, 0.0)):
    ticks = len(cmds) if ticks is None else ticks
    env = BatchedEnv(1, cfg, seed=3, device="cpu")
    w = _world(start)
    env._worlds = [w]
    env._build_segments()
    env.x[:] = start[0]
    env.y[:] = start[1]
    env.theta[:] = start[2]
    tr = []
    for i in range(ticks):
        cmd = torch.tensor([[cmds[i][0], cmds[i][1]]], dtype=torch.float32)
        env.step(cmd, DT)
        tr.append((float(env.x[0]), float(env.y[0]), float(env.theta[0]),
                   float(env.wheel_l[0]), float(env.wheel_r[0]),
                   float(env.v_left[0]), float(env.v_right[0])))
    return tr


# ------------------------------------------------------------------ wheels
def w1_model_mode(device):
    """model = the deploy runner's substitution: exact v/r, no noise."""
    cfg = _quiet(wheel_source="model")
    tr = _numpy_trace(cfg, [(0.6, 0.3)] * 40)
    for x, y, th, wl, wr, vl, vr in tr:
        assert abs(wl - vl / cfg.wheel_radius) < 1e-12, (wl, vl)
        assert abs(wr - vr / cfg.wheel_radius) < 1e-12, (wr, vr)
    bt = _batched_trace(cfg, [(0.6, 0.3)] * 40)
    for x, y, th, wl, wr, vl, vr in bt:
        assert abs(wl - vl / cfg.wheel_radius) < 1e-6
        assert abs(wr - vr / cfg.wheel_radius) < 1e-6
    print("W1 ok: wheel_source='model' -> both channels exactly v/wheel_radius "
          f"({tr[-1][3]:.3f}, {tr[-1][4]:.3f} rad/s), no encoder noise")


def w2_dead_right(device):
    """dead_right = the rover's actual defect: the right channel is 0."""
    cfg = _quiet(wheel_source="dead_right")
    tr = _numpy_trace(cfg, [(0.6, 0.3)] * 40)
    assert all(abs(t[4]) == 0.0 for t in tr), "right encoder is not dead"
    assert abs(tr[-1][3]) > 1.0, "left encoder should show motion"
    bt = _batched_trace(cfg, [(0.6, 0.3)] * 40)
    assert all(t[4] == 0.0 for t in bt)
    assert abs(bt[-1][3]) > 1.0
    print(f"W2 ok: wheel_source='dead_right' -> right channel exactly 0 "
          f"(left {bt[-1][3]:.3f} rad/s while the track turns at "
          f"{bt[-1][5]:.3f} m/s)")


def w3_legacy_true(device):
    """'true' is the legacy path: noisy encoders around v/r, both sides."""
    cfg = _quiet(wheel_source="true")
    tr = _numpy_trace(cfg, [(0.6, 0.3)] * 40)
    x, y, th, wl, wr, vl, vr = tr[-1]
    assert abs(wl - vl / cfg.wheel_radius) > 0.0, "no left noise"
    assert abs(wr - vr / cfg.wheel_radius) > 0.0, "no right noise"
    assert abs(wr - vr / cfg.wheel_radius) < 0.2, "noise off scale"
    bt = _batched_trace(cfg, [(0.6, 0.3)] * 40)
    assert abs(bt[-1][4] - bt[-1][6] / cfg.wheel_radius) < 0.2
    print("W3 ok: wheel_source='true' (legacy) keeps noisy encoders on both "
          f"sides (right off by {abs(wr - vr / cfg.wheel_radius):.4f} rad/s)")


def w4_engine_obs(device):
    """The modes must be distinguishable through the obs the policy sees."""
    def wheels(obs, prev):
        # steer on the wheel channels only: a policy that lives on proprio
        return torch.stack([(obs[:, 1] - 0.5) * 2.0,
                            (obs[:, 0] - 0.5) * 2.0], dim=1)

    out = {}
    for mode in ("true", "model", "dead_right"):
        eng = RolloutEngine(1, 4, seed=777_000, noise_seed=99,
                            device=device, cells=False,
                            rover_cfg=_quiet(wheel_source=mode))
        out[mode] = eng.run_games(wheels, 300)
    assert not torch.equal(out["true"]["dist_m"], out["model"]["dist_m"]), \
        "model and legacy encoder obs produced identical rollouts"
    assert not torch.equal(out["model"]["dist_m"],
                           out["dead_right"]["dist_m"]), \
        "dead_right and model produced identical rollouts"
    for mode, m in out.items():
        assert bool(torch.isfinite(m["dist_m"]).all()), mode
    print("W4 ok: through the obs, the three wheel realities give different "
          "rollouts (" + ", ".join(
              f"{k} {float(v['dist_m'].mean()):.2f} m" for k, v in out.items())
          + ")")


# ------------------------------------------------------------------- stall
def s1_both_stalled(device):
    """A pivot below break-away must be a dead stop, not a free pirouette."""
    cmds = [(0.15, -0.15)] * 60          # 0.024 / 0.030 m/s of track speed
    cfg = _quiet(track_stall_speed=0.05)
    tr = _numpy_trace(cfg, cmds)
    d = max(abs(t[0]) + abs(t[1]) for t in tr)
    assert d == 0.0, f"rover moved {d} m on a stalled pivot"
    assert all(t[5] == 0.0 and t[6] == 0.0 for t in tr), "tracks not stalled"
    assert all(abs(t[2]) == 0.0 for t in tr), "yaw on a stalled pivot"
    bt = _batched_trace(cfg, cmds)
    assert max(abs(t[0]) + abs(t[1]) for t in bt) == 0.0
    assert all(t[5] == 0.0 and t[6] == 0.0 for t in bt)
    print("S1 ok: both tracks under break-away -> dead still (0.000 m, 0 rad "
          "of yaw in 60 ticks at +-0.15 command)")


def s2_orbit(device):
    """One track stalled -> the rover orbits it at radius track_width/2."""
    cfg = _quiet(track_stall_speed=0.05)
    tr = _numpy_trace(cfg, [(0.0, 0.5)] * 60)
    assert all(abs(t[5]) == 0.0 for t in tr), "left track should be stalled"
    assert all(t[6] > 0.0 for t in tr), "right track should drive"
    radii = []
    for a, b in zip(tr[10:], tr[11:]):
        dp = np.hypot(b[0] - a[0], b[1] - a[1])
        dth = abs(b[2] - a[2])
        assert dth > 0, "no rotation"
        radii.append(dp / dth)
    r = float(np.mean(radii))
    assert abs(r - cfg.track_width / 2) < 0.005, \
        f"orbit radius {r:.4f} != track_width/2 {cfg.track_width / 2:.4f}"
    print(f"S2 ok: one stalled track -> orbit about it at radius {r:.4f} m "
          f"(= track_width/2 {cfg.track_width / 2:.4f}; the live-run signature)")


def s3_stall_off(device):
    """Legacy (no stall): the same command is a free in-place pivot."""
    cfg = _quiet()                        # track_stall_speed = 0.0
    tr = _numpy_trace(cfg, [(0.15, -0.15)] * 60)
    assert abs(tr[-1][2]) > 0.5, "no yaw in the legacy pivot"
    drift = np.hypot(tr[-1][0], tr[-1][1])
    assert drift < 0.02, f"legacy pivot translated {drift:.3f} m"
    print(f"S3 ok: stall off -> in-place pivot ({abs(tr[-1][2]):.2f} rad of "
          f"yaw, {drift:.4f} m of drift) — the behaviour the real rover "
          "could not reproduce")


def s4_parity(device):
    """numpy SimRover and BatchedEnv must agree under stall."""
    cfg = _quiet(track_stall_speed=0.05)
    cmds = [(0.0, 0.5)] * 8 + [(0.4, 0.4)] * 8 + [(0.15, -0.15)] * 8
    a = _numpy_trace(cfg, cmds)
    b = _batched_trace(cfg, cmds)
    for i, (ta, tb) in enumerate(zip(a, b)):
        dx = abs(ta[0] - tb[0])
        dy = abs(ta[1] - tb[1])
        dth = abs(ta[2] - tb[2])
        assert dx < 1e-5 and dy < 1e-5 and dth < 1e-5, (i, ta, tb)
    print("S4 ok: batched env == numpy SimRover under stall "
          f"(max |Δpose| {max(max(abs(x[0]-y[0]), abs(x[1]-y[1])) for x, y in zip(a, b)):.2e} m)")


def t1_trim_vs_bias(device):
    """The 0.8 left trim is a CORRECTION for a mechanically faster left track.
    Modelled as a bias, a correctly-trimmed rover drives STRAIGHT on cmd
    (1,1); the legacy sim folds the bias into the trim and curves left."""
    cmds = [(0.5, 0.5)] * 120
    legacy = _numpy_trace(_quiet(), cmds)                     # bias (1,1)
    true = _numpy_trace(_quiet(track_bias=(1.25, 1.0)), cmds)  # real pair
    dy_legacy = abs(legacy[-1][2])
    dy_true = abs(true[-1][2])
    assert dy_legacy > 0.5, f"legacy straight command should curve ({dy_legacy})"
    assert dy_true < 0.01, f"corrected pair must run straight ({dy_true})"
    print(f"T1 ok: cmd (0.5, 0.5) — legacy trim-only model yaws "
          f"{dy_legacy:.3f} rad in 8 s (a false curve ES learns to fight); with "
          f"track_bias=(1.25, 1.0) it yaws {dy_true:.2e} rad")


def t2_bias_parity(device):
    cfg = _quiet(track_bias=(1.25, 1.0))
    cmds = [(0.6, 0.3)] * 10 + [(0.5, 0.5)] * 10
    a, b = _numpy_trace(cfg, cmds), _batched_trace(cfg, cmds)
    for i, (ta, tb) in enumerate(zip(a, b)):
        assert abs(ta[0] - tb[0]) < 1e-5 and abs(ta[2] - tb[2]) < 1e-5, (i, ta)
    print("T2 ok: batched env == numpy SimRover with the mechanical bias")


def t3_bias_vs_encoder(device):
    """Item 2 + item 5 together: the ENCODER still reports the trimmed wheel
    speed (which is why evo_runner's wheel model stays trimmed) while the BODY
    is symmetric."""
    cfg = _quiet(track_bias=(1.25, 1.0), wheel_source="model")
    tr = _numpy_trace(cfg, [(0.5, 0.5)] * 60)
    x, y, th, wl, wr, vl, vr = tr[-1]
    ratio = wl / wr if wr else float("nan")
    assert abs(ratio - 0.8) < 0.01, f"encoder ratio {ratio}"
    assert abs(th) < 1e-6, "body should be straight"
    print(f"T3 ok: encoder still reports the 0.8 ratio ({ratio:.4f}) while the "
          f"body drives straight (yaw {abs(th):.2e} rad) — encoder != ground")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    w1_model_mode(args.device)
    w2_dead_right(args.device)
    w3_legacy_true(args.device)
    w4_engine_obs(args.device)
    s1_both_stalled(args.device)
    s2_orbit(args.device)
    s3_stall_off(args.device)
    s4_parity(args.device)
    t1_trim_vs_bias(args.device)
    t2_bias_parity(args.device)
    t3_bias_vs_encoder(args.device)
    print("ALL PROPRIO/STALL CHECKS PASSED")


if __name__ == "__main__":
    main()
