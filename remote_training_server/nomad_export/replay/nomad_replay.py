#!/usr/bin/env python3
"""NoMaD fan-selection replay harness — sim, not hardware.

Question this answers: does NoMaD's swag (left/right oscillation, the
"which of the drawn fan do we follow?" problem) get fixed by a LIDAR-SCORED
trajectory selector instead of the jitter band-aids the rover runner uses
(deterministic seed + median-fan pick + waypoint EMA)?

Three selectors, identical everything else (same episode seeds, same
diffusion noise stream, same controller, same safety gate):

  first   — upstream: follow sample 0 of the fan
  median  — the rover runner's heuristic: sample whose heading at
            waypoint_index is closest to the fan median
  lidar   — score every trajectory against world geometry (clearance,
            forward progress, curvature); infeasible (colliding) samples
            are excluded; hysteresis keeps the committed trajectory unless
            another beats it by a margin

Renderer: egocentric "depth panorama" from the sim world's wall segments
(lidar rays -> range-shaded vertical wall bands, doorway gaps appear as
floor-only columns). The pretrained encoder saw real photos + RECON depth,
so absolute policy quality on this renderer is NOT what we measure — the
WITHIN-run comparison across selectors is.

Usage (on the Spark):
  source /home/benson/venv/bin/activate
  python nomad_replay.py --buildings 6 --seconds 60 --out ~/nomad_replay
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
VINT_REPO = os.path.expanduser("~/visualnav-transformer")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/diffusion_policy"))
sys.path.insert(0, os.path.join(REPO, "remote_training_server", "nomad_export"))
sys.path.insert(0, os.path.join(REPO, "src", "tractor_bringup"))

from export_nomad_onnx import load_nomad                      # noqa: E402
from tractor_bringup.nomad_scheduler import NoMaDDDPMScheduler  # noqa: E402
from pnn_sim.buildings import make_level                       # noqa: E402
from pnn_sim.rover import SimRover, RoverConfig                # noqa: E402
from pnn_sim.safety_gate import SimSafetyGate, GateConfig      # noqa: E402

# ---------------------------------------------------------------- constants
CONTEXT_SIZE = 3
NUM_OBS_FRAMES = CONTEXT_SIZE + 1
IMAGE_SIZE = 96
PRED_HORIZON = 8
ENCODING_SIZE = 256
NUM_DIFFUSION_ITERS = 10
NUM_SAMPLES = 8
WPT_SCALE = 0.25          # m per waypoint (runner default; NoMaD spacing)
POLICY_HZ = 4.0           # reference deployment cadence (robot.yaml frame_rate)
SIM_DT = 0.1              # 10 Hz sim = lidar rate
V_MAX = 0.2               # m/s per track at cmd=1 (RoverConfig)
TRACK_W = 0.154
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
ACTION_MIN = np.array([-2.5, -4.0], np.float32)
ACTION_MAX = np.array([5.0, 4.0], np.float32)

# camera (renderer) — square FOV like the reference's center-cropped inputs
CAM_H = 0.35              # m above ground
WALL_H = 2.0              # m wall height drawn
FOV = math.radians(90.0)


def _viridis():
    import matplotlib.cm as cm
    lut = cm.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]
    return (lut * 255).astype(np.uint8)


VIRIDIS = _viridis()


def render_view(world, x: float, y: float, theta: float,
                max_range: float = 11.9) -> np.ndarray:
    """Egocentric depth-panorama RGB (96x96x3 uint8). Columns are camera
    rays; walls drawn as range-shaded bands from ground contact upward."""
    w = IMAGE_SIZE
    img = np.full((w, w, 3), 190, np.uint8)          # ceiling/sky gray
    fx = (w - 1) / 2.0 / math.tan(FOV / 2.0)
    cx = cy = (w - 1) / 2.0
    col_ang = theta + np.arctan((np.arange(w) - cx) / fx)
    r = world.raycast(x, y, col_ang, max_range)      # [w]
    floor_y = cy + fx * CAM_H / np.maximum(np.cos(np.arctan((np.arange(w) - cx) / fx)) * 0.3, 1e-3)
    for j in range(w):
        rj = r[j]
        if not np.isfinite(rj) or rj >= max_range - 0.05:
            rj = max_range                            # far gap -> horizon
        rx = max(rj * math.cos((j - cx) / fx), 0.05)  # forward depth of this column
        v_base = int(round(cy + fx * CAM_H / rx))     # wall/ground contact row
        v_top = int(round(cy - fx * (WALL_H - CAM_H) / rx))
        v_base = min(max(v_base, 0), w - 1)
        v_top = min(max(v_top, 0), w - 1)
        shade = VIRIDIS[int(np.clip(rj / max_range, 0, 1) * 255)]
        img[v_top:v_base + 1, j] = shade              # wall band
        # floor below the contact row: darkened range shade
        fb = max(v_base + 1, 0)
        if fb < w:
            img[fb:, j] = (shade * 0.35).astype(np.uint8)
    return img


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - torch.from_numpy(IMAGENET_MEAN)) / torch.from_numpy(IMAGENET_STD)


# ---------------------------------------------------------------- selectors
def unnormalize_traj(naction: np.ndarray) -> np.ndarray:
    """(N,8,2) in [-1,1] -> metric metres (mirrors train_utils.get_action)."""
    d = (naction + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
    return np.cumsum(d, axis=1) * WPT_SCALE


def sample_paths(traj: np.ndarray, n: int = 6) -> np.ndarray:
    """(8,2) waypoints -> (n,2) dense path points, robot frame."""
    ts = np.linspace(0, len(traj) - 1, n)
    return np.stack([np.interp(ts, np.arange(len(traj)), traj[:, k]) for k in (0, 1)], axis=1)


def score_trajs(trajs: np.ndarray, world, x: float, y: float, theta: float,
                radius: float) -> np.ndarray:
    """(N,8,2) -> (N,) scores. Robot-frame geometry, exact world segments."""
    n = trajs.shape[0]
    scores = np.full(n, -np.inf)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    for i in range(n):
        pts = np.asarray(trajs[i], np.float64)
        pts = np.vstack([np.zeros(2), pts])           # include current pos
        dense = sample_paths(pts[1:], 8)
        world_pts = dense @ rot.T + np.array([x, y])
        clear = np.array([world.clearance(px, py) for px, py in world_pts])
        if clear.min() < radius + 0.02:
            continue                                   # infeasible — will scrape
        heading = rot[0]                               # world +x of robot
        progress = float((world_pts[-1] - [x, y]) @ heading)
        # path curvature = mean |turn angle| along the dense path
        seg = np.diff(np.vstack([[0, 0], dense]), axis=0)
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        d_ang = np.abs(np.diff((ang + np.pi) % (2 * np.pi) - np.pi))
        curv = float(d_ang.mean())
        scores[i] = min(clear.min(), 0.5) * 2.0 + progress * 0.5 - curv * 0.3
    if not np.isfinite(scores).any():                  # everything collides
        for i in range(n):
            pts = np.vstack([np.zeros(2), np.asarray(trajs[i], np.float64)])
            dense = sample_paths(pts[1:], 8)
            world_pts = dense @ rot.T + np.array([x, y])
            clear = np.array([world.clearance(px, py) for px, py in world_pts])
            scores[i] = clear.min()
    return scores


# ---------------------------------------------------------------- policy
class NomadPolicy:
    def __init__(self, ckpt: str, device: str):
        model, params = load_nomad(ckpt, VINT_REPO)
        self.model = model.to(device).eval()
        self.device = device
        self.sched = NoMaDDDPMScheduler(NUM_DIFFUSION_ITERS)
        self.tsteps = self.sched.timesteps

    @torch.no_grad()
    def fan(self, context: list[np.ndarray], rng: np.random.Generator):
        """context: last NUM_OBS_FRAMES rendered RGB uint8 (oldest first).
        Returns (N,8,2) metric trajectories + raw naction (N,8,2)."""
        obs = torch.cat([to_tensor(f) for f in context], dim=0)[None].to(self.device)
        goal = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        mask = torch.ones(1, dtype=torch.long, device=self.device)   # exploration
        cond = self.model("vision_encoder", obs_img=obs, goal_img=goal,
                          input_goal_mask=mask)
        if cond.ndim == 3:
            cond = cond.flatten(start_dim=1)
        cond = cond.repeat(NUM_SAMPLES, 1)
        naction = torch.from_numpy(
            rng.standard_normal((NUM_SAMPLES, PRED_HORIZON, 2)).astype(np.float32)
        ).to(self.device)
        for k in self.tsteps:
            eps = self.model("noise_pred_net", sample=naction,
                             timestep=torch.tensor(int(k), device=self.device)
                             .expand(NUM_SAMPLES), global_cond=cond)
            naction = torch.from_numpy(
                self.sched.step(model_output=eps.cpu().numpy(),
                                timestep=int(k),
                                sample=naction.cpu().numpy(),
                                noise=rng.standard_normal(
                                    (NUM_SAMPLES, PRED_HORIZON, 2)).astype(np.float64)
                                ).astype(np.float32)
            ).to(self.device)
        return unnormalize_traj(naction.cpu().numpy())


# ---------------------------------------------------------------- rollout
@dataclass
class Episode:
    selector: str
    seed: int
    rooms: int = 0
    displ: float = 0.0
    dist: float = 0.0
    stops: int = 0
    collided: bool = False
    jerk: float = 0.0          # mean |d(track_diff)| per second (sway proxy)
    reversals: float = 0.0     # sign changes of (right-left) per minute
    fan_std_mean: float = 0.0  # policy uncertainty (heading spread of fan)
    switch_rate: float = 0.0   # selector switches per policy call
    path: list = field(default_factory=list)


def pure_pursuit(path: np.ndarray, v_ref: float, lookahead: float = 0.5):
    """path (n,2) robot frame -> (v, omega) m/s, rad/s."""
    d = np.linalg.norm(path, axis=1)
    idx = int(np.argmax(d >= lookahead)) if (d >= lookahead).any() else len(path) - 1
    wp = path[idx]
    if wp[0] <= 0.05:                      # target behind/beside -> pivot in place
        return 0.0, float(np.clip(2.0 * math.atan2(wp[1], max(wp[0], 0.05)), -1.2, 1.2))
    alpha = math.atan2(wp[1], wp[0])
    omega = 2.0 * v_ref * math.sin(alpha) / max(lookahead, 0.1)
    return v_ref, float(np.clip(omega, -1.2, 1.2))


def vw_to_tracks(v: float, omega: float):
    left = (v - omega * TRACK_W / 2) / V_MAX
    right = (v + omega * TRACK_W / 2) / V_MAX
    return (float(np.clip(left, -1, 1)), float(np.clip(right, -1, 1)))


def rollout(policy: NomadPolicy, building, selector: str, seed: int,
            seconds: float) -> Episode:
    rng = np.random.default_rng(seed)
    cfg = RoverConfig(seed=seed)
    rover = SimRover(building, cfg)
    rover.x, rover.y, rover.theta = building.start_pose
    gate = SimSafetyGate(GateConfig(), lambda: t)
    ep = Episode(selector=selector, seed=seed)
    ctx: list[np.ndarray] = []
    t = 0.0
    chosen_path = None
    prev_chosen_idx = None
    prev_track_diff = 0.0
    last_diff_sign = 0
    jerk_acc, jerk_n, switches, calls, stop_state = 0.0, 0, 0, 0, False
    fan_stds = []
    policy_period = int(round(1.0 / POLICY_HZ / SIM_DT))
    x0, y0 = rover.x, rover.y

    for tick in range(int(seconds / SIM_DT)):
        ranges, amin, ainc = rover.scan()
        gate.process_scan(ranges, amin, ainc)
        if gate.front_blocked and not stop_state:
            ep.stops += 1
        stop_state = gate.front_blocked

        if tick % policy_period == 0:
            img = render_view(building, rover.x, rover.y, rover.theta)
            ctx.append(img)
            if len(ctx) > NUM_OBS_FRAMES:
                ctx.pop(0)
            if len(ctx) == NUM_OBS_FRAMES:
                trajs = policy.fan(ctx, rng)
                # fan heading spread (policy uncertainty)
                head = np.arctan2(trajs[:, 2, 1], np.maximum(trajs[:, 2, 0], 0.05))
                fan_stds.append(float(np.std(head)))
                if selector == "first":
                    idx = 0
                elif selector == "median":
                    med = float(np.median(head))
                    idx = int(np.argmin(np.abs(head - med)))
                else:  # lidar
                    sc = score_trajs(trajs, building, rover.x, rover.y,
                                     rover.theta, cfg.robot_radius)
                    best = int(np.argmax(sc))
                    idx = best
                    if (prev_chosen_idx is not None and np.isfinite(sc[prev_chosen_idx])
                            and sc[prev_chosen_idx] >= sc[best] - 0.10):
                        idx = prev_chosen_idx          # hysteresis: keep committed
                if prev_chosen_idx is not None and idx != prev_chosen_idx:
                    switches += 1
                prev_chosen_idx = idx
                calls += 1
                # EMA smoothing identical across selectors (runner parity)
                cp = trajs[idx]
                chosen_path = cp if chosen_path is None else 0.5 * cp + 0.5 * chosen_path
        # control at sim rate
        if chosen_path is not None:
            v, w = pure_pursuit(sample_paths(chosen_path, 8), v_ref=V_MAX * 1.5)
            if stop_state and v > 0:
                v = 0.0
            cl, cr = vw_to_tracks(v, w)
        else:
            cl = cr = 0.0
        d = cr - cl
        if abs(d - prev_track_diff) > 1e-9:
            jerk_acc += abs(d - prev_track_diff)
            jerk_n += 1
        s = int(np.sign(d)) if abs(d) > 0.05 else last_diff_sign
        if s != 0 and last_diff_sign != 0 and s != last_diff_sign:
            ep.reversals += 1
        if s != 0:
            last_diff_sign = s
        prev_track_diff = d
        rover.step(cl, cr, SIM_DT)
        t += SIM_DT
        if tick % 4 == 0:
            ep.path.append((rover.x, rover.y, rover.theta))
        if rover.collided:
            ep.collided = True

    ep.rooms = len({building.room_id(px, py) for px, py, _ in ep.path
                    if building.room_id(px, py) >= 0})
    ep.displ = math.hypot(rover.x - x0, rover.y - y0)
    p = np.asarray(ep.path)
    ep.dist = float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
    ep.jerk = jerk_acc / max(jerk_n, 1) / SIM_DT
    ep.reversals = ep.reversals / (seconds / 60.0)
    ep.fan_std_mean = float(np.mean(fan_stds)) if fan_stds else 0.0
    ep.switch_rate = switches / max(calls, 1)
    return ep


def draw(building, eps: list[Episode], out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for a, b, c, d in building.segments:
        ax.plot([a, c], [b, d], "k-", lw=1.2)
    cols = {"first": "tab:red", "median": "tab:orange", "lidar": "tab:green"}
    for ep in eps:
        p = np.asarray(ep.path)
        ax.plot(p[:, 0], p[:, 1], color=cols[ep.selector], lw=1.2, alpha=0.85,
                label=f"{ep.selector}: rooms={ep.rooms} stops={ep.stops} "
                      f"sway={ep.jerk:.2f} rev/min={ep.reversals:.1f}")
    sx, sy, _ = building.start_pose
    ax.plot(sx, sy, "b^", ms=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(os.path.basename(out_png))
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/nomad.pth"))
    ap.add_argument("--buildings", type=int, default=6)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.expanduser("~/nomad_replay"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-base", type=int, default=91000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    policy = NomadPolicy(args.ckpt, args.device)
    print(f"policy loaded on {args.device} in {time.time()-t0:.1f}s", flush=True)

    results = []
    for bi in range(args.buildings):
        rng = np.random.default_rng(args.seed_base + bi)
        building = make_level(rng, args.level)
        eps = []
        for sel in ("first", "median", "lidar"):
            # identical episode conditions: one fresh RNG stream per selector
            # derived from the SAME seed => identical start, noise, sim noise
            ep = rollout(policy, building, sel, args.seed_base + 5000 + bi,
                         args.seconds)
            eps.append(ep)
            results.append({**{k: v for k, v in vars(ep).items() if k != "path"},
                            "building": bi})
            print(f"b{bi} {sel:7s} rooms={ep.rooms} dist={ep.dist:5.1f}m "
                  f"displ={ep.displ:4.1f} stops={ep.stops} coll={ep.collided} "
                  f"sway={ep.jerk:.3f} rev/min={ep.reversals:4.1f} "
                  f"fan_std={ep.fan_std_mean:.2f} switch={ep.switch_rate:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        draw(building, eps, os.path.join(args.out, f"b{bi}.png"))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", args.out, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
