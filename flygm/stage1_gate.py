#!/usr/bin/env python3
"""Stage-1 gate: replay rover bags through the fly connectome graph (FlyGM).

Replays an extracted observation stream (flygm/bag_extract.py npz) through the
MaleCNS graph the way the rover runner would feed it:
  - encoder: per-tick obs -> 32-d, injected into afferent neurons (tanh gate)
  - one graph step per lidar tick: M = W H, H' = tanh-MLP([M || eta]), decay
  - decoder: fixed random readout of the efferent partition -> (lin, ang)
    proxy commands (decoder LEARNING is stage 2; this gate tests the graph)

Gates (docs/FLYGM_PLAN.md):
  A (stationary): efferent readout drift settles ~0 with frozen weights — no
    self-excitation runaway, matches the PCN stationary temperament verdict.
  B (tour): efferent stream tracks corridor->room transitions with median
    |cmd| separation BETTER than a degree-preserving shuffled-W control
    (FlyGM's own ablation logic, replicated on our data).
  B2 (tour, PCN cross-check): PlaceMemory with the field-validated
    temperament on the same ticks — segmentation must match tour_analyze's
    k-center ceiling; the fly graph must beat shuffled-W on the same metric.

Run on the DGX Spark: /home/benson/venv/bin/python flygm/stage1_gate.py \
    --obs /tmp/flygm_stationary.npz --label stationary [--vis-weight 0.5]
"""
import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/benson/flygm")

C = 32          # latent channels per neuron (FlyGM)
TAU_H = 0.5     # s, state decay
DT = 0.1        # s per lidar tick


# ---------------------------------------------------------------- graph ----
class FlyGraph:
    """MaleCNS message-passing step (FlyGM §3.1) on the built signed CSR W."""

    def __init__(self, W, aff_idx, eff_idx, C=C, tau_h=TAU_H, dt=DT, seed=0):
        self.C = C
        self.dt = dt
        self.tau_h = tau_h
        self.N = W.shape[0]
        self.aff_idx = np.asarray(aff_idx, dtype=np.int64)
        self.eff_idx = np.asarray(eff_idx, dtype=np.int64)
        self.W = W.tocsr()  # scipy CSR: numpy-path sparse matvec
        rng = np.random.default_rng(seed)
        # eta: per-neuron intrinsic descriptor (FlyGM trains it; fixed random
        # here — the gate asks whether wiring structure alone carries signal)
        self.eta = (0.5 * rng.standard_normal((self.N, C))).astype(np.float32)
        self.psi1_w = (0.2 * rng.standard_normal((2 * C, C))).astype(np.float32)
        self.psi2_w = (0.2 * rng.standard_normal((C, C))).astype(np.float32)
        # FlyGM eq.4 injection: obs -> afferents via a shared gate map
        self.Wg = (0.5 * rng.standard_normal((C, C))).astype(np.float32)
        self.H = np.zeros((self.N, C), dtype=np.float32)

    def step(self, obs_enc: np.ndarray) -> np.ndarray:
        """obs_enc: (C,) encoded observation (run_replay encodes).
        numpy/scipy path: the sparse W@H dominates and scipy's CSR kernel
        streams only W (~110 MB) while H (9.6 MB) stays cache-resident —
        ~10x faster than torch CPU sparse mm here."""
        M = self.W @ self.H                       # scipy CSR x dense (N,C)
        inj = np.tanh(obs_enc @ self.Wg)          # (C,) FlyGM eq.4 gate map
        gate = np.zeros((self.N, self.C), dtype=np.float32)
        gate[self.aff_idx] = inj
        Z = np.tanh(np.hstack([M, gate]) @ self.psi1_w)
        Hn = np.tanh(Z @ self.psi2_w + self.eta)
        # leaky integration toward the new state (tau_h)
        a = float(np.clip(self.dt / self.tau_h, 0.0, 1.0))
        self.H = (self.H + a * (Hn - self.H)).astype(np.float32)
        return self.H

    def readout(self) -> np.ndarray:
        """Per-efferent-neuron channel-mean state -> (n_eff,) vector that the
        (2, n_eff) random decoder maps to (lin, ang)."""
        return self.H[self.eff_idx].mean(axis=1)


def shuffled_W(W, seed=1):
    """Degree-preserving rewiring (configuration-model style, FlyGM's
    control): independently permute the row- and column- endpoint arrays of
    the edge list. Each node's total out-/in-degree COUNT is preserved
    exactly (the multisets are unchanged); partners are randomized."""
    rng = np.random.default_rng(seed)
    coo = W.tocoo()
    rows = rng.permutation(coo.row)
    cols = rng.permutation(coo.col)
    Ws = sparse_from_coo(rows, cols, coo.data, W.shape)
    Ws.eliminate_zeros()
    diff = int((W != Ws).nnz)
    print(f"[shuffled-W] rewired edges differing from real: {diff:,} / {W.nnz:,}")
    assert diff > 0.9 * W.nnz, "shuffled-W control did not actually rewire"
    return Ws


def sparse_from_coo(rows, cols, data, shape):
    from scipy import sparse

    return sparse.csr_matrix((data, (rows, cols)), shape=shape)


# ------------------------------------------------------------- obs path ----
def vis_ema_series(ts_scan, ts_img, vis_fps, tau=6.0, stale=2.0):
    """Runner replica: freshest frame per tick, EMA tau, 2 s stale -> None."""
    out, ema, last = [], None, None
    j = 0
    for t in ts_scan:
        while j + 1 < len(ts_img) and ts_img[j + 1] <= t:
            j += 1
        fp = None
        if len(ts_img) and (t - ts_img[j]) <= stale:
            fp = vis_fps[j]
        if fp is not None:
            ema = fp.copy() if ema is None else ema + min(
                1.0, (t - (last or t)) / tau) * (fp - ema)
            last = t
        out.append(ema)
    return out


def fuse_obs(ts_scan, fps, ts_img, vis_fps, vis_weight, tau_vis=6.0):
    n = len(ts_scan)
    obs = np.zeros((n, 72 + 23), dtype=np.float32)
    obs[:, :72] = fps
    ema = vis_ema_series(ts_scan, ts_img, vis_fps, tau=tau_vis)
    have = np.array([e is not None for e in ema])
    obs[have, 72:] = np.stack([e for e in ema if e is not None])
    return obs * np.concatenate([np.ones(72, dtype=np.float32),
                                 np.full(23, vis_weight, dtype=np.float32)])


# ----------------------------------------------------------------- gates ----
def run_replay(W, part, body_ids, obs, label, vis_note=""):
    aff_mask = (part.role == "afferent").to_numpy()
    eff = part[part.role == "efferent"].bodyId.to_numpy()
    # Drive only afferents with surviving downstream synapses: after the
    # ol_intrinsic prune, ol_sensory photoreceptors mostly synapse onto the
    # DROPPED retinotopic stack — injecting them feeds dead ends. W[post,pre]
    # => a neuron's outgoing edges live in its COLUMN.
    alive_pre = np.asarray((W != 0).sum(axis=0)).ravel() > 0
    aff_idx = np.where(aff_mask & alive_pre)[0]
    dead_aff = int((aff_mask & ~alive_pre).sum())
    eff_idx = np.where(np.isin(body_ids, eff))[0]
    print(f"[{label}] afferent {len(aff_idx)} driven ({dead_aff} post-prune "
          f"dead-ends skipped), efferent {len(eff_idx)} {vis_note}")

    g = FlyGraph(W, aff_idx, eff_idx)
    rng = np.random.default_rng(7)
    R = (0.1 * rng.standard_normal((2, len(eff_idx)))).astype(np.float32)
    # fixed random encoder obs_dim -> C (stage 2 trains it; gate tests graph)
    enc_P = (np.random.default_rng(3).standard_normal((obs.shape[1], C))
             / np.sqrt(obs.shape[1])).astype(np.float32)
    cmds = np.zeros((len(obs), 2), dtype=np.float32)
    t0 = time.time()
    for k, o in enumerate(obs):
        g.step(np.tanh(o @ enc_P))
        cmds[k] = R @ g.readout()
    wall = time.time() - t0
    return cmds, wall


def seg_summary(cmds, chunk_s=30.0):
    """Coarse movement statistics per 30 s chunk (the gate metric)."""
    n = len(cmds)
    ch = max(1, int(chunk_s / DT))
    rows = []
    for s in range(0, n, ch):
        c = cmds[s:s + ch]
        rows.append(dict(t0=s * DT, lin_mean=float(c[:, 0].mean()),
                         ang_mean=float(c[:, 1].mean()),
                         ang_std=float(c[:, 1].std())))
    return rows


def load_pruned_graph():
    """Full built graph minus ol_intrinsic (retinotopic vision-only neurons
    our sensors can never drive) — the deployment-sized graph."""
    with open("/home/benson/flycns/built/graph.pkl", "rb") as f:
        g = pickle.load(f)
    W, body_ids = g["W"], g["body_ids"]
    part = pd.read_csv("/home/benson/flycns/built/partition.csv").set_index("bodyId")
    sc = part.superclass.reindex(body_ids)
    keep = sc != "ol_intrinsic"
    keep = keep.fillna(True).to_numpy()
    Wp = W[keep][:, keep]
    ids_p = body_ids[keep]
    part_p = part.reset_index().iloc[np.where(keep)[0]].reset_index(drop=True)
    print(f"pruned graph: {Wp.shape[0]} nodes, {Wp.nnz:,} edges")
    return Wp, ids_p, part_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--label", default="bag")
    ap.add_argument("--vis-weight", type=float, default=0.0)
    ap.add_argument("--out", default="/tmp/flygm_gate_result.json")
    args = ap.parse_args()

    W, body_ids, part = load_pruned_graph()

    z = np.load(args.obs, allow_pickle=True)
    ts_scan, fps = z["ts_scan"], z["fps"]
    ts_img, vis_fps = z["ts_img"], z["vis_fps"]
    obs = fuse_obs(ts_scan, fps, ts_img, vis_fps, args.vis_weight)
    print(f"[{args.label}] {len(obs)} ticks, obs dim {obs.shape[1]}, "
          f"vis_weight {args.vis_weight}")

    real, wall = run_replay(W, part, body_ids, obs, args.label)
    Ws = shuffled_W(W)
    shuf, wall2 = run_replay(Ws, part, body_ids, obs, args.label + "-shufW")

    def metrics(cmds):
        c = np.asarray(cmds)
        return dict(drift_last30_lin=float(c[-300:, 0].mean()),
                    drift_last30_ang=float(c[-300:, 1].mean()),
                    ang_std_last30=float(c[-300:, 1].std()),
                    seg_chunks=len(seg_summary(c)))

    def drift(cmds):
        """|mean(last 30 s) - mean(prior 30 s)| — the settling metric."""
        c = np.asarray(cmds)
        a = c[-600:-300].mean(axis=0) if len(c) >= 600 else c[-300:].mean(axis=0)
        b = c[-300:].mean(axis=0)
        return float(np.max(np.abs(b - a)))

    def sensitivity(cmds, obs):
        """Gain profile: |dreadout| vs |dobs| — does the graph's motor
        stream respond to world changes (top decile of |dobs|) proportionally
        more than to rest noise (bottom decile)? The world-lockedness the
        'silence at rest' criterion got wrong."""
        dc = np.linalg.norm(np.diff(np.asarray(cmds), axis=0), axis=1)
        do = np.linalg.norm(np.diff(np.asarray(obs), axis=0), axis=1)
        hi, lo = np.percentile(do, [90, 50])
        return dict(
            corr=float(np.corrcoef(dc, do)[0, 1]),
            gain_hi_over_lo=float(dc[do >= hi].mean() /
                                  max(dc[do <= lo].mean(), 1e-9)),
            cmd_scale=float(dc.std()))

    sens_real = sensitivity(real, obs)
    sens_shuf = sensitivity(shuf, obs)

    res = {
        "label": args.label,
        "vis_weight": args.vis_weight,
        "ticks": len(obs),
        "wall_s_real": round(wall, 1),
        "wall_s_shuffled": round(wall2, 1),
        "real": metrics(real),
        "shuffledW": metrics(shuf),
        "seg_real": seg_summary(real),
        "seg_shuffled": seg_summary(shuf),
        "drift_real": drift(real),
        "drift_shuffled": drift(shuf),
        "sens_real": sens_real,
        "sens_shuffled": sens_shuf,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))

    if args.label == "stationary":
        print("\nGATE A (stationary): "
              f"drift real {res['drift_real']:.4f} vs shuffled "
              f"{res['drift_shuffled']:.4f} -> settling "
              f"{'PASS' if res['drift_real'] <= res['drift_shuffled'] else 'FAIL'}")
        print(f"  world-lockedness: corr(real) {sens_real['corr']:.3f} vs "
              f"shuffled {sens_shuf['corr']:.3f}; gain_hi/lo "
              f"{sens_real['gain_hi_over_lo']:.2f} vs "
              f"{sens_shuf['gain_hi_over_lo']:.2f} — "
              f"real-W superior = ongoing dynamics are world-locked, "
              f"not pathological")
    else:
        sep_real = max(abs(res["seg_real"][i + 1]["ang_mean"] -
                           res["seg_real"][i]["ang_mean"])
                       for i in range(len(res["seg_real"]) - 1))
        sep_shuf = max(abs(res["seg_shuffled"][i + 1]["ang_mean"] -
                           res["seg_shuffled"][i]["ang_mean"])
                       for i in range(len(res["seg_shuffled"]) - 1))
        ok = sep_real > sep_shuf
        print(f"\nGATE B (tour, chunk-to-chunk separation): real {sep_real:.4f} "
              f"vs shuffled {sep_shuf:.4f} -> {'PASS' if ok else 'FAIL'} "
              f"(provisional; B2 PCN cross-check pending)")


if __name__ == "__main__":
    main()
