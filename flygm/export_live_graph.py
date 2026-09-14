#!/usr/bin/env python3
"""Export the LIVE fly-graph artifact for the rover (FlyGM tier-2, live test).

Two-stage prune, both data-driven:
  1. drop ol_intrinsic (retinotopic vision-only stack our sensors cannot
     honestly drive),
  2. drop edges with < 5 synapses (blendi cutoff — keeps ~73% of synapse
     weight at ~25% of edges) and then any node left edge-less.
Index remapping is baked in here so the rover node loads one clean npz.

  /home/benson/venv/bin/python ~/flygm/export_live_graph.py /tmp/flygm_live.npz
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

C = 4          # live budget: 38 ms/tick measured on RK3588 A76 (rover)
MIN_SYN = 5    # significant-only already used >=1; deploy artifact uses >=5


def main(out_path="/tmp/flygm_live.npz"):
    with open("/home/benson/flycns/built/graph.pkl", "rb") as f:
        g = pickle.load(f)
    W, body_ids = g["W"], g["body_ids"]
    part = pd.read_csv("/home/benson/flycns/built/partition.csv").set_index("bodyId")

    sc = part.superclass.reindex(body_ids)
    keep1 = (sc != "ol_intrinsic").fillna(True).to_numpy()
    W1 = W[keep1][:, keep1].tocsr()
    part1 = part.reset_index().iloc[np.where(keep1)[0]].reset_index(drop=True)

    # prune weak edges, then nodes that lost all edges
    W2 = W1.copy()
    W2.data = np.where(np.abs(W2.data) >= MIN_SYN, W2.data, 0).astype(np.float32)
    W2.eliminate_zeros()
    deg = np.asarray((W2 != 0).sum(axis=1)).ravel() + \
        np.asarray((W2 != 0).sum(axis=0)).ravel()
    keep2 = deg > 0
    Wp = W2[keep2][:, keep2].tocsr()
    ids_p = body_ids[keep1][keep2]
    part_p = part1.iloc[np.where(keep2)[0]].reset_index(drop=True)
    cum = np.cumsum(keep2)

    alive_pre = np.asarray((Wp != 0).sum(axis=0)).ravel() > 0
    aff_old = np.where((part_p.role == "afferent").to_numpy() & alive_pre)[0]
    eff_old = np.where(np.isin(ids_p,
                       part_p[part_p.role == "efferent"].bodyId.to_numpy()))[0]
    aff_idx = (cum[aff_old] - 1).astype(np.int64)
    eff_idx = (cum[eff_old] - 1).astype(np.int64)

    n_obs = 72 + 23
    enc_P = (np.random.default_rng(3).standard_normal((n_obs, C))
             / np.sqrt(n_obs)).astype(np.float32)
    R = (0.1 * np.random.default_rng(7).standard_normal((2, len(eff_idx)))
         ).astype(np.float32)

    np.savez(out_path,
             W_data=Wp.data, W_indices=Wp.indices, W_indptr=Wp.indptr,
             W_shape=np.array(Wp.shape),
             body_ids=ids_p, aff_idx=aff_idx, eff_idx=eff_idx,
             enc_P=enc_P, R=R, C=np.array([C]))
    meta = dict(nodes=int(Wp.shape[0]), nnz=int(Wp.nnz),
                n_aff=len(aff_idx), n_eff=len(eff_idx), C=C,
                min_syn=MIN_SYN)
    print(json.dumps(meta), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/flygm_live.npz")
