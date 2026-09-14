#!/usr/bin/env python3
"""Stage-0 build: signed sparse graph W + partition tables from MaleCNS v1.0.

FlyGM recipe (arXiv 2602.17997 §3.1): W[v,u] = N_exc(u,v) - N_inh(u,v).
Excitatory = {acetylcholine, glutamate, histamine}; inhibitory = {gaba}.
(FlyGM also lists aspartate — absent from this NT table's labels; modulatory
NTs dopamine/octopamine/serotonin are excluded from the sign and left for an
optional plasticity channel.) Neurons kept = annotated, not fragment/tbc,
with >=1 significant edge. Outputs (in ~/flycns/built/):
  graph_meta.json      — counts, partitions, spot-check results
  partition.csv        — bodyId -> role, class, type, nt, sign sums
  W (scipy sparse) saved via pickle as csr_matrix fp32 + node index table
Run on the DGX Spark: /home/benson/venv/bin/python ~/flygm/build_graph.py
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse

D = "/home/benson/flycns"
OUT = f"{D}/built"
os.makedirs(OUT, exist_ok=True)

EXC = {"acetylcholine", "glutamate", "histamine"}
INH = {"gaba"}

t0 = time.time()
ann = pd.read_feather(f"{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather")
# Keep real neurons: annotated, drop to-be-checked / fragment status labels.
ann = ann[ann["superclass"].notna()].copy()
ann = ann[~ann["superclass"].str.endswith("_tbc", na=False)]
keep_ids = set(ann["bodyId"])
print(f"[{time.time()-t0:5.1f}s] annotations kept: {len(ann):,} bodies")

nt = pd.read_feather(f"{D}/body-neurotransmitters-male-cns-v1.0.feather")
nt = nt.drop_duplicates("body").set_index("body")["predicted_nt"]
nt = nt[nt.isin(EXC | INH)]
print(f"[{time.time()-t0:5.1f}s] NT-typed bodies: {len(nt):,}")

w = pd.read_feather(
    f"{D}/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather")
w = w[w["body_pre"].isin(keep_ids) & w["body_post"].isin(keep_ids)]
print(f"[{time.time()-t0:5.1f}s] edges among kept bodies: {len(w):,} "
      f"({w.weight.sum():,} synapses)")

# nodes = bodies that carry at least one significant edge OR have NT + class
core = sorted(set(w.body_pre) | set(w.body_post))
node_ix = {b: i for i, b in enumerate(core)}
N = len(core)
print(f"[{time.time()-t0:5.1f}s] graph nodes: {N:,}")

pre = w.body_pre.map(node_ix).to_numpy()
post = w.body_post.map(node_ix).to_numpy()
is_exc = w.body_pre.map(nt).isin(EXC).to_numpy()
sign = np.where(is_exc, 1.0, -1.0).astype(np.float32)
vals = (w.weight.to_numpy() * sign)

W = sparse.csr_matrix((vals, (post, pre)), shape=(N, N), dtype=np.float32)
print(f"[{time.time()-t0:5.1f}s] W built: {W.nnz:,} nnz, "
      f"{(W.data > 0).mean():.1%} excitatory-weighted, "
      f"{W.data.nbytes/1e9:.2f} GB data")

# ---- partition table ----
ann_i = ann.set_index("bodyId")
role = {}
AFF = {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending",
       "sensory_descending", "visual_projection", "visual_centrifugal"}
EFF = {"vnc_motor", "cb_motor", "vnc_efferent", "descending_neuron"}
part = pd.DataFrame({"bodyId": core})
part["i"] = part.bodyId.map(node_ix)
part["superclass"] = part.bodyId.map(ann_i.superclass)
part["class"] = part.bodyId.map(ann_i["class"])
part["type"] = part.bodyId.map(ann_i.type)
part["nt"] = part.bodyId.map(nt)
part["role"] = part.superclass.map(
    lambda s: "afferent" if s in AFF else
              "efferent" if s in EFF else
              "bridge" if s in ("ascending_neuron",) else
              "intrinsic" if pd.notna(s) else "unknown")
part.to_csv(f"{OUT}/partition.csv", index=False)
role_counts = part.role.value_counts().to_dict()
print(f"[{time.time()-t0:5.1f}s] roles: {role_counts}")

# ---- spot checks on known circuits ----
cls = part.set_index("i")["class"]
cx_ids = part.i[part["class"] == "CX"].to_numpy()
kc_ids = part.i[part["class"] == "Kenyon_Cell"].to_numpy()
mbon_ids = part.i[part["class"] == "MBON"].to_numpy()

Wc = W[cx_ids][:, cx_ids]
cx_edges = int((Wc != 0).sum())
kc_to_mbon = W[mbon_ids][:, kc_ids]
kc_mbon_edges = int((kc_to_mbon != 0).sum())
kc_out = (W[kc_ids] != 0).sum(axis=1)
kc_convergence = float(np.asarray(kc_out).ravel().astype(np.float64).mean())

spot = {
    "CX_neurons": len(cx_ids), "CX_internal_edges": cx_edges,
    "CX_internal_synapses": int(abs(Wc.data).sum()),
    "KC_neurons": len(kc_ids), "MBON_neurons": len(mbon_ids),
    "KC_to_MBON_edges": kc_mbon_edges,
    "KC_median_out_degree": kc_convergence,
}
print("SPOT CHECKS:", json.dumps(spot, indent=1))

meta = {
    "dataset": "MaleCNS v1.0 significant-only (Janelia/Google, CC BY 4.0)",
    "n_nodes": N, "n_edges_raw": int(len(w)),
    "n_synapses": int(w.weight.sum()),
    "nnz_after_sign_collapse": int(W.nnz),
    "roles": role_counts,
    "spot_checks": spot,
    "exc_set": sorted(EXC), "inh_set": sorted(INH),
}
with open(f"{OUT}/graph_meta.json", "w") as f:
    json.dump(meta, f, indent=1)
with open(f"{OUT}/graph.pkl", "wb") as f:
    pickle.dump({"W": W, "body_ids": np.array(core)}, f, protocol=4)
print(f"[{time.time()-t0:5.1f}s] saved -> {OUT}/graph.pkl, partition.csv, graph_meta.json")
