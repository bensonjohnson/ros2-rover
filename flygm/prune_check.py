import pickle
import numpy as np
import pandas as pd

g = pickle.load(open("/home/benson/flycns/built/graph.pkl", "rb"))
W, ids = g["W"], g["body_ids"]
print("W shape:", W.shape, "nnz:", W.nnz, "ids len:", len(ids))

part = pd.read_csv("/home/benson/flycns/built/partition.csv").set_index("bodyId")
sc = part.superclass.reindex(ids)
keep = sc != "ol_intrinsic"      # drop retinotopic optic-lobe intrinsic: unstimulatable by our sensors
keep = keep.fillna(True).to_numpy()
print("kept nodes:", int(keep.sum()), "of", len(ids))

Wi = W[keep][:, keep]
print("kept edges nnz:", Wi.nnz, f"({Wi.nnz / int(keep.sum()):.1f}/node)")

surv = part.reindex(ids[keep])
for cls in ("CX", "Kenyon_Cell", "MBON", "DAN"):
    n = int((surv["class"] == cls).sum())
    print(f"  {cls} surviving: {n}")

for C in (8, 16):
    traf = Wi.nnz * (8 + 4 * C) / 1e9   # per nnz: weight+idx 8B + C fp32 H reads
    print(f"  C={C}: ~{traf:.2f} GB/tick -> RK3588 CPU at 15-25 GB/s eff: "
          f"{traf / 25e9 * 1e3:.0f}-{traf / 15e9 * 1e3:.0f} ms/tick")

print(f"dense fp16 W would be {ids.size * ids.size * 2 / 1e9:.1f} GB")
print(f"H fp32 C=8: {ids.size * 8 * 4 / 1e6:.1f} MB")
