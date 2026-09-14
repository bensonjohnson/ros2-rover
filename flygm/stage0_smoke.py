#!/usr/bin/env python3
"""Stage-0 smoke: schemas + partition counts for MaleCNS v1.0 connectome files.

Run on the DGX Spark:  /home/benson/venv/bin/python flygm/stage0_smoke.py
Verifies the three feathers actually parse and that super-class annotations give
the afferent/efferent/intrinsic partition FlyGM needs. No robot, no ROS.
"""
import time

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

D = "/home/benson/flycns"

WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"
ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def schema_of(path):
    # schema-only read: footers only, does not materialize 500 MB of columns
    try:
        with ipc.open_file(path) as f:
            return f.schema
    except Exception:
        import pyarrow.feather as feather
        return feather.read_table(path).schema


for name in (WEIGHTS, NT, ANN):
    t0 = time.time()
    print(f"\n===== {name}")
    print(schema_of(f"{D}/{name}"))
    print(f"(schema read {time.time() - t0:.1f}s)")

ann = pd.read_feather(f"{D}/{ANN}")
print("\n===== annotations:", ann.shape)
print(ann.columns.tolist())
for col in [c for c in ann.columns
            if c in ("class", "super_class", "group", "cell_type", "type",
                     "side", "hemilineage", "flow", "subclass")]:
    vc = ann[col].value_counts(dropna=False)
    print(f"\n-- {col} ({len(vc)} unique):")
    print(vc.head(25).to_string())

nt = pd.read_feather(f"{D}/{NT}")
print("\n===== neurotransmitters:", nt.shape)
print(nt.columns.tolist())
print(nt.head(3).to_string())
