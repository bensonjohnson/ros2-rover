#!/usr/bin/env python3
"""How much could tensor cores buy the policy matmuls? Time net.step x N
with (a) fp32/default (TF32 off = CUDA-core SIMT) and (b) TF32 on
(tensor cores). Shares GPU with live run — compare A vs B, not absolutes.
"""
import time

import numpy as np
import torch

from evo.arena import OBS_DIM
from evo.policy import PopulationNet, sample_population

dev = "cuda"
P, G, hidden, N = 128, 64, 64, 400          # 400 policy steps ~ 1 game
rng = np.random.default_rng(3)
thetas = torch.as_tensor(sample_population(P, OBS_DIM, hidden, rng),
                         device=dev)
net = PopulationNet(thetas, OBS_DIM, hidden)
net.bind(P, G)
o = torch.rand(P, G, OBS_DIM, device=dev)
h = torch.zeros(P, G, hidden, device=dev)


def bench(tag):
    for _ in range(20):
        net.step(o, h)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        net.step(o, h)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops = 2 * P * G * (OBS_DIM * hidden + hidden * hidden + hidden * 2)
    print(f"{tag:34s} {dt/N*1e3:6.3f} ms/step   {flops/(dt/N)/1e9:7.1f} GFLOP/s")


torch.backends.cuda.matmul.allow_tf32 = False
bench("fp32 SIMT (allow_tf32=False)")
torch.backends.cuda.matmul.allow_tf32 = True
bench("TF32 tensor cores (allow=True)")
torch.backends.cuda.matmul.allow_tf32 = False
bench("fp32 SIMT again (control)")

# and the headroom question: same einsum shape as a plain GEMM, batched
w = torch.randn(P * G, OBS_DIM, hidden, device=dev, dtype=torch.half)
x = torch.randn(P * G, 1, OBS_DIM, device=dev, dtype=torch.half)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    torch.bmm(x, w)
torch.cuda.synchronize()
print(f"\nfp16 bmm same shape (tensor cores): {(time.perf_counter()-t0)/N*1e3:.3f} ms/step")
