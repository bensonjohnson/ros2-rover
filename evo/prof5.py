#!/usr/bin/env python3
"""What is the run-5 workload actually doing on the GB10? (run on Spark)

1) matmul precision flags (TF32 = the fp32 tensor-core path)
2) time split at production B: scan+gate vs scan+gate+policy
3) profiler top CUDA kernels during a full rollout
4) einsum materialization check (does expand->bmm copy the weights?)

Shares the GPU with the live run-5 chain, so ABSOLUTE times are inflated
by contention — the RATIOS and kernel names are what we read.
"""
import time

import numpy as np
import torch

from evo.arena import Arena, OBS_DIM
from evo.policy import PopulationNet, sample_population

dev = "cuda"
P, G, K, hidden, ticks = 128, 16, 4, 64, 150
B = P * G * K

print("== flags ==")
print("matmul allow_tf32 :", torch.backends.cuda.matmul.allow_tf32)
print("cudnn  allow_tf32 :", torch.backends.cudnn.allow_tf32)
try:
    print("float32_matmul_precision:", torch.get_float32_matmul_precision())
except Exception:
    pass
p = torch.cuda.get_device_properties(0)
print("device:", p.name, f"SMs={p.multi_processor_count}")

rng = np.random.default_rng(3)
thetas = torch.as_tensor(sample_population(P, OBS_DIM, hidden, rng),
                         device=dev)
tr = Arena(P, G, seed=62000, device=dev, fp16=True, merged_houses=K)
net = PopulationNet(thetas, OBS_DIM, hidden)
net.bind(P, G * K)
h = torch.zeros(P, G * K, hidden, device=dev)
tr._reset_hooks.append(h.zero_)


def policy_step(obs, prev_act):
    a, hn = net.step(obs.view(P, G * K, OBS_DIM), h)
    h.copy_(hn)
    return a.view(B, 2)


def static_step(obs, prev_act):        # scan+gate only, no matmul
    return prev_act


# warm both paths once (compile caches etc.)
tr.run_games(policy_step, ticks)
tr.run_games(static_step, ticks)
torch.cuda.synchronize()


def timed(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr.run_games(fn, ticks)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


t_sg = timed(static_step)
t_full = timed(policy_step)
print("\n== timing (eager, B=8192, ticks=150, contention-shared) ==")
print(f"scan+gate        : {t_sg:7.3f}s")
print(f"scan+gate+policy : {t_full:7.3f}s")
print(f"policy share     : {100*(t_full-t_sg)/t_full:5.1f}% of rollout")

# einsum materialization: peak alloc during ONE policy step
h_ = torch.zeros(P, G * K, hidden, device=dev)
o = torch.rand(P, G * K, OBS_DIM, device=dev)
net2 = PopulationNet(thetas, OBS_DIM, hidden)
net2.bind(P, G * K)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
for _ in range(5):
    net2.step(o, h_)
torch.cuda.synchronize()
peak = torch.cuda.memory_allocated() - base
print(f"\npolicy-step persistent alloc: {peak/2**20:.1f} MiB")
print(f"(a per-tick [B,82,64] fp32 materialization would be "
      f"{B*82*64*4/2**20:.0f} MiB x2 einsums)")

# profiler: kernel-name attribution for one full rollout
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    tr.run_games(policy_step, ticks)
    torch.cuda.synchronize()
evs = [(e.self_device_time_total, e.key) for e in prof.key_averages()
       if e.self_device_time_total > 0]
evs.sort(reverse=True)
tot = sum(t for t, _ in evs)
print(f"\n== top CUDA kernels (of {tot/1e3:.0f} ms device time) ==")
for t, k in evs[:12]:
    print(f"  {100*t/tot:5.1f}%  {k[:90]}")
