# Memory Genome — design (build 2026-09-19)

## Why (pre-registered trigger)

Three consecutive structural FAILs closed the reactive scan→wheels
class at rooms/game ≈ 1.6 of 6.1 (holdout L3):

| run | lever | result |
|---|---|---|
| 21 | forward enforcement (w_rev 0.15) | cross ceiling 0.51, rooms 1.6 |
| 23 | translation fitness (w_net/w_spin) | exploit killed, honest cross 0.385 |
| 24 | blocked-flag inputs (obs v2) | cleanest genomes ever (coll 1.2), ceiling unmoved |

The failure mode is identical in all three: the rover reaches a dead
end, retreats, and returns to it later — it cannot represent "that
direction already failed." A 128-unit tanh recurrence *could* encode
this, and 24 runs of ES proved mutation pressure will not find that
solution. So we give the genome explicit, addressable memory with
learned write/read, CAMEMBE-style (memory and black-box evolution).

## Architecture (policy.py, mem>0)

State per env: hidden h (128) + memory m (M slots, default M=32).
One tick:

```
pre = Wx·o + Wh·h + Wm·m + bh      # memory READ -> hidden (1-tick delay)
h   = tanh(pre)
q   = [o, h]
g   = sigmoid(Wg·q + bg)           # per-slot WRITE gate, bg init -3 (closed)
v   = tanh(Wv·q + bv)              # write candidate (Wv init ×0.1)
m   = (1-g)·m + g·v                # EMA-style write
a   = tanh(Wo·h + Wom·m + bo)      # memory READ -> actions
```

* `mem=0` reproduces the current flat MLP **byte-identically** (packing
  order appends memory params only when mem>0; every saved genome
  without a `mem_slots` field means mem=0).
* Memory starts OFF: gate closed + small write init ⇒ gen-0 population
  ≈ the run-24 v2 baseline; evolution only turns memory on if it pays.
* Genome ≈ 45k params (M=32) vs 11.3k — rollout cost is dominated by
  scan physics, not the net, so gen wall-time barely moves.
* Graph-safe: elementwise/bmm only, static h/m buffers copied in-place,
  device-side clock — same contract as the current step().

Fitness recipe: rooms + w_cov 0.3 + w_coll 1.0 + w_rev 0.15
+ w_net 0.15 + w_spin 0.3 (spin pricing is mandatory post-hardware:
the loaded track stalls on pivots — live runs 5–8), obs v2 (blocked
latches, run 24), trim-rand 0.75, buildings L3, legacy gate.

## Validation chain (all before trusting a champion)

1. Unit: mem=0 genome bytes + forward equal current policy (regression).
2. Init sanity: mem>0 gen-0 fitness distribution ≈ run-24 gen-0.
3. 3-gen GPU smoke (graph mode, capture_fail=0).
4. Full run (120 gens) — PRE-REGISTERED: PASS worst-trim cross >= 0.60
   AND rooms/game >= 2.5 AND spin <= 0.30 (rooms is the point: memory
   must break the 1.6 ceiling). Cross 0.56–0.60 or rooms 2.0–2.5 =
   PARTIAL, extend +60 once. Else FAIL -> the 1.6 ceiling is not a
   memory-representability problem; investigation moves to world-model
   priors (offline) before any further on-robot training.
5. Deploy: runner numpy forward (v2 + mem), monitor publishes latched
   front/rear bools (new topic), then the usual parity/trim/dry/field
   ladder ending in the doorway test.

## Files touched

`evo/policy.py` (shapes+net+meta), `evo/evolve.py` (`--mem`, buffers,
meta), scoring tools read mem from genome meta (`deploy_pick`,
`trim_eval`, `orbit_audit`, `baselines`, `merge_pools` guard),
`evo/deploy/evo_runner.py` (numpy mirror + v2 centering), tests.
