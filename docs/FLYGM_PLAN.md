# FLYGM Tier-2 Spike — Fly Connectome as Rover Policy Graph

**Status: LIVE WIRED & TESTED (2026-09-14).** Stage 0 done (graph built,
spot-checks pass). Stage 1: bag-replay gate + 30 s LIVE run on the rover.
Results below; stage 2 (training) is the open gate. NOT a new long-running
stack — verdict-gated per the original scoping.

## Stage-1 results (2026-09-14, real bags + live rover)
- Gate A bag replay (stationary dock, 668 ticks): "silence at rest" FAILED —
  real-W readout drift 0.231 vs shuffled-W 0.081 (real wiring sustains
  ongoing internal dynamics). World-lockedness: corr(dcmd,dobs) +0.040 real
  vs -0.033 shuffled; gain hi/lo 1.18 vs 0.98 — real-W marginally
  world-locked. Interpretation: wiring gives a small real edge but NO free
  controller; FlyGM's paper agrees — the bias pays off through training.
  NOTE: first shuffled-control run was broken (no-op shuffle → identical
  outputs to 7 decimals); only trust the asserted >90% rewired runs.
- Live wiring: `tractor_bringup/fly_brain_node.py` publishes /track_cmd_ai
  (Float32MultiArray [L,R]) exactly like the PCN runner → safety monitor →
  motor driver. HARDSTOP latch honored (mode=3 zeroed 1054/1354 msgs).
  Xbox deadman override wired (/cmd_vel_teleop, 5-tick persist, graph keeps
  observing during override).
- Live deploy artifact = pruned graph: ol_intrinsic dropped, then edges
  <5 syn dropped (25% of edges, ~73% of synapse weight), 72,900 nodes /
  3.48M edges / C=4 → **38 ms/tick single-thread on RK3588 A76** (C=32 full
  graph is 226 ms — torch-CPU-sparse overhead; scipy CSR is the live path).
- 30 s live run (HARDSTOP released 04:32:08–04:32:38 UTC): 300/300 fly
  ticks, zero stale, no obstacle within 0.15 m (safety never fired).
  Behavior: UNTRAINED readout saturates both rails (L +0.567±0.06, R
  -0.598±0.03 ≈ ±cap) → persistent spin-in-place, v≈0, w=-0.13±0.24,
  0 direction flips, corr(|dcmd|,|dmin_scan|) −0.03. Degenerate fixed
  action = expected pre-training pathology (FlyGM fixed it with imitation
  pretraining). Bag: rover:/tmp/flygm_livetest.
- Conclusion: the loop is safe, realtime, and wired end-to-end; behavior is
  degenerate-but-bounded until stage-2 training (IL on teleop + PPO on the
  Spark, decoder R + encoder enc_P + eta trainable, W frozen).

## Verified data access (2026-09-13)

Public bucket `gs://flyem-male-cns` (no auth needed, plain HTTPS works). v1.0 files at
`v1.0/connectome-data/flat-connectome/`, Apache Feather, read with pyarrow/pandas:

| file | size | what |
|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.05 GB | full directed graph, per-pair synapse counts |
| `...-significant-only.feather` | 502 MB | pruned variant (≥5 syn per pair per blendi: ~6.3M edges carry ~90M syn) |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43 MB | 10-NT predictions per neuron (ternary) |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14 MB | class / super-class / type / side → the afferent/efferent/intrinsic partition |

Landing page: https://male-cns.janelia.org/download — Berg et al., *Cell*, Sept 2026.
Connectome derivatives: CC BY 4.0 with attribution (as done by fly-brain-minecraft).

Reference implementations found:
- **FlyGM** (paper only, no public code) — our spec is the paper's §3.
- **blendi-remade/fly-brain-minecraft** — runnable; LIF dynamics per Shiu et al. 2024,
  connectome loader + encoder/decoder + benches, MIT code, CC-BY-4.0 data derivative.
- **fly-craftax** (liuzihe02) — PPO training of a descending-neuron readout on a
  connectome graph + baselines + replay viewer. Closest training-loop reference.
- **neurocraft-fly-public** — NOT runnable (project materials only, release pending).

## Architecture (FlyGM recipe, adapted)

- Signed weight matrix: `W[v,u] = N_exc(u,v) − N_inh(u,v)`; excitatory = {ACH, GLU,
  ASP, HIS}, inhibitory = {GABA, GLY}. Modulatory NTs (dopamine, octopamine, serotonin,
  tyramine) → separate optional plasticity channel, default off.
- Partition from super-class annotations: afferent = sensory + optic; efferent =
  motor + descending (+ ascending as bridge); intrinsic = central + optic rest.
- Per step (C = 32 latent channels, per FlyGM):
  1. Encoder Enc_θ: obs (lidar 360 bins, vis fingerprint 23-d, proprio 6·K, novelty) →
     32-dim, injected to all afferents via a gating map (tanh).
  2. `M_t = W H_t` (sparse CSR; frozen = the connectome).
  3. `H_t+1[v] = f_ψ([M_t[v] ∥ η_v])` — shared small MLP, per-neuron trainable
     intrinsic descriptor η_v ∈ R^32.
  4. Decoder Dec_φ: efferent states → linear + angular velocity (track_cmd range).
- Train: imitation on teleop/expert-policy trajectories (KL + annealed MSE), then PPO
  with an MLP critic on the Spark. Frozen W; trainable = encoder, η_v, f_ψ, decoder.

## Compute budget (measured arithmetic)

- Full W as CSR fp32+int32: ~1.0 GB; forward pass ≈ 2·C·|E| = 8 GFLOP/step @ C=32
  → 80 GFLOPS @ 10 Hz. Spark: trivial. Rover RK3588 Mali: memory-bandwidth-bound
  (~1 GB read/step) — use the **significant-only graph** (~50 MB CSR → CPU-realtime)
  or fp16 W / C=8, or run the brain on the Spark over wifi (the PCN HAL already
  decouples ROS from the brain, so this is a config change, not surgery).

## Stages & gates

- **Stage 0 — data + graph build (offline, this box):** download 3 feathers (~1.6 GB),
  build signed CSR W + partition tables + intrinsic descriptors; report edge/synapse
  histograms. Gate: every neuron id resolvable, W shape matches annotations, spot-check
  known circuits (CX ring attractor connectivity present; MB KCs → MBONs).
- **Stage 1 — offline gates on real bags (no robot):** replay field bags through the
  graph exactly like bag_place_replay: inject lidar/vis fingerprints as obs, record
  efferent readouts. Gate A (stationary dock bag): readout drift < threshold with
  frozen weights. Gate B (carried tour): efferent stream responds to corridor→room
  transitions at medians better than a shuffled-W control (the FlyGM baseline logic,
  on OUR data). No gate pass → no hardware.
- **Stage 2 — train on Spark in the sim adapter:** IL on teleop trajectories then PPO
  in the pnn_sim sim/house environment with the same obs pipeline. Gate: 3 waypoints
  or novelty-objective progress equivalent to the PCN pet gates, better than
  degree-preserving-rewired control (FlyGM's own ablation, replicated once).
- **Stage 3 — on-rover shadow mode:** graph runs AWAKE, publishes telemetry +
  proposed cmd but downstream rate-limited/zero (same pattern as place_vis_weight=0.0),
  compare proposed vs teleop commands; only then supervised autonomy.
- **Stage 4 (optional, composite):** CX module as goal-steering prior + MB/PCN place
  memory fusion — the tier-1/tier-2 hybrid, only if stages 0–3 pass.

## Risks (honest list)

1. Efferent identity: FlyWire flow labels tell us the SET of motor/descending neurons,
   not which are "left wheel" — decoder training must find this, and poorly-chosen
   readout neurons = the Minecraft-hack failure mode we're explicitly avoiding
   (that's why stage 2 trains the decoder instead of hand-picking).
2. Sign errors in NT table propagate as wrong polarity — verify against literature-
   known polarities for CX/MB circuits during stage 0.
3. Sim-to-real gap on a body the fly never had (wheels vs legs) — IL anchor essential;
   the connectome's inductive bias claim was only proven on the fly's own body plan.
4. Bandwidth-bound full graph on the rover GPU (see budget) — mitigations listed.
5. The big one: this could eat 4–8 weeks and become yet another abandoned family.
   Time-boxed verdict at end of stage 2. If it fails, archive behind a tag like the
   others and the PCN line is untouched.
