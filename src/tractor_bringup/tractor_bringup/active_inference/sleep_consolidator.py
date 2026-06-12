#!/usr/bin/env python3
"""Biological Offline Memory Consolidation (Sleep) for the PC Rover.

Processes the recorded experience dataset (~/.ros/pnn_experience.jsonl) using
purely local predictive-coding rules (no backpropagation). Consists of:
  1. Slow-Wave Sleep (SWS): Hebbian replay of real experience sequences with
     sensory noise injection and bootstrapped updates.
  2. REM Sleep: Generative dreaming rolling out counterfactual actions from
     historical states to reinforce dynamic consistency.
  3. Synaptic Homeostasis: Downscaling weights (pruning) to regularize the network.
"""

import os
import sys
import glob
import argparse
import time
import json
import numpy as np
import torch

# Ensure ROS2 package imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from active_inference.pc_world_model import PCWorldModel, PCConfig


def experience_files(log_path: str) -> list[str]:
    """The main log plus any size-rotation parts the runner split off.

    Parts are named <base>_part_<timestamp>.jsonl, so lexical order is
    chronological; the main log holds the newest experience and comes last.
    """
    base = log_path[:-len(".jsonl")] if log_path.endswith(".jsonl") else log_path
    paths = sorted(glob.glob(base + "_part_*.jsonl"))
    if os.path.exists(log_path):
        paths.append(log_path)
    return paths


def load_experience(log_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load logged experience (obs, act) lines from the JSONL file(s)."""
    dataset = []
    for path in experience_files(log_path):
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    obs = np.array(data["obs"], dtype=np.float32)
                    act = np.array(data["act"], dtype=np.float32)
                    dataset.append((obs, act))
        except Exception as e:
            print(f"Error reading experience log {path}: {e}")
    return dataset


def consolidate_slow_layer(args, model, dataset, rng, dash=None):
    """Sleep for the hierarchy's second story.

    The slow layer's sensory world IS the fast layer's latent space, and the
    fast consolidation above just shifted that space. So slow sleep first
    RE-GROUNDS: it replays the logged observations through the freshly
    consolidated fast model to regenerate the slow training sequences in the
    fast layer's NEW latent space, then runs SWS replay and member-owned REM
    dreaming on the slow model — the same recipe, one level up.
    """
    slow_path = os.path.expanduser(args.slow_model_path)
    if args.slow_sws_epochs <= 0 and args.slow_rem_epochs <= 0:
        return
    if not os.path.exists(slow_path):
        print("\nNo slow-layer checkpoint — skipping slow consolidation")
        return

    print("\n--- Slow-Layer Consolidation ---")
    try:
        ssd = torch.load(slow_path, map_location="cpu", weights_only=False)
        meta = ssd.get("meta", {})
        if meta.get("fast_latent_dim") != model.cfg.latent_dim:
            print(f"Slow layer was trained against fast latent_dim="
                  f"{meta.get('fast_latent_dim')} (now {model.cfg.latent_dim})"
                  f" — skipping")
            return
        smodel = PCWorldModel(ssd["model"]["cfg"])
        smodel.load_state_dict(ssd["model"])
    except Exception as e:
        print(f"Could not load slow layer: {e} — skipping")
        return

    period = int(meta.get("period_ticks", 15))
    D = model.cfg.latent_dim
    n_intero = int(getattr(model.cfg, "n_intero", 0))
    num_bins = model.cfg.obs_dim - int(getattr(model.cfg, "n_proprio", 0)) \
        - n_intero
    sqrt_obs = float(np.sqrt(model.cfg.obs_dim))

    # 1. Re-ground: regenerate slow observations under the NEW fast weights.
    print(f"Re-grounding: replaying {len(dataset)} steps through the "
          f"consolidated fast model (windows of {period})...")
    t0 = time.time()
    last_dash = 0.0
    total_windows = max(1, len(dataset) // period)
    # Visualization-only slow context: settles the slow model over each
    # regenerated window so the dashboard's slow story lights up live.
    zp2_vis = torch.zeros(smodel.cfg.latent_dim)
    last_slow_s = None
    slow_steps = []          # (slow_obs np, mean_action np)
    zp = torch.zeros(D)
    sum_s = torch.zeros(D)
    sum_open = sum_err = sum_nov = 0.0
    sum_act = np.zeros(2)
    n_in_window = 0
    step_i = 0
    for obs_np, act_np in dataset:
        o = torch.from_numpy(obs_np.copy())
        a = torch.from_numpy(act_np.copy())
        z, F_g, err = model.infer(o, a, z_prev=zp, n_iters=10)
        zp = z
        step_i += 1
        sum_s += torch.tanh(z)
        sum_open += float(obs_np[:num_bins].mean())
        sum_err += min(1.0, err / sqrt_obs)
        sum_nov += float(obs_np[-1]) if n_intero > 0 else 0.0
        sum_act += act_np
        n_in_window += 1
        if n_in_window >= period:
            slow_obs = np.concatenate([
                0.5 + 0.5 * (sum_s / n_in_window).numpy(),
                [sum_open / n_in_window, sum_err / n_in_window,
                 sum_nov / n_in_window],
            ]).astype(np.float32)
            slow_steps.append((slow_obs,
                               (sum_act / n_in_window).astype(np.float32)))
            if dash is not None:
                z2v, _, _ = smodel.infer(
                    torch.from_numpy(slow_obs.copy()),
                    torch.from_numpy(slow_steps[-1][1].copy()),
                    z_prev=zp2_vis)
                zp2_vis = z2v
                last_slow_s = [float(x) for x in torch.tanh(z2v)]
            sum_s.zero_(); sum_open = sum_err = sum_nov = 0.0
            sum_act[:] = 0.0; n_in_window = 0
        if dash is not None and dash.active():
            now = time.time()
            if args.step_delay > 0.0 or (now - last_dash) >= 0.033:
                last_dash = now
                s_lat = torch.tanh(z).numpy()
                dash.update(
                    obs=obs_np[:num_bins],
                    pred=model.reconstruct(z).numpy()[:num_bins],
                    F=F_g, err=err, epi=0.0, epi_max=0.0, prag=0.0,
                    L=float(act_np[0]), R=float(act_np[1]), step=step_i,
                    s=s_lat, z_abs=np.abs(s_lat),
                    mode="slow-ground",
                    epoch=len(slow_steps), epoch_total=total_windows,
                    slow_s=last_slow_s,
                    slow_window=[n_in_window, period],
                )
                if args.step_delay > 0.0:
                    time.sleep(args.step_delay)
    print(f"Regenerated {len(slow_steps)} slow steps in {time.time()-t0:.1f}s")

    L = args.seq_len
    if len(slow_steps) < L * 4:
        print("Not enough slow steps for consolidation — skipping")
        return
    sequences = [slow_steps[i:i + L]
                 for i in range(0, len(slow_steps) - L + 1, L // 2)]

    def probe_slow() -> float:
        total = 0.0
        idx = np.random.default_rng(123).integers(
            0, len(slow_steps), size=min(32, len(slow_steps)))
        for i in idx:
            o_np, a_np = slow_steps[int(i)]
            z, _, _ = smodel.infer(torch.from_numpy(o_np.copy()),
                                   torch.from_numpy(a_np.copy()),
                                   z_prev=torch.zeros(smodel.cfg.latent_dim))
            acts = torch.rand(8, smodel.cfg.action_dim) * 2.0 - 1.0
            total += float(smodel.epistemic_value(z, acts).mean())
        return total / len(idx)

    dis_before = probe_slow()
    print(f"Slow ensemble disagreement before: {dis_before:.5f}")

    # 2. SWS: replay the regenerated slow sequences (real experience, no
    # injected noise — the windowed averaging already smooths the signal).
    sws_steps = 0
    for epoch in range(args.slow_sws_epochs):
        t0 = time.time()
        err_sum = 0.0
        for idx in rng.permutation(len(sequences)):
            seq = sequences[idx]
            zp2 = torch.zeros(smodel.cfg.latent_dim)
            for t, (o_np, a_np) in enumerate(seq):
                o = torch.from_numpy(o_np.copy())
                a = torch.from_numpy(a_np.copy())
                z2, F2, err2 = smodel.infer(o, a, z_prev=zp2)
                err_sum += err2
                if t >= 4:
                    smodel.learn(z2, a, o, z_prev=zp2, advance=False,
                                 update_precision=False)
                    sws_steps += 1
                if dash is not None and dash.active():
                    now = time.time()
                    if args.step_delay > 0.0 or (now - last_dash) >= 0.033:
                        last_dash = now
                        dash.update(
                            obs=[], pred=[],
                            F=F2, err=err2, epi=0.0, epi_max=0.0, prag=0.0,
                            L=float(a_np[0]), R=float(a_np[1]),
                            step=sws_steps,
                            mode="slow-sws",
                            epoch=epoch + 1, epoch_total=args.slow_sws_epochs,
                            slow_s=[float(x) for x in torch.tanh(z2)],
                            slow_F=F2, slow_err=err2,
                            slow_dis_before=dis_before,
                        )
                        if args.step_delay > 0.0:
                            time.sleep(args.step_delay)
                zp2 = z2
        smodel.W_o *= args.decay_rate
        for m in range(smodel.cfg.ensemble_size):
            smodel.W_z[m] *= args.decay_rate
        print(f" Slow SWS {epoch+1}/{args.slow_sws_epochs} | "
              f"Avg Err: {err_sum/max(1,len(sequences)*L):.4f} | "
              f"Time: {time.time()-t0:.1f}s")

    # 3. REM: member-owned dreaming, same decoupling rule as the fast layer
    # (members never regress toward a shared target — disagreement IS the
    # slow epistemic signal).
    for epoch in range(args.slow_rem_epochs):
        t0 = time.time()
        rem_updates = 0
        for _ in range(len(sequences)):
            seq = sequences[rng.integers(len(sequences))]
            zp2 = torch.zeros(smodel.cfg.latent_dim)
            for t in range(min(4, len(seq))):
                o = torch.from_numpy(seq[t][0].copy())
                a = torch.from_numpy(seq[t][1].copy())
                zp2, _, _ = smodel.infer(o, a, z_prev=zp2)
            member = int(rng.integers(smodel.cfg.ensemble_size))
            for _ in range(L - 4):
                a_dream = torch.rand(smodel.cfg.action_dim) * 2.0 - 1.0
                s_in = smodel._trans_input(zp2, a_dream)
                z_pred = smodel._predict_member(member, s_in)
                z_noisy = z_pred + args.dream_noise * \
                    torch.randn(smodel.cfg.latent_dim)
                o_dream = smodel.reconstruct(z_noisy)
                z_next, F_d, err_d = smodel.infer(o_dream, a_dream, z_prev=zp2,
                                                  member=member)
                e_zm = z_next - z_pred
                smodel.W_z[member] += smodel.cfg.lr_trans * smodel.pi_z * \
                    torch.outer(e_zm, s_in)
                smodel.b_z[member] += smodel.cfg.lr_trans * smodel.pi_z * e_zm
                rem_updates += 1
                if dash is not None and dash.active():
                    now = time.time()
                    if args.step_delay > 0.0 or (now - last_dash) >= 0.033:
                        last_dash = now
                        dash.update(
                            obs=[], pred=[],
                            F=F_d, err=err_d, epi=0.0, epi_max=0.0, prag=0.0,
                            L=float(a_dream[0]), R=float(a_dream[1]),
                            step=rem_updates,
                            mode="slow-rem",
                            epoch=epoch + 1, epoch_total=args.slow_rem_epochs,
                            slow_s=[float(x) for x in torch.tanh(z_next)],
                            slow_F=F_d, slow_err=err_d,
                            slow_dis_before=dis_before,
                        )
                        if args.step_delay > 0.0:
                            time.sleep(args.step_delay)
                zp2 = z_next
        smodel.W_o *= args.decay_rate
        for m in range(smodel.cfg.ensemble_size):
            smodel.W_z[m] *= args.decay_rate
        print(f" Slow REM {epoch+1}/{args.slow_rem_epochs} | "
              f"Dream Steps: {rem_updates} | Time: {time.time()-t0:.1f}s")

    dis_after = probe_slow()
    print(f"Slow ensemble disagreement after: {dis_after:.5f} "
          f"({dis_after/max(dis_before,1e-12):.2f}x of pre-sleep)")

    try:
        tmp = slow_path + ".tmp"
        torch.save({"model": smodel.state_dict(), "meta": meta}, tmp)
        os.replace(tmp, slow_path)
        print("✅ Saved consolidated slow layer")
    except Exception as e:
        print(f"❌ Could not save slow layer: {e}")


def run_sleep_cycle(args):
    print("==================================================")
    print("PNN Rover - Biological Sleep Consolidation Cycle")
    print("==================================================")
    
    # Keep PyTorch single-threaded by default: the per-step ops are tiny
    # (<=72-dim matmuls), so OpenMP fork/join sync costs more than it saves.
    torch.set_num_threads(args.torch_threads)
    
    model_path = os.path.expanduser(args.model_path)
    experience_path = os.path.expanduser(args.experience_log_path)
    
    # 1. Load experience
    print(f"Loading experience log from: {experience_path}")
    dataset = load_experience(experience_path)
    n_steps = len(dataset)
    print(f"Loaded {n_steps} raw transition steps")
    if n_steps < args.seq_len * 4:
        print("Error: Not enough experience data to run sleep consolidation.")
        return
        
    # 2. Initialize World Model
    print(f"Loading current brain weights from: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Brain weight file not found. Run the rover first to initialize it.")
        return
        
    try:
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = sd["cfg"]
        model = PCWorldModel(cfg)
        model.load_state_dict(sd)
        # Carried through to the saved checkpoint untouched (the awake runner
        # warns if it ever runs the brain with a different scale).
        saved_action_scale = sd.get("action_scale")
        print(f"World model initialized: obs={cfg.obs_dim}, latent={cfg.latent_dim}, ensemble={cfg.ensemble_size}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Lidar bin count = obs minus the proprio + intero tail (old checkpoints
    # predate the n_proprio/n_intero config fields; the runner has always used
    # 8 proprio channels whenever the obs vector is wider than 72 lidar bins).
    n_proprio = int(getattr(cfg, "n_proprio", 8 if cfg.obs_dim > 72 else 0))
    n_intero = int(getattr(cfg, "n_intero", 0))
    num_bins = cfg.obs_dim - n_proprio - n_intero

    def probe_disagreement() -> float:
        """Mean ensemble disagreement over a fixed probe set — the health
        metric for the epistemic drive. Sleep must not crater this."""
        probe_rng = np.random.default_rng(123)
        idx = probe_rng.integers(0, n_steps, size=min(64, n_steps))
        total = 0.0
        for i in idx:
            obs_np, act_np = dataset[int(i)]
            z, _, _ = model.infer(torch.from_numpy(obs_np.copy()),
                                  torch.from_numpy(act_np.copy()),
                                  z_prev=torch.zeros(model.cfg.latent_dim))
            actions = torch.rand(8, model.cfg.action_dim) * 2.0 - 1.0
            total += float(model.epistemic_value(z, actions).mean())
        return total / len(idx)

    disagreement_before = probe_disagreement()
    print(f"Ensemble disagreement before sleep: {disagreement_before:.5f}")

    # Temporarily override learning rates for sleep if specified
    original_lr_obs = model.cfg.lr_obs
    original_lr_trans = model.cfg.lr_trans
    if args.lr_obs_sleep > 0.0:
        model.cfg.lr_obs = args.lr_obs_sleep
    if args.lr_trans_sleep > 0.0:
        model.cfg.lr_trans = args.lr_trans_sleep

    # Prepare sequences of length L
    L = args.seq_len
    n_seqs = n_steps - L + 1
    sequences = []
    for i in range(0, n_seqs, L // 2):  # overlap sequences for density
        if i + L <= n_steps:
            sequences.append(dataset[i : i + L])
            
    print(f"Extracted {len(sequences)} overlapping sequences of length {L}")

    # Initialize Dream Dashboard
    dash = None
    if args.visualize and args.dashboard_port > 0:
        try:
            from tractor_bringup.active_inference.pc_dashboard import (
                PCDashboardState, start_dashboard_server)
            dash = PCDashboardState()
            start_dashboard_server(dash, port=args.dashboard_port)
            print(f"Dream Dashboard active at: http://localhost:{args.dashboard_port}")
        except Exception as e:
            print(f"Could not start dream dashboard: {e}")

    last_dash_update = 0.0

    # --- SWS: Memory Replay Epochs ---
    print(f"\n--- Starting Slow-Wave Sleep (SWS): Replaying {args.sws_epochs} epochs ---")
    rng = np.random.default_rng(args.seed)
    
    for epoch in range(args.sws_epochs):
        epoch_start = time.time()
        # Shuffle sequences
        shuffled_idx = rng.permutation(len(sequences))
        sws_error = 0.0
        updates_count = 0
        
        for idx in shuffled_idx:
            seq = sequences[idx]
            zp = torch.zeros(model.cfg.latent_dim)
            
            for t in range(L):
                obs_np, act_np = seq[t]
                
                # Convert to PyTorch tensors
                o = torch.from_numpy(obs_np.copy())
                a = torch.from_numpy(act_np.copy())
                
                # Biological Mutation: Inject sensory noise into the LiDAR channels
                if args.noise_std > 0.0:
                    noise = torch.randn(num_bins) * args.noise_std
                    o[:num_bins] = torch.clamp(o[:num_bins] + noise, 0.0, 1.0)
                
                # Iterative local inference settling
                z, F, err = model.infer(o, a, z_prev=zp)
                sws_error += err
                
                # Hebbian local update (no backprop)
                # Burn-in: let the first 4 steps settle the recurrence before learning
                # update_precision=False: SWS injects artificial noise — it
                # must not be written into the learned sensor-noise ledger.
                if t >= 4:
                    model.learn(z, a, o, z_prev=zp, advance=False,
                                update_precision=False)
                    updates_count += 1

                # Update visualizer when someone is watching (rate-limited to
                # 30 Hz unless step_delay is active)
                if dash is not None and dash.active():
                    now = time.time()
                    if args.step_delay > 0.0 or (now - last_dash_update) >= 0.033:
                        last_dash_update = now
                        o_hat, s_latent = model._decode(z)
                        z_prev_in = model._trans_input(zp, a)
                        e_o = (o - o_hat).numpy()[:num_bins]
                        e_z = (z - model._prior_mean(z_prev_in)).numpy()
                        trans_errors = np.array([
                            float((z - model._predict_member(m, z_prev_in)).pow(2).mean())
                            for m in range(model.cfg.ensemble_size)
                        ])
                        epi = float(model.epistemic_value(zp, a.unsqueeze(0))[0])

                        dash.update(
                            obs=obs_np[:num_bins],
                            pred=model.reconstruct(z).numpy()[:num_bins],
                            F=F, err=err,
                            epi=epi, epi_max=epi,
                            prag=0.0,
                            L=float(act_np[0]), R=float(act_np[1]), step=updates_count,
                            s=s_latent.numpy(),
                            e_o=e_o,
                            W_o=model.W_o.detach().cpu().numpy(),
                            trans_errors=trans_errors,
                            z_abs=np.abs(s_latent.numpy()),
                            e_z_abs=np.abs(e_z),
                            mode="sws",
                            epoch=epoch + 1,
                            epoch_total=args.sws_epochs,
                            disagreement_before=disagreement_before,
                        )
                        if args.step_delay > 0.0:
                            time.sleep(args.step_delay)
                zp = z
                
        # Synaptic Homeostasis: apply synaptic decay (pruning)
        model.W_o *= args.decay_rate
        for m in range(model.cfg.ensemble_size):
            model.W_z[m] *= args.decay_rate
            
        avg_err = sws_error / max(1, len(sequences) * L)
        print(f" SWS Epoch {epoch+1}/{args.sws_epochs} | Avg Reconstruct Error: {avg_err:.4f} | Time: {time.time()-epoch_start:.1f}s")

    # --- REM Sleep: Generative Dreaming ---
    # Each dream rollout belongs to ONE randomly chosen ensemble member: it
    # dreams with its own transition prediction (plus latent noise), settles
    # against the dreamed observation using its own prior, and only its own
    # weights are updated. Members never regress toward a shared (mean) target
    # — that would collapse ensemble disagreement, which IS the awake brain's
    # epistemic exploration signal.
    print(f"\n--- Starting REM Sleep: Generative Dreaming ({args.rem_epochs} epochs) ---")

    for epoch in range(args.rem_epochs):
        epoch_start = time.time()
        rem_updates = 0

        # We will dream by starting from random historical states and rolling out random actions
        for _ in range(len(sequences)):
            # Pick a random sequence to settle the starting state
            rand_seq = sequences[rng.integers(len(sequences))]
            zp = torch.zeros(model.cfg.latent_dim)

            # 1. Settle z_0 (warm-up/burn-in)
            for t in range(min(4, len(rand_seq))):
                o = torch.from_numpy(rand_seq[t][0])
                a = torch.from_numpy(rand_seq[t][1])
                z_curr, _, _ = model.infer(o, a, z_prev=zp)
                zp = z_curr

            # 2. Dream rollout owned by a single ensemble member
            member = int(rng.integers(model.cfg.ensemble_size))
            for _ in range(L - 4):
                # Generate random exploratory action
                a_dream = (torch.rand(model.cfg.action_dim) * 2.0 - 1.0)

                # Member's own next-state prediction, perturbed in latent space
                # so the dream explores around (not exactly on) its prediction.
                s_in = model._trans_input(zp, a_dream)
                z_pred = model._predict_member(member, s_in)
                z_noisy = z_pred + args.dream_noise * torch.randn(model.cfg.latent_dim)

                # Reconstruct expected observation (hallucination)
                o_dream = model.reconstruct(z_noisy)

                # Settle against the dreamed observation with the member's own
                # prior, then pull only this member's transition toward the
                # settled (decoder-consistent) state.
                z_next, F_dream, err_dream = model.infer(o_dream, a_dream, z_prev=zp, member=member)
                e_zm = z_next - z_pred
                model.W_z[member] += model.cfg.lr_trans * model.pi_z * torch.outer(e_zm, s_in)
                model.b_z[member] += model.cfg.lr_trans * model.pi_z * e_zm
                rem_updates += 1

                # Update visualizer when someone is watching (rate-limited to
                # 30 Hz unless step_delay is active)
                if dash is not None and dash.active():
                    now = time.time()
                    if args.step_delay > 0.0 or (now - last_dash_update) >= 0.033:
                        last_dash_update = now
                        o_hat, s_latent = model._decode(z_next)
                        z_prev_in = model._trans_input(zp, a_dream)
                        e_o = (o_dream - o_hat).numpy()[:num_bins]
                        trans_errors = np.array([
                            float((z_next - model._predict_member(m, z_prev_in)).pow(2).mean())
                            for m in range(model.cfg.ensemble_size)
                        ])
                        epi = float(model.epistemic_value(zp, a_dream.unsqueeze(0))[0])

                        dash.update(
                            obs=o_dream.numpy()[:num_bins],
                            pred=o_hat.numpy()[:num_bins],
                            F=F_dream, err=err_dream,
                            epi=epi, epi_max=epi,
                            prag=0.0,
                            L=float(a_dream[0]), R=float(a_dream[1]), step=rem_updates,
                            s=s_latent.numpy(),
                            e_o=e_o,
                            W_o=model.W_o.detach().cpu().numpy(),
                            trans_errors=trans_errors,
                            z_abs=np.abs(s_latent.numpy()),
                            e_z_abs=np.abs(e_zm.numpy()),
                            mode="rem",
                            epoch=epoch + 1,
                            epoch_total=args.rem_epochs,
                            disagreement_before=disagreement_before,
                        )
                        if args.step_delay > 0.0:
                            time.sleep(args.step_delay)

                zp = z_next
                
        # Apply synaptic decay after dreaming too
        model.W_o *= args.decay_rate
        for m in range(model.cfg.ensemble_size):
            model.W_z[m] *= args.decay_rate
            
        print(f" REM Epoch {epoch+1}/{args.rem_epochs} | Dream Steps Consolidated: {rem_updates} | Time: {time.time()-epoch_start:.1f}s")

    # Restore original configuration learning rates
    model.cfg.lr_obs = original_lr_obs
    model.cfg.lr_trans = original_lr_trans

    disagreement_after = probe_disagreement()
    ratio = disagreement_after / max(disagreement_before, 1e-12)
    print(f"Ensemble disagreement after sleep: {disagreement_after:.5f} "
          f"({ratio:.2f}x of pre-sleep)")
    if ratio < 0.5:
        print("⚠️ WARNING: sleep more than halved ensemble disagreement — the "
              "epistemic (curiosity) signal is weakening. Consider fewer REM "
              "epochs or higher --dream_noise.")

    # 3. Save consolidated weights
    print(f"\nSaving consolidated weights back to {model_path}...")
    try:
        # Write-then-rename so a crash mid-save can't corrupt the brain.
        tmp_path = model_path + ".tmp"
        out_sd = model.state_dict()
        if saved_action_scale is not None:
            out_sd["action_scale"] = saved_action_scale
        torch.save(out_sd, tmp_path)
        os.replace(tmp_path, model_path)
        print("✅ Saved consolidated brain weights successfully!")
    except Exception as e:
        print(f"❌ Could not save brain weights: {e}")
        return

    # 3b. Slow-layer sleep: re-ground on the NEW fast weights, then SWS + REM
    # one level up. Must run before the logs are archived — re-grounding
    # replays them.
    try:
        consolidate_slow_layer(args, model, dataset, rng, dash=dash)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Slow-layer consolidation failed: {e}")

    # 4. Archive the consolidated experience log(s), rotation parts included
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = experience_path[:-len(".jsonl")] if experience_path.endswith(".jsonl") else experience_path
    try:
        # Archive names must NOT contain "_part_" or the next sleep cycle's
        # glob would re-load already-consolidated experience.
        for i, path in enumerate(experience_files(experience_path)):
            archive_path = f"{base}_done_{timestamp}_{i:02d}.jsonl"
            os.rename(path, archive_path)
            print(f"Archived {path} -> {archive_path}")
        print("✅ Experience log archived. Fresh log will begin next session.")
    except Exception as e:
        print(f"⚠️ Could not archive experience log: {e}")

    # Keep dashboard alive for review if visualize is enabled (skipped under
    # the brain supervisor, which serves the page itself and wants the child
    # to exit so the rover returns to idle).
    if dash is not None and not args.exit_when_done:
        print("\n✅ Sleep cycle completed! Keeping dashboard alive. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting.")


def main(args=None):
    parser = argparse.ArgumentParser(description="Biological PC Sleep Consolidator")
    parser.add_argument("--model_path", type=str, default="~/.ros/pnn_brain.pt")
    parser.add_argument("--slow_model_path", type=str, default="~/.ros/pnn_brain_slow.pt")
    parser.add_argument("--slow_sws_epochs", type=int, default=3, help="Slow-layer SWS epochs (0 disables slow sleep)")
    parser.add_argument("--slow_rem_epochs", type=int, default=2, help="Slow-layer REM epochs")
    parser.add_argument("--experience_log_path", type=str, default="~/.ros/pnn_experience.jsonl")
    parser.add_argument("--sws_epochs", type=int, default=5, help="Slow-Wave Sleep epochs (replay)")
    parser.add_argument("--rem_epochs", type=int, default=3, help="REM Sleep epochs (dreaming)")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence window length")
    parser.add_argument("--noise_std", type=float, default=0.02, help="Lidar sensory noise amplitude")
    parser.add_argument("--dream_noise", type=float, default=0.05, help="Latent noise amplitude during REM dream rollouts")
    parser.add_argument("--decay_rate", type=float, default=0.999, help="Synaptic homeostasis decay multiplier")
    parser.add_argument("--lr_obs_sleep", type=float, default=0.02, help="Observation learning rate during sleep")
    parser.add_argument("--lr_trans_sleep", type=float, default=0.01, help="Transition learning rate during sleep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", default=True, help="Stream dreaming steps to web dashboard")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", help="Disable web dashboard streaming")
    parser.add_argument("--dashboard_port", type=int, default=8082, help="Web dashboard port")
    parser.add_argument("--step_delay", type=float, default=0.0, help="Time delay (seconds) between replay/dreaming steps to regulate animation speed")
    parser.add_argument("--exit_when_done", action="store_true",
                        help="Exit after the cycle instead of keeping the dashboard alive "
                             "(used by brain_supervisor, which owns the public page)")
    parser.add_argument("--torch_threads", type=int, default=1, help="Number of CPU threads for PyTorch ops (default: 1; tensors here are tiny (<=72-dim) so multi-threaded BLAS adds more sync overhead than it saves)")
    
    parsed_args = parser.parse_args(args)
    run_sleep_cycle(parsed_args)


if __name__ == "__main__":
    main()
