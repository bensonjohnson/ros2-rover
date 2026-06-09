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
import argparse
import time
import json
import numpy as np
import torch

# Ensure ROS2 package imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from active_inference.pc_world_model import PCWorldModel, PCConfig


def load_experience(log_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load logged experience (obs, act) lines from JSONL file."""
    if not os.path.exists(log_path):
        return []
    
    dataset = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                obs = np.array(data["obs"], dtype=np.float32)
                act = np.array(data["act"], dtype=np.float32)
                dataset.append((obs, act))
    except Exception as e:
        print(f"Error reading experience log: {e}")
    return dataset


def run_sleep_cycle(args):
    print("==================================================")
    print("PNN Rover - Biological Sleep Consolidation Cycle")
    print("==================================================")
    
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
        print(f"World model initialized: obs={cfg.obs_dim}, latent={cfg.latent_dim}, ensemble={cfg.ensemble_size}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

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
                
                # Biological Mutation: Inject sensory noise into LiDAR channels (first 72 bins)
                if args.noise_std > 0.0:
                    noise = torch.randn(72) * args.noise_std
                    o[:72] = torch.clamp(o[:72] + noise, 0.0, 1.0)
                
                # Iterative local inference settling
                z, F, err = model.infer(o, a, z_prev=zp)
                sws_error += err
                
                # Hebbian local update (no backprop)
                # Burn-in: let the first 4 steps settle the recurrence before learning
                if t >= 4:
                    model.learn(z, a, o, z_prev=zp, advance=False)
                    updates_count += 1
                zp = z
                
        # Synaptic Homeostasis: apply synaptic decay (pruning)
        model.W_o *= args.decay_rate
        for m in range(model.cfg.ensemble_size):
            model.W_z[m] *= args.decay_rate
            
        avg_err = sws_error / max(1, len(sequences) * L)
        print(f" SWS Epoch {epoch+1}/{args.sws_epochs} | Avg Reconstruct Error: {avg_err:.4f} | Time: {time.time()-epoch_start:.1f}s")

    # --- REM Sleep: Generative Dreaming ---
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
                
            # 2. Dream rollout: generate imaginary actions and predict states
            for _ in range(L - 4):
                # Generate random exploratory action
                a_dream = (torch.rand(model.cfg.action_dim) * 2.0 - 1.0)
                
                # Project next state through the prior transition ensemble
                s_in = model._trans_input(zp, a_dream)
                preds = torch.stack([model._predict_member(m, s_in)
                                     for m in range(model.cfg.ensemble_size)])
                z_next = preds.mean(dim=0)
                
                # Reconstruct expected observation (hallucination)
                o_dream = model.reconstruct(z_next)
                
                # PC Learning on dreamed transitions to enforce self-consistency
                model.learn(z_next, a_dream, o_dream, z_prev=zp, advance=False)
                rem_updates += 1
                
                zp = z_next
                
        # Apply synaptic decay after dreaming too
        model.W_o *= args.decay_rate
        for m in range(model.cfg.ensemble_size):
            model.W_z[m] *= args.decay_rate
            
        print(f" REM Epoch {epoch+1}/{args.rem_epochs} | Dream Steps Consolidated: {rem_updates} | Time: {time.time()-epoch_start:.1f}s")

    # Restore original configuration learning rates
    model.cfg.lr_obs = original_lr_obs
    model.cfg.lr_trans = original_lr_trans

    # 3. Save consolidated weights
    print(f"\nSaving consolidated weights back to {model_path}...")
    try:
        torch.save(model.state_dict(), model_path)
        print("✅ Saved consolidated brain weights successfully!")
    except Exception as e:
        print(f"❌ Could not save brain weights: {e}")
        return

    # 4. Archive old experience log
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = experience_path.replace(".jsonl", f"_done_{timestamp}.jsonl")
    print(f"Archiving experience log to: {archive_path}")
    try:
        os.rename(experience_path, archive_path)
        print("✅ Experience log archived. Fresh log will begin next session.")
    except Exception as e:
        print(f"⚠️ Could not archive experience log: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Biological PC Sleep Consolidator")
    parser.add_argument("--model_path", type=str, default="~/.ros/pnn_brain.pt")
    parser.add_argument("--experience_log_path", type=str, default="~/.ros/pnn_experience.jsonl")
    parser.add_argument("--sws_epochs", type=int, default=5, help="Slow-Wave Sleep epochs (replay)")
    parser.add_argument("--rem_epochs", type=int, default=3, help="REM Sleep epochs (dreaming)")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence window length")
    parser.add_argument("--noise_std", type=float, default=0.02, help="Lidar sensory noise amplitude")
    parser.add_argument("--decay_rate", type=float, default=0.999, help="Synaptic homeostasis decay multiplier")
    parser.add_argument("--lr_obs_sleep", type=float, default=0.02, help="Observation learning rate during sleep")
    parser.add_argument("--lr_trans_sleep", type=float, default=0.01, help="Transition learning rate during sleep")
    parser.add_argument("--seed", type=int, default=42)
    
    parsed_args = parser.parse_args(args)
    run_sleep_cycle(parsed_args)


if __name__ == "__main__":
    main()
