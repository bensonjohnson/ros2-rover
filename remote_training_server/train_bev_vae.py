#!/usr/bin/env python3
"""BEV Autoencoder Pre-training Script.

Trains a Denoising Autoencoder on the BEV observations from a saved SAC replay buffer.
This allows the UnifiedBEVEncoder to learn spatial representations (walls, gaps) 
without needing RL rewards.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import models
from model_architectures import BEVAutoencoder

class BEVDataset(Dataset):
    """
    Custom Dataset to load BEV frames dynamically.
    Avoids OOM by keeping the buffer in uint8 and normalizing on the fly.
    """
    def __init__(self, data_tensor):
        # Keep data as uint8 to save RAM
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract frame and normalize to [0, 1] on the fly
        frame = self.data[idx]
        return (frame.float() / 255.0,)


def train_autoencoder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading buffer from {args.buffer_path}...")
    if not os.path.exists(args.buffer_path):
        print(f"Error: Buffer file not found at {args.buffer_path}")
        return

    data = torch.load(args.buffer_path, map_location='cpu')
    size = data['size']
    if size == 0:
        print("Error: Buffer is empty!")
        return

    # Limit samples if requested
    if getattr(args, 'max_samples', 0) > 0 and size > args.max_samples:
        print(f"Limiting to the most recent {args.max_samples} frames...")
        # Get the most recent frames (end of buffer)
        bev_data = data['bev'][size - args.max_samples:size]
        size = args.max_samples
    else:
        bev_data = data['bev'][:size]
        
    print(f"Loaded {size} BEV frames of shape {bev_data.shape}")

    # Create DataLoader with custom Dataset (normalizes on the fly)
    print("Initializing dynamic BEVDataset...")
    
    # --- PERFORMANCE OPTIMIZATION: GPU PRELOADING ---
    if getattr(args, 'gpu_buffer', False) and device.type == 'cuda':
        print("🚀 Preloading entire dataset into GPU VRAM for maximum speed...")
        # Move the uint8 dataset to GPU. 100k frames is ~3.27 GB.
        bev_data = bev_data.to(device, non_blocking=True)
        dataset = BEVDataset(bev_data)
        # Since data is on GPU, num_workers must be 0 and pin_memory False
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=False
        )
    else:
        dataset = BEVDataset(bev_data)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )

    # 2. Setup Model & Optimizer
    model = BEVAutoencoder(input_channels=2).to(device)
    
    # --- PERFORMANCE OPTIMIZATION: torch.compile and TF32 ---
    if device.type == 'cuda':
        print("🚀 Enabling TF32 and torch.compile...")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    # We use MSELoss for reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # --- PERFORMANCE OPTIMIZATION: Automatic Mixed Precision (AMP) ---
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
         print("🚀 Enabling Automatic Mixed Precision (AMP)...")

    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (batch_x,) in enumerate(dataloader):
            # If we preloaded to GPU, it's already there
            if batch_x.device != device:
                 batch_x = batch_x.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                # Forward pass with noise for denoising AE
                recon = model(batch_x, noise_factor=args.noise_factor)
                # Loss against the original CLEAN input
                loss = criterion(recon, batch_x)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"==== Epoch {epoch} Summary ==== Avg Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Extract the raw encoder to avoid saving the torch.compile wrapper
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            # We save ONLY the encoder so it can be easily loaded into SAC models
            save_path = os.path.join(args.save_dir, "best_bev_encoder.pt")
            torch.save(raw_model.encoder.state_dict(), save_path)
            print(f"  -> Saved new best encoder to {save_path}")

    print("Training complete! Best encoder saved to:", os.path.join(args.save_dir, "best_bev_encoder.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BEV Autoencoder from Replay Buffer")
    parser.add_argument("--buffer_path", type=str, default="checkpoints_sac/replay_buffer.pt", help="Path to saved replay buffer")
    parser.add_argument("--save_dir", type=str, default="checkpoints_sac/vae", help="Directory to save encoder weights")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (increase for higher GPU utilization)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_factor", type=float, default=0.1, help="Amount of noise to add for denoising objective")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum number of frames to load from buffer (0 for all)")
    parser.add_argument("--gpu_buffer", action="store_true", help="Preload the entire dataset into GPU VRAM for maximum speed")
    
    args = parser.parse_args()
    train_autoencoder(args)
