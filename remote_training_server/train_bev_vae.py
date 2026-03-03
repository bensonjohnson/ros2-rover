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
from torch.utils.data import TensorDataset, DataLoader

# Import models
from model_architectures import BEVAutoencoder

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

    # Extract BEV data (uint8, 0-255)
    bev_data = data['bev'][:size]
    print(f"Loaded {size} BEV frames of shape {bev_data.shape}")

    # Convert to float32 and normalize to [0, 1]
    # To save RAM, we might do this in the DataLoader if size is huge, 
    # but for typical buffers <1M it fits in CPU RAM.
    print("Normalizing data to [0, 1]...")
    bev_data = bev_data.float() / 255.0

    # Create DataLoader
    dataset = TensorDataset(bev_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    # 2. Setup Model & Optimizer
    model = BEVAutoencoder(input_channels=2).to(device)
    
    # We use MSELoss for reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"Starting training for {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (batch_x,) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with noise for denoising AE
            recon = model(batch_x, noise_factor=args.noise_factor)
            
            # Loss against the original CLEAN input
            loss = criterion(recon, batch_x)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"==== Epoch {epoch} Summary ==== Avg Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # We save ONLY the encoder so it can be easily loaded into SAC models
            save_path = os.path.join(args.save_dir, "best_bev_encoder.pt")
            torch.save(model.encoder.state_dict(), save_path)
            print(f"  -> Saved new best encoder to {save_path}")

    print("Training complete! Best encoder saved to:", os.path.join(args.save_dir, "best_bev_encoder.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BEV Autoencoder from Replay Buffer")
    parser.add_argument("--buffer_path", type=str, default="checkpoints_sac/replay_buffer.pt", help="Path to saved replay buffer")
    parser.add_argument("--save_dir", type=str, default="checkpoints_sac/vae", help="Directory to save encoder weights")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_factor", type=float, default=0.1, help="Amount of noise to add for denoising objective")
    
    args = parser.parse_args()
    train_autoencoder(args)
