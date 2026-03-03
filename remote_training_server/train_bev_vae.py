#!/usr/bin/env python3
"""BEV Autoencoder Pre-training Script.

Trains a Denoising Autoencoder on the BEV observations from a saved SAC replay buffer.
This allows the UnifiedBEVEncoder to learn spatial representations (walls, gaps) 
without needing RL rewards.
"""

import os
import io
import time
import logging
import threading
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response, jsonify, render_template_string

# Import models
from model_architectures import BEVAutoencoder, BEVAutoencoderWithForwardPrediction

DASHBOARD_HTML = """
<!DOCTYPE html>
<html><head>
<title>BEV Encoder Pre-training</title>
<meta charset="utf-8">
<style>
  body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; }
  h1 { color: #00d4ff; margin-bottom: 5px; }
  .subtitle { color: #888; margin-bottom: 20px; }
  .stats { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 12px 18px; min-width: 140px; }
  .stat-label { color: #888; font-size: 12px; text-transform: uppercase; }
  .stat-value { color: #00d4ff; font-size: 24px; font-weight: bold; }
  .stat-value.good { color: #4ade80; }
  .samples-container { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 15px; }
  .samples-container img { max-width: 100%; height: auto; border-radius: 4px; }
  .no-data { color: #666; font-style: italic; padding: 40px; text-align: center; }
  .loss-history { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
  .loss-bar { display: flex; align-items: center; gap: 10px; margin: 3px 0; font-size: 13px; font-family: monospace; }
  .loss-bar .bar { height: 14px; background: #00d4ff; border-radius: 2px; transition: width 0.3s; }
  .loss-bar .epoch { color: #888; width: 60px; }
</style>
</head><body>
<h1>BEV Encoder Pre-training</h1>
<p class="subtitle">Denoising Autoencoder — Reconstruction Monitor</p>

<div class="stats">
  <div class="stat-card"><div class="stat-label">Epoch</div><div class="stat-value" id="epoch">-</div></div>
  <div class="stat-card"><div class="stat-label">Best Loss</div><div class="stat-value good" id="best_loss">-</div></div>
  <div class="stat-card"><div class="stat-label">Current Loss</div><div class="stat-value" id="current_loss">-</div></div>
  <div class="stat-card" id="recon_card" style="display:none"><div class="stat-label">Recon Loss</div><div class="stat-value" id="recon_loss">-</div></div>
  <div class="stat-card" id="fwd_card" style="display:none"><div class="stat-label">Fwd Loss</div><div class="stat-value" id="fwd_loss">-</div></div>
  <div class="stat-card"><div class="stat-label">Learning Rate</div><div class="stat-value" id="lr">-</div></div>
  <div class="stat-card"><div class="stat-label">Epoch Time</div><div class="stat-value" id="epoch_time">-</div></div>
  <div class="stat-card"><div class="stat-label">Status</div><div class="stat-value" id="status">-</div></div>
</div>

<div class="loss-history" id="loss_section">
  <h3 style="margin-top:0; color:#00d4ff;">Loss History</h3>
  <div id="loss_bars"></div>
</div>

<div class="samples-container">
  <h3 style="margin-top:0; color:#00d4ff;">Reconstruction Samples</h3>
  <div id="samples_img"><div class="no-data">Waiting for first epoch...</div></div>
</div>

<script>
function update() {
  fetch('/api/stats').then(r => r.json()).then(d => {
    document.getElementById('epoch').textContent = d.epoch + '/' + d.total_epochs;
    document.getElementById('best_loss').textContent = d.best_loss.toFixed(6);
    document.getElementById('current_loss').textContent = d.current_loss.toFixed(6);
    document.getElementById('lr').textContent = d.lr.toExponential(1);
    document.getElementById('epoch_time').textContent = d.epoch_time.toFixed(1) + 's';
    document.getElementById('status').textContent = d.status;

    if (d.recon_loss !== undefined && d.recon_loss !== null) {
      document.getElementById('recon_card').style.display = '';
      document.getElementById('recon_loss').textContent = d.recon_loss.toFixed(6);
      document.getElementById('fwd_card').style.display = '';
      document.getElementById('fwd_loss').textContent = d.fwd_loss.toFixed(6);
    }

    // Loss bars
    if (d.loss_history && d.loss_history.length > 0) {
      var maxLoss = d.loss_history[0][1];
      var html = '';
      d.loss_history.forEach(function(item) {
        var pct = Math.min(100, (item[1] / maxLoss) * 100);
        html += '<div class="loss-bar"><span class="epoch">E' + item[0] + '</span>'
              + '<div class="bar" style="width:' + pct + '%"></div>'
              + '<span>' + item[1].toFixed(6) + '</span></div>';
      });
      document.getElementById('loss_bars').innerHTML = html;
    }
  }).catch(function(){});

  // Refresh sample image with cache bust
  var img = document.getElementById('sample_image');
  if (img) {
    img.src = '/api/samples?t=' + Date.now();
  } else {
    var container = document.getElementById('samples_img');
    fetch('/api/samples').then(r => {
      if (r.ok) {
        container.innerHTML = '<img id="sample_image" src="/api/samples?t=' + Date.now() + '">';
      }
    }).catch(function(){});
  }
}
setInterval(update, 3000);
update();
</script>
</body></html>
"""


class VAEDashboard:
    """Lightweight Flask dashboard for monitoring encoder pre-training."""

    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        # Shared state updated by the training loop
        self.stats = {
            'epoch': 0, 'total_epochs': 0, 'best_loss': float('inf'),
            'current_loss': 0.0, 'recon_loss': None, 'fwd_loss': None,
            'lr': 0.0, 'epoch_time': 0.0,
            'status': 'Initializing', 'loss_history': [],
        }
        self.latest_samples_png = None

        # Suppress Flask request logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.app.add_url_rule('/', 'index', self._index)
        self.app.add_url_rule('/api/stats', 'stats', self._stats)
        self.app.add_url_rule('/api/samples', 'samples', self._samples)

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        print(f"📊 VAE Dashboard running at http://{self.host}:{self.port}")

    def _run(self):
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def _index(self):
        return render_template_string(DASHBOARD_HTML)

    def _stats(self):
        return jsonify(self.stats)

    def _samples(self):
        if self.latest_samples_png is None:
            return Response('No samples yet', status=204)
        return Response(self.latest_samples_png, mimetype='image/png')

    def update_stats(self, epoch, total_epochs, avg_loss, best_loss, lr, elapsed, status='Training', recon_loss=None, fwd_loss=None):
        self.stats.update({
            'epoch': epoch, 'total_epochs': total_epochs,
            'current_loss': avg_loss, 'best_loss': best_loss,
            'recon_loss': recon_loss, 'fwd_loss': fwd_loss,
            'lr': lr, 'epoch_time': elapsed, 'status': status,
        })
        self.stats['loss_history'].append((epoch, avg_loss))

    def update_samples(self, fig):
        """Store a matplotlib figure as PNG bytes."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        self.latest_samples_png = buf.read()


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


class BEVTransitionDataset(Dataset):
    """Dataset of (bev_t, action_t, bev_t+1) tuples for forward prediction.

    Pre-filters episode boundaries: indices where dones[i] == 1.0 are excluded
    since bev[i+1] would be from a different episode.
    """
    def __init__(self, bev_data, actions, dones):
        self.bev = bev_data
        self.actions = actions
        # Valid indices: not the last frame, and not a terminal state
        self.valid_indices = []
        n = len(bev_data)
        for i in range(n - 1):
            if dones[i] < 0.5:  # not a terminal transition
                self.valid_indices.append(i)
        self.valid_indices = torch.tensor(self.valid_indices, dtype=torch.long)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx].item()
        bev_t = self.bev[i].float() / 255.0
        action_t = self.actions[i].float()
        bev_t1 = self.bev[i + 1].float() / 255.0
        return bev_t, action_t, bev_t1


def save_samples(model, sample_batch, epoch, save_dir, device, dashboard=None):
    """Save a grid of original vs reconstructed BEV frames."""
    model.eval()
    with torch.no_grad():
        recon = model(sample_batch)

    n = min(6, sample_batch.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))

    for i in range(n):
        orig = sample_batch[i].cpu()
        rec = recon[i].cpu()

        # Channel 0: LiDAR occupancy
        axes[0, i].imshow(orig[0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'LiDAR #{i}' if i > 0 else 'Original LiDAR')
        axes[0, i].axis('off')

        # Reconstruction channel 0
        axes[1, i].imshow(rec[0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon #{i}' if i > 0 else 'Recon LiDAR')
        axes[1, i].axis('off')

        # Difference (amplified)
        diff = (orig[0] - rec[0]).abs()
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].set_title(f'Diff #{i}' if i > 0 else 'Error (hot)')
        axes[2, i].axis('off')

    fig.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    # Push to dashboard
    if dashboard:
        dashboard.update_samples(fig)

    if save_dir:
        path = os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png')
        fig.savefig(path, dpi=100)
        print(f"  -> Saved sample images to {path}")

    plt.close(fig)


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
        start_idx = size - args.max_samples
        bev_data = data['bev'][start_idx:size]
        if args.forward_prediction:
            actions_data = data['actions'][start_idx:size]
            dones_data = data['dones'][start_idx:size]
        size = args.max_samples
    else:
        bev_data = data['bev'][:size]
        if args.forward_prediction:
            actions_data = data['actions'][:size]
            dones_data = data['dones'][:size]

    print(f"Loaded {size} BEV frames of shape {bev_data.shape}")

    # Create DataLoader with custom Dataset (normalizes on the fly)
    use_fwd = args.forward_prediction
    if use_fwd:
        print("Initializing BEVTransitionDataset for forward prediction...")
        dataset = BEVTransitionDataset(bev_data, actions_data, dones_data)
        print(f"Valid transitions: {len(dataset)} / {size}")
    else:
        print("Initializing dynamic BEVDataset...")
        dataset = BEVDataset(bev_data)

    # --- PERFORMANCE OPTIMIZATION: GPU PRELOADING ---
    if getattr(args, 'gpu_buffer', False) and device.type == 'cuda':
        print("🚀 Preloading entire dataset into GPU VRAM for maximum speed...")
        if use_fwd:
            dataset.bev = dataset.bev.to(device, non_blocking=True)
            dataset.actions = dataset.actions.to(device, non_blocking=True)
            dataset.valid_indices = dataset.valid_indices.to(device, non_blocking=True)
        else:
            bev_data = bev_data.to(device, non_blocking=True)
            dataset = BEVDataset(bev_data)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )

    # 2. Setup Model & Optimizer
    if use_fwd:
        model = BEVAutoencoderWithForwardPrediction(input_channels=2).to(device)
    else:
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

    # BCEWithLogitsLoss is AMP-safe and numerically stable — combines sigmoid + BCE
    # in one step. Correct loss for binary occupancy grids.
    recon_criterion = nn.BCEWithLogitsLoss()
    fwd_criterion = nn.MSELoss() if use_fwd else None
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # --- PERFORMANCE OPTIMIZATION: Automatic Mixed Precision (AMP) ---
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
         print("🚀 Enabling Automatic Mixed Precision (AMP)...")

    # Start web dashboard
    dashboard = VAEDashboard(port=args.dashboard_port)
    dashboard.start()

    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    # Grab fixed sample frames for visual comparison across epochs
    sample_indices = torch.randperm(len(dataset))[:6]
    # Both BEVDataset and BEVTransitionDataset return bev as first element
    sample_batch = torch.stack([dataset[i][0] for i in sample_indices]).to(device)

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_fwd_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if use_fwd:
                bev_t, action_t, bev_t1 = batch
                if bev_t.device != device:
                    bev_t = bev_t.to(device, non_blocking=True)
                    action_t = action_t.to(device, non_blocking=True)
                    bev_t1 = bev_t1.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    # Reconstruction path
                    logits, z_t = model.forward_reconstruct(bev_t, noise_factor=args.noise_factor)
                    recon_loss = recon_criterion(logits, bev_t)

                    # Forward prediction path (stop gradient on target)
                    # L2-normalize both sides so MSE ∈ [0, 4] — prevents
                    # unbounded latent magnitudes from exploding the loss
                    with torch.no_grad():
                        z_t1_target = F.normalize(model.encode(bev_t1), dim=1).detach()
                    z_t1_pred = F.normalize(model.forward_predict(z_t, action_t), dim=1)
                    fwd_loss = fwd_criterion(z_t1_pred, z_t1_target)

                    loss = recon_loss + args.fwd_weight * fwd_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_fwd_loss += fwd_loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}/{args.epochs} [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.6f} (recon: {recon_loss.item():.6f}, fwd: {fwd_loss.item():.6f})")
            else:
                (batch_x,) = batch
                if batch_x.device != device:
                    batch_x = batch_x.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    recon = model(batch_x, noise_factor=args.noise_factor)
                    loss = recon_criterion(recon, batch_x)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}/{args.epochs} [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.6f}")

        scheduler.step()
        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        if use_fwd:
            avg_recon = epoch_recon_loss / n_batches
            avg_fwd = epoch_fwd_loss / n_batches
            print(f"==== Epoch {epoch} Summary ==== Loss: {avg_loss:.6f} (recon: {avg_recon:.6f}, fwd: {avg_fwd:.6f}) | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")
            dashboard.update_stats(epoch, args.epochs, avg_loss, min(best_loss, avg_loss), current_lr, elapsed, recon_loss=avg_recon, fwd_loss=avg_fwd)
        else:
            print(f"==== Epoch {epoch} Summary ==== Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")
            dashboard.update_stats(epoch, args.epochs, avg_loss, min(best_loss, avg_loss), current_lr, elapsed)

        # Save sample reconstructions — every epoch to dashboard, every 5 to disk
        save_to_disk = (epoch == 1 or epoch % 5 == 0)
        save_samples(model, sample_batch, epoch, args.save_dir if save_to_disk else None, device, dashboard)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Extract the raw encoder to avoid saving the torch.compile wrapper
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            # We save ONLY the encoder so it can be easily loaded into SAC models
            save_path = os.path.join(args.save_dir, "best_bev_encoder.pt")
            torch.save(raw_model.encoder.state_dict(), save_path)
            print(f"  -> Saved new best encoder to {save_path}")

    dashboard.stats['status'] = 'Complete'
    print("Training complete! Best encoder saved to:", os.path.join(args.save_dir, "best_bev_encoder.pt"))
    print(f"📊 Dashboard still running at http://0.0.0.0:{args.dashboard_port} — Ctrl+C to exit")


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
    parser.add_argument("--dashboard_port", type=int, default=5001, help="Port for the web dashboard (5001 to avoid conflict with SAC dashboard on 5000)")
    parser.add_argument("--forward_prediction", action="store_true", help="Enable latent forward-prediction loss alongside reconstruction")
    parser.add_argument("--fwd_weight", type=float, default=1.0, help="Weight for forward prediction loss (default: 1.0)")

    args = parser.parse_args()
    train_autoencoder(args)
