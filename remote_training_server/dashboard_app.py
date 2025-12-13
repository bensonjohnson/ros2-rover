import threading
import time
import logging
import os
from flask import Flask, jsonify, render_template_string, Response
import torch
import cv2
import numpy as np
import io

# Disable Flask banner
cli = logging.getLogger('flask.cli')
cli.setLevel(logging.ERROR)

class TrainingDashboard:
    """Flask-based dashboard for monitoring SAC training."""

    def __init__(self, trainer, host='0.0.0.0', port=5000):
        self.trainer = trainer
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.server_thread = None

        # Register routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/api/stats', 'get_stats', self.get_stats)
        self.app.add_url_rule('/api/metrics_history', 'get_metrics_history', self.get_metrics_history)
        self.app.add_url_rule('/api/system_resources', 'get_system_resources', self.get_system_resources)
        self.app.add_url_rule('/api/system_resources', 'get_system_resources', self.get_system_resources)
        self.app.add_url_rule('/api/laser', 'get_laser', self.get_laser)
        self.app.add_url_rule('/api/depth', 'get_depth', self.get_depth)
        self.app.add_url_rule('/api/rgbd', 'get_rgbd', self.get_rgbd)
        # Keep /api/grid for backward compatibility/debugging if needed, but we'll use laser now
        self.app.add_url_rule('/api/grid', 'get_grid', self.get_grid)

    def start(self):
        """Start the dashboard in a background thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"📊 Dashboard running at http://{self.host}:{self.port}")

    def _run_server(self):
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def get_stats(self):
        """Return current training statistics as JSON."""
        try:
            # Get latest metrics
            latest_metrics = {}
            if self.trainer.metrics_history:
                latest_metrics = dict(self.trainer.metrics_history[-1])

            # Get checkpoint info
            checkpoint_files = list(self.trainer.args.checkpoint_dir.glob('sac_step_*.pt')) if hasattr(self.trainer.args.checkpoint_dir, 'glob') else []
            last_checkpoint_time = None
            last_checkpoint_size = 0
            if checkpoint_files:
                from pathlib import Path
                checkpoint_dir = Path(self.trainer.args.checkpoint_dir)
                checkpoint_files = list(checkpoint_dir.glob('sac_step_*.pt'))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    last_checkpoint_time = latest_checkpoint.stat().st_mtime
                    last_checkpoint_size = latest_checkpoint.stat().st_size / (1024 * 1024)  # MB

            stats = {
                # System Status
                'status': 'Training' if self.trainer.training_active else 'Paused/Collecting',
                'device': str(self.trainer.device),
                'nats_connected': self.trainer.nc.is_connected if self.trainer.nc else False,

                # Training Progress
                'total_steps': self.trainer.total_steps,
                'gradient_steps': self.trainer.gradient_steps,
                'model_version': self.trainer.model_version,

                # Buffer
                'buffer_size': self.trainer.buffer.size,
                'buffer_capacity': self.trainer.buffer.capacity,
                'buffer_percent': (self.trainer.buffer.size / self.trainer.buffer.capacity) * 100,

                # Hyperparameters
                'batch_size': self.trainer.args.batch_size,
                'learning_rate': self.trainer.args.lr,
                'gamma': self.trainer.args.gamma,
                'utd_ratio': self.trainer.args.utd_ratio,
                'actor_update_freq': self.trainer.args.actor_update_freq,
                'droq_dropout': self.trainer.args.droq_dropout,
                'droq_samples': self.trainer.args.droq_samples,
                'droq_enabled': self.trainer.args.droq_dropout > 0.0,
                'augment_enabled': self.trainer.args.augment_data,

                # Training Metrics (latest)
                'actor_loss': latest_metrics.get('actor_loss', 0.0),
                'critic_loss': latest_metrics.get('critic_loss', 0.0),
                'alpha': latest_metrics.get('alpha', 0.0),
                'alpha_loss': latest_metrics.get('alpha_loss', 0.0),
                'policy_entropy': latest_metrics.get('policy_entropy', 0.0),
                'q1_mean': latest_metrics.get('q1_mean', 0.0),
                'q2_mean': latest_metrics.get('q2_mean', 0.0),
                'q_target_mean': latest_metrics.get('q_target_mean', 0.0),
                'q_value_mean': latest_metrics.get('q_value_mean', 0.0),
                'reward_mean': latest_metrics.get('reward_mean', 0.0),
                'reward_std': latest_metrics.get('reward_std', 0.0),

                # Throughput
                'steps_per_sec': self.trainer.steps_per_sec,
                'samples_per_sec': self.trainer.steps_per_sec * self.trainer.args.batch_size,
                'grad_steps_per_sec': self.trainer.steps_per_sec * self.trainer.args.utd_ratio,
                'sample_efficiency': self.trainer.gradient_steps / max(self.trainer.total_steps, 1),

                # Checkpoint Info
                'last_checkpoint_time': last_checkpoint_time,
                'last_checkpoint_size_mb': last_checkpoint_size,

                # Misc
                'perception_mode': 'Occupancy Grid (64x64)',
                'target_entropy': float(self.trainer.target_entropy),
            }
            return jsonify(stats)
        except Exception as e:
            print(f"Error in get_stats: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    def get_metrics_history(self):
        """Return metrics history for plotting."""
        try:
            history = []
            for metrics in self.trainer.metrics_history:
                history.append({
                    'step': metrics.get('step', 0),
                    'timestamp': metrics.get('timestamp', 0),
                    'actor_loss': metrics.get('actor_loss', 0.0),
                    'critic_loss': metrics.get('critic_loss', 0.0),
                    'alpha': metrics.get('alpha', 0.0),
                    'policy_entropy': metrics.get('policy_entropy', 0.0),
                    'q1_mean': metrics.get('q1_mean', 0.0),
                    'q2_mean': metrics.get('q2_mean', 0.0),
                    'q_target_mean': metrics.get('q_target_mean', 0.0),
                    'reward_mean': metrics.get('reward_mean', 0.0),
                    'reward_std': metrics.get('reward_std', 0.0),
                })
            return jsonify(history)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_system_resources(self):
        """Return system resource usage."""
        try:
            resources = {}

            # GPU Memory
            if torch.cuda.is_available():
                resources['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
                resources['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
                resources['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                resources['gpu_memory_percent'] = (resources['gpu_memory_allocated'] / resources['gpu_memory_total']) * 100
            else:
                resources['gpu_memory_allocated'] = 0
                resources['gpu_memory_reserved'] = 0
                resources['gpu_memory_total'] = 0
                resources['gpu_memory_percent'] = 0

            # Try to get system RAM (requires psutil)
            try:
                import psutil
                mem = psutil.virtual_memory()
                resources['system_memory_used'] = mem.used / (1024 ** 3)  # GB
                resources['system_memory_total'] = mem.total / (1024 ** 3)  # GB
                resources['system_memory_percent'] = mem.percent
                resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            except ImportError:
                resources['system_memory_used'] = 0
                resources['system_memory_total'] = 0
                resources['system_memory_percent'] = 0
                resources['cpu_percent'] = 0

            return jsonify(resources)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_laser(self):
        """Return the latest laser grid as a PNG image."""
        try:
            grid = self.trainer.latest_laser_vis
            if grid is None:
                vis = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                # grid is (1, 128, 128) or (128, 128) float32 (0 or 1)
                # Ensure 2D
                if grid.ndim == 3:
                     grid = grid[0]
                
                # Resize to display
                vis_grid = cv2.resize(grid, (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # Colorize: 0=Free (Light), 1=Occupied (Red)
                # Note: Logic in processor is 1=Occupied?
                # Usually 0=Free, 1=Occupied.
                
                vis = np.zeros((256, 256, 3), dtype=np.uint8)
                vis[:, :] = [200, 200, 200] # Free (Light Gray)
                
                mask_occupied = (vis_grid > 0.5)
                vis[mask_occupied] = [0, 0, 255] # Red
                
            _, buffer = cv2.imencode('.png', vis)
            return Response(buffer.tobytes(), mimetype='image/png')
        except Exception as e:
            print(f"Error serving laser: {e}")
            return Response(status=500)

    def get_depth(self):
        """Return the latest depth image as a PNG (extracted from RGBD)."""
        try:
            rgbd = self.trainer.latest_rgbd_vis
            if rgbd is None:
                vis = np.zeros((240, 424, 3), dtype=np.uint8)
            else:
                # rgbd is (4, 240, 424) - extract depth channel (index 3)
                if rgbd.shape[0] == 4:
                    depth = rgbd[3]  # 4th channel is depth
                else:
                    # Fallback if shape is different
                    vis = np.zeros((240, 424, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.png', vis)
                    return Response(buffer.tobytes(), mimetype='image/png')

                # Convert float [0, 1] to uint8 [0, 255] if needed
                if depth.dtype != np.uint8:
                    depth = (depth * 255.0).astype(np.uint8)

                # Apply colormap (close=red, far=blue)
                # Depth 0=Close, 255=Far
                # Invert so close becomes red in JET colormap
                depth_inverted = 255 - depth
                vis = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

            _, buffer = cv2.imencode('.png', vis)
            return Response(buffer.tobytes(), mimetype='image/png')
        except Exception as e:
            print(f"Error serving depth: {e}")
            import traceback
            traceback.print_exc()
            return Response(status=500)

    def get_rgbd(self):
        """Return the latest RGBD image as a PNG (RGB + Depth side-by-side)."""
        try:
            rgbd = self.trainer.latest_rgbd_vis
            if rgbd is None:
                # Create placeholder with RGB and Depth side-by-side
                rgb_placeholder = np.zeros((240, 424, 3), dtype=np.uint8)
                depth_placeholder = np.zeros((240, 424, 3), dtype=np.uint8)
                vis = np.hstack([rgb_placeholder, depth_placeholder])
            else:
                # rgbd is (4, 240, 424) uint8 - [R, G, B, D]
                # Reshape to (240, 424, 4) for easier processing
                if rgbd.shape[0] == 4:
                    rgbd = rgbd.transpose(1, 2, 0)  # (4, 240, 424) -> (240, 424, 4)

                # Extract RGB and Depth
                rgb = rgbd[:, :, :3]  # First 3 channels
                depth = rgbd[:, :, 3]  # 4th channel

                # Ensure RGB is uint8
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255.0).astype(np.uint8)

                # Convert depth to colormap (close=red, far=blue)
                if depth.dtype != np.uint8:
                    depth = (depth * 255.0).astype(np.uint8)

                depth_inverted = 255 - depth
                depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

                # Convert RGB from RGB to BGR for OpenCV
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Stack side-by-side: RGB | Depth
                vis = np.hstack([rgb_bgr, depth_colored])

            _, buffer = cv2.imencode('.png', vis)
            return Response(buffer.tobytes(), mimetype='image/png')
        except Exception as e:
            print(f"Error serving RGBD: {e}")
            import traceback
            traceback.print_exc()
            return Response(status=500)

    def get_grid(self):
        # Redirect wrapper for legacy
        return self.get_laser()

    def index(self):
        """Render the main dashboard page."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAC Training Dashboard - Full Featured</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --card-border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-primary: #58a6ff;
            --accent-secondary: #1f6feb;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --grid-gap: 16px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 20px;
            line-height: 1.5;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }

        .header h1 {
            font-size: 2rem;
            color: var(--accent-primary);
            margin-bottom: 10px;
        }

        .header .subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: var(--grid-gap);
            max-width: 1800px;
            margin: 0 auto;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title .icon {
            font-size: 1.2rem;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 0.9rem;
        }

        .stat-label {
            color: var(--text-secondary);
        }

        .stat-value {
            font-weight: 600;
            color: var(--accent-primary);
        }

        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.danger { color: var(--danger); }

        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: var(--card-border);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
            transition: width 0.5s ease;
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }

        .status-indicator.active {
            background-color: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        .status-indicator.inactive {
            background-color: var(--danger);
        }

        .grid-container {
            text-align: center;
            margin: 10px 0;
        }

        .grid-image {
            width: 256px;
            height: 256px;
            border: 2px solid var(--card-border);
            border-radius: 4px;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }

        .chart-container {
            position: relative;
            height: 250px;
            margin-top: 10px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 12px;
        }

        .metric-box {
            background-color: rgba(88, 166, 255, 0.1);
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid var(--accent-primary);
        }

        .metric-box .label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-box .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 4px;
        }

        .wide-card {
            grid-column: span 2;
        }

        @media (max-width: 768px) {
            .wide-card {
                grid-column: span 1;
            }
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .hyperparams-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 0.85rem;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.enabled {
            background-color: rgba(63, 185, 80, 0.2);
            color: var(--success);
        }

        .badge.disabled {
            background-color: rgba(248, 81, 73, 0.2);
            color: var(--danger);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SAC Training Dashboard</h1>
        <p class="subtitle">DroQ + UTD + Data Augmentation | Real-time Monitoring</p>
    </div>

    <div class="dashboard-grid">
        <!-- System Status -->
        <div class="card">
            <div class="card-title"><span class="icon">⚡</span> System Status</div>
            <div class="stat-row">
                <span class="stat-label">State</span>
                <span id="status-text" class="stat-value">Loading...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span id="device" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">NATS Connection</span>
                <span id="nats-status" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Throughput</span>
                <span id="steps-per-sec" class="stat-value">... steps/s</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Sample Efficiency</span>
                <span id="sample-efficiency" class="stat-value">...x</span>
            </div>
        </div>

        <!-- Training Progress -->
        <div class="card">
            <div class="card-title"><span class="icon">📊</span> Training Progress</div>
            <div class="stat-row">
                <span class="stat-label">Environment Steps</span>
                <span id="total-steps" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Gradient Steps</span>
                <span id="gradient-steps" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Model Version</span>
                <span id="model-version" class="stat-value">v0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">UTD Ratio</span>
                <span id="utd-ratio" class="stat-value">...</span>
            </div>
        </div>

        <!-- Replay Buffer -->
        <div class="card">
            <div class="card-title"><span class="icon">💾</span> Replay Buffer</div>
            <div class="stat-row">
                <span class="stat-label">Size</span>
                <span id="buffer-size" class="stat-value">0 / 0</span>
            </div>
            <div class="progress-bar">
                <div id="buffer-progress" class="progress-fill"></div>
            </div>
            <div class="stat-row" style="margin-top: 12px;">
                <span class="stat-label">Utilization</span>
                <span id="buffer-percent" class="stat-value">0%</span>
            </div>
        </div>

        <!-- DroQ Status -->
        <div class="card">
            <div class="card-title"><span class="icon">🎲</span> DroQ Status</div>
            <div class="stat-row">
                <span class="stat-label">DroQ Enabled</span>
                <span id="droq-status">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Dropout Rate</span>
                <span id="droq-dropout" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Forward Samples (M)</span>
                <span id="droq-samples" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Augmentation</span>
                <span id="augment-status">...</span>
            </div>
        </div>

        <!-- Training Metrics -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">📈</span> Training Metrics</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="label">Actor Loss</div>
                    <div class="value" id="actor-loss">0.00</div>
                </div>
                <div class="metric-box">
                    <div class="label">Critic Loss</div>
                    <div class="value" id="critic-loss">0.00</div>
                </div>
                <div class="metric-box">
                    <div class="label">Q-Value</div>
                    <div class="value" id="q-value-mean">0.00</div>
                </div>
                <div class="metric-box">
                    <div class="label">Entropy</div>
                    <div class="value" id="policy-entropy">0.00</div>
                </div>
                <div class="metric-box">
                    <div class="label">Alpha</div>
                    <div class="value" id="alpha-value">0.00</div>
                </div>
                <div class="metric-box">
                    <div class="label">Reward Mean</div>
                    <div class="value" id="reward-mean">0.00</div>
                </div>
            </div>
        </div>

        <!-- Live Sensors -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">👁️</span> Live Sensors</div>
            <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
                
                <!-- Laser Grid -->
                <div class="grid-container">
                    <div style="margin-bottom: 5px; color: var(--text-secondary); font-size: 0.9rem;">Laser Occupancy (128x128)</div>
                    <img id="laser-img" src="/api/laser" alt="Laser Grid" class="grid-image">
                    <div class="stat-row" style="margin-top: 5px; font-size: 0.8rem; justify-content: center; gap: 10px;">
                        <span class="badgem" style="color: #ff0000;">🔴 Occupied</span>
                        <span class="badgem" style="color: #cccccc;">⚪ Free</span>
                    </div>
                </div>

                <!-- Depth Image -->
                <div class="grid-container">
                    <div style="margin-bottom: 5px; color: var(--text-secondary); font-size: 0.9rem;">Raw Depth (424x240)</div>
                    <img id="depth-img" src="/api/depth" alt="Depth Image" class="grid-image" style="width: 424px; max-width: 100%;">
                </div>

                <!-- RGBD Image -->
                <div class="grid-container">
                    <div style="margin-bottom: 5px; color: var(--text-secondary); font-size: 0.9rem;">RGBD (RGB + Depth) (848x240)</div>
                    <img id="rgbd-img" src="/api/rgbd" alt="RGBD Image" class="grid-image" style="width: 848px; max-width: 100%;">
                    <div class="stat-row" style="margin-top: 5px; font-size: 0.8rem; justify-content: center; gap: 10px;">
                        <span class="badgem">📷 RGB | 🌡️ Depth</span>
                    </div>
                </div>

            </div>
        </div>

        <!-- System Resources -->
        <div class="card">
            <div class="card-title"><span class="icon">🖥️</span> System Resources</div>
            <div class="stat-row">
                <span class="stat-label">GPU Memory</span>
                <span id="gpu-memory" class="stat-value">... GB</span>
            </div>
            <div class="progress-bar">
                <div id="gpu-memory-progress" class="progress-fill"></div>
            </div>
            <div class="stat-row" style="margin-top: 12px;">
                <span class="stat-label">System RAM</span>
                <span id="system-memory" class="stat-value">... GB</span>
            </div>
            <div class="progress-bar">
                <div id="system-memory-progress" class="progress-fill"></div>
            </div>
            <div class="stat-row" style="margin-top: 12px;">
                <span class="stat-label">CPU Usage</span>
                <span id="cpu-usage" class="stat-value">...%</span>
            </div>
        </div>

        <!-- Checkpoint Info -->
        <div class="card">
            <div class="card-title"><span class="icon">💿</span> Checkpoint Info</div>
            <div class="stat-row">
                <span class="stat-label">Last Saved</span>
                <span id="last-checkpoint-time" class="stat-value">Never</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">File Size</span>
                <span id="checkpoint-size" class="stat-value">... MB</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Save Frequency</span>
                <span class="stat-value">Every 2000 steps</span>
            </div>
        </div>

        <!-- Hyperparameters -->
        <div class="card">
            <div class="card-title"><span class="icon">⚙️</span> Hyperparameters</div>
            <div class="hyperparams-grid">
                <div class="stat-row">
                    <span class="stat-label">Batch Size</span>
                    <span id="batch-size" class="stat-value">...</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Learning Rate</span>
                    <span id="learning-rate" class="stat-value">...</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Gamma</span>
                    <span id="gamma" class="stat-value">...</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">UTD Ratio</span>
                    <span id="utd-ratio-param" class="stat-value">...</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Actor Update Freq</span>
                    <span id="actor-update-freq" class="stat-value">...</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Target Entropy</span>
                    <span id="target-entropy" class="stat-value">...</span>
                </div>
            </div>
        </div>

        <!-- Loss Curves -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">📉</span> Loss Curves</div>
            <div class="chart-container">
                <canvas id="loss-chart"></canvas>
            </div>
        </div>

        <!-- Q-Value Evolution -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">📊</span> Q-Value Evolution</div>
            <div class="chart-container">
                <canvas id="qvalue-chart"></canvas>
            </div>
        </div>

        <!-- Entropy & Alpha -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">🌡️</span> Entropy & Temperature</div>
            <div class="chart-container">
                <canvas id="entropy-chart"></canvas>
            </div>
        </div>

        <!-- Reward Statistics -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">🎯</span> Reward Statistics</div>
            <div class="chart-container">
                <canvas id="reward-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Chart.js default configuration
        Chart.defaults.color = '#8b949e';
        Chart.defaults.borderColor = '#30363d';

        // Initialize charts
        let lossChart, qvalueChart, entropyChart, rewardChart;

        function initCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                        intersect: false,
                        mode: 'index',
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Training Steps' },
                            grid: { color: '#30363d' }
                        },
                        y: {
                            grid: { color: '#30363d' }
                        }
                    }
                }
            };

            // Loss Chart
            lossChart = new Chart(document.getElementById('loss-chart'), {
                ...chartConfig,
                data: {
                    datasets: [
                        {
                            label: 'Actor Loss',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            tension: 0.4,
                        },
                        {
                            label: 'Critic Loss',
                            data: [],
                            borderColor: '#3fb950',
                            backgroundColor: 'rgba(63, 185, 80, 0.1)',
                            tension: 0.4,
                        }
                    ]
                }
            });

            // Q-Value Chart
            qvalueChart = new Chart(document.getElementById('qvalue-chart'), {
                ...chartConfig,
                data: {
                    datasets: [
                        {
                            label: 'Q1',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            tension: 0.4,
                        },
                        {
                            label: 'Q2',
                            data: [],
                            borderColor: '#3fb950',
                            backgroundColor: 'rgba(63, 185, 80, 0.1)',
                            tension: 0.4,
                        },
                        {
                            label: 'Target Q',
                            data: [],
                            borderColor: '#d29922',
                            backgroundColor: 'rgba(210, 153, 34, 0.1)',
                            tension: 0.4,
                        }
                    ]
                }
            });

            // Entropy Chart
            entropyChart = new Chart(document.getElementById('entropy-chart'), {
                ...chartConfig,
                data: {
                    datasets: [
                        {
                            label: 'Policy Entropy',
                            data: [],
                            borderColor: '#bb86fc',
                            backgroundColor: 'rgba(187, 134, 252, 0.1)',
                            tension: 0.4,
                            yAxisID: 'y',
                        },
                        {
                            label: 'Alpha (Temperature)',
                            data: [],
                            borderColor: '#f85149',
                            backgroundColor: 'rgba(248, 81, 73, 0.1)',
                            tension: 0.4,
                            yAxisID: 'y1',
                        }
                    ]
                },
                options: {
                    ...chartConfig.options,
                    scales: {
                        x: chartConfig.options.scales.x,
                        y: {
                            type: 'linear',
                            position: 'left',
                            title: { display: true, text: 'Entropy' },
                            grid: { color: '#30363d' }
                        },
                        y1: {
                            type: 'linear',
                            position: 'right',
                            title: { display: true, text: 'Alpha' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });

            // Reward Chart
            rewardChart = new Chart(document.getElementById('reward-chart'), {
                ...chartConfig,
                data: {
                    datasets: [
                        {
                            label: 'Mean Reward',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            tension: 0.4,
                        }
                    ]
                }
            });
        }

        function updateCharts(history) {
            if (history.length === 0) return;

            // Prepare data
            const steps = history.map(h => h.step);
            const actorLoss = history.map(h => ({ x: h.step, y: h.actor_loss }));
            const criticLoss = history.map(h => ({ x: h.step, y: h.critic_loss }));
            const q1 = history.map(h => ({ x: h.step, y: h.q1_mean }));
            const q2 = history.map(h => ({ x: h.step, y: h.q2_mean }));
            const qTarget = history.map(h => ({ x: h.step, y: h.q_target_mean }));
            const entropy = history.map(h => ({ x: h.step, y: h.policy_entropy }));
            const alpha = history.map(h => ({ x: h.step, y: h.alpha }));
            const reward = history.map(h => ({ x: h.step, y: h.reward_mean }));

            // Update charts
            lossChart.data.datasets[0].data = actorLoss;
            lossChart.data.datasets[1].data = criticLoss;
            lossChart.update();

            qvalueChart.data.datasets[0].data = q1;
            qvalueChart.data.datasets[1].data = q2;
            qvalueChart.data.datasets[2].data = qTarget;
            qvalueChart.update();

            entropyChart.data.datasets[0].data = entropy;
            entropyChart.data.datasets[1].data = alpha;
            entropyChart.update();

            rewardChart.data.datasets[0].data = reward;
            rewardChart.update();
        }

        function formatTime(timestamp) {
            if (!timestamp) return 'Never';
            const date = new Date(timestamp * 1000);
            const now = new Date();
            const diff = Math.floor((now - date) / 1000);

            if (diff < 60) return `${diff}s ago`;
            if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
            return `${Math.floor(diff / 86400)}d ago`;
        }

        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                // System Status
                document.getElementById('status-text').textContent = data.status;
                document.getElementById('status-text').className = data.status === 'Training' ? 'stat-value success' : 'stat-value warning';

                document.getElementById('device').textContent = data.device;
                document.getElementById('nats-status').textContent = data.nats_connected ? '✓ Connected' : '✗ Disconnected';
                document.getElementById('nats-status').className = data.nats_connected ? 'stat-value success' : 'stat-value danger';

                document.getElementById('steps-per-sec').textContent = `${data.steps_per_sec.toFixed(1)} steps/s`;
                document.getElementById('sample-efficiency').textContent = `${data.sample_efficiency.toFixed(1)}x`;

                // Training Progress
                document.getElementById('total-steps').textContent = data.total_steps.toLocaleString();
                document.getElementById('gradient-steps').textContent = data.gradient_steps.toLocaleString();
                document.getElementById('model-version').textContent = 'v' + data.model_version;
                document.getElementById('utd-ratio').textContent = data.utd_ratio;

                // Buffer
                document.getElementById('buffer-size').textContent = `${data.buffer_size.toLocaleString()} / ${data.buffer_capacity.toLocaleString()}`;
                document.getElementById('buffer-progress').style.width = data.buffer_percent + '%';
                document.getElementById('buffer-percent').textContent = data.buffer_percent.toFixed(1) + '%';

                // DroQ Status
                document.getElementById('droq-status').innerHTML = data.droq_enabled ?
                    '<span class="badge enabled">Enabled</span>' :
                    '<span class="badge disabled">Disabled</span>';
                document.getElementById('droq-dropout').textContent = data.droq_dropout.toFixed(3);
                document.getElementById('droq-samples').textContent = data.droq_samples;
                document.getElementById('augment-status').innerHTML = data.augment_enabled ?
                    '<span class="badge enabled">Enabled</span>' :
                    '<span class="badge disabled">Disabled</span>';

                // Training Metrics
                document.getElementById('actor-loss').textContent = data.actor_loss.toFixed(3);
                document.getElementById('critic-loss').textContent = data.critic_loss.toFixed(3);
                document.getElementById('q-value-mean').textContent = data.q_value_mean.toFixed(2);
                document.getElementById('policy-entropy').textContent = data.policy_entropy.toFixed(3);
                document.getElementById('alpha-value').textContent = data.alpha.toFixed(3);
                document.getElementById('reward-mean').textContent = data.reward_mean.toFixed(3);

                // Hyperparameters
                document.getElementById('batch-size').textContent = data.batch_size;
                document.getElementById('learning-rate').textContent = data.learning_rate.toExponential(1);
                document.getElementById('gamma').textContent = data.gamma;
                document.getElementById('utd-ratio-param').textContent = data.utd_ratio;
                document.getElementById('actor-update-freq').textContent = data.actor_update_freq;
                document.getElementById('target-entropy').textContent = data.target_entropy.toFixed(2);

                // Checkpoint
                document.getElementById('last-checkpoint-time').textContent = formatTime(data.last_checkpoint_time);
                document.getElementById('checkpoint-size').textContent = data.last_checkpoint_size_mb ?
                    data.last_checkpoint_size_mb.toFixed(1) + ' MB' : '...';

            } catch (err) {
                console.error('Error fetching stats:', err);
            }
        }

        async function updateResources() {
            try {
                const response = await fetch('/api/system_resources');
                const data = await response.json();

                document.getElementById('gpu-memory').textContent =
                    `${data.gpu_memory_allocated.toFixed(1)} / ${data.gpu_memory_total.toFixed(1)} GB`;
                document.getElementById('gpu-memory-progress').style.width = data.gpu_memory_percent + '%';

                if (data.system_memory_total > 0) {
                    document.getElementById('system-memory').textContent =
                        `${data.system_memory_used.toFixed(1)} / ${data.system_memory_total.toFixed(1)} GB`;
                    document.getElementById('system-memory-progress').style.width = data.system_memory_percent + '%';
                    document.getElementById('cpu-usage').textContent = data.cpu_percent.toFixed(1) + '%';
                }
            } catch (err) {
                console.error('Error fetching resources:', err);
            }
        }

        async function updateMetricsHistory() {
            try {
                const response = await fetch('/api/metrics_history');
                const history = await response.json();
                updateCharts(history);
            } catch (err) {
                console.error('Error fetching metrics history:', err);
            }
        }

        // Initialize
        initCharts();
        updateStats();
        updateResources();
        updateMetricsHistory();

        // Update intervals
        setInterval(updateStats, 1000);  // Stats every 1s
        setInterval(updateResources, 2000);  // Resources every 2s
        setInterval(updateMetricsHistory, 2000);  // Charts every 2s

        // Update images every 500ms
        setInterval(() => {
            const timestamp = new Date().getTime();
            document.getElementById('laser-img').src = '/api/laser?t=' + timestamp;
            document.getElementById('depth-img').src = '/api/depth?t=' + timestamp;
            document.getElementById('rgbd-img').src = '/api/rgbd?t=' + timestamp;
        }, 500);
    </script>
</body>
</html>
        """
        return render_template_string(html)
