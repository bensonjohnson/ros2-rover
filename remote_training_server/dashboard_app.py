import threading
import time
import logging
from flask import Flask, jsonify, render_template_string
import torch

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

    def start(self):
        """Start the dashboard in a background thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"📊 Dashboard running at http://{self.host}:{self.port}")

    def _run_server(self):
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def get_stats(self):
        """Return current training statistics as JSON."""
        # safely access trainer state
        try:
            stats = {
                'status': 'Training' if self.trainer.training_active else 'Paused/Collecting',
                'total_steps': self.trainer.total_steps,
                'model_version': self.trainer.model_version,
                'buffer_size': self.trainer.buffer.size,
                'buffer_capacity': self.trainer.buffer.capacity,
                'buffer_percent': (self.trainer.buffer.size / self.trainer.buffer.capacity) * 100,
                'device': str(self.trainer.device),
                'nats_connected': self.trainer.nc.is_connected if self.trainer.nc else False,
                'semantic_augmentation': self.trainer.use_semantic_augmentation,
            }
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def index(self):
        """Render the main dashboard page."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAC Training Dashboard</title>
    <style>
        :root {
            --bg-color: #121212;
            --card-bg: #1e1e1e;
            --text-color: #e0e0e0;
            --accent-color: #bb86fc;
            --success-color: #03dac6;
            --warning-color: #cf6679;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max_width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: var(--accent-color);
            margin-bottom: 30px;
        }
        h2 {
            margin-top: 0;
            font-size: 1.2rem;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        .stat-value {
            font-weight: bold;
            color: var(--success-color);
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #333;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            background-color: var(--accent-color);
            width: 0%;
            transition: width 0.5s ease;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active { background-color: var(--success-color); }
        .status-paused { background-color: var(--warning-color); }
    </style>
</head>
<body>
    <h1>🚀 SAC Training Dashboard (V620)</h1>
    
    <div class="container">
        <!-- Status Card -->
        <div class="card">
            <h2>System Status</h2>
            <div class="stat-row">
                <span>State:</span>
                <span id="status-text" class="stat-value">Loading...</span>
            </div>
            <div class="stat-row">
                <span>NATS Connection:</span>
                <span id="nats-status" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span>Device:</span>
                <span id="device" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span>Semantic Augmentation:</span>
                <span id="semantic-aug" class="stat-value">...</span>
            </div>
        </div>

        <!-- Training Progress -->
        <div class="card">
            <h2>Training Progress</h2>
            <div class="stat-row">
                <span>Total Steps:</span>
                <span id="total-steps" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span>Model Version:</span>
                <span id="model-version" class="stat-value">v0</span>
            </div>
        </div>

        <!-- Buffer Health -->
        <div class="card">
            <h2>Replay Buffer</h2>
            <div class="stat-row">
                <span>Size:</span>
                <span id="buffer-size" class="stat-value">0 / 0</span>
            </div>
            <div class="progress-bar">
                <div id="buffer-progress" class="progress-fill"></div>
            </div>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status-text').textContent = data.status;
                    document.getElementById('status-text').style.color = data.status === 'Training' ? '#03dac6' : '#cf6679';
                    
                    document.getElementById('nats-status').textContent = data.nats_connected ? 'Connected' : 'Disconnected';
                    document.getElementById('nats-status').style.color = data.nats_connected ? '#03dac6' : '#cf6679';
                    
                    document.getElementById('device').textContent = data.device;
                    document.getElementById('semantic-aug').textContent = data.semantic_augmentation ? 'Enabled' : 'Disabled';
                    
                    document.getElementById('total-steps').textContent = data.total_steps.toLocaleString();
                    document.getElementById('model-version').textContent = 'v' + data.model_version;
                    
                    document.getElementById('buffer-size').textContent = `${data.buffer_size.toLocaleString()} / ${data.buffer_capacity.toLocaleString()}`;
                    document.getElementById('buffer-progress').style.width = data.buffer_percent + '%';
                })
                .catch(err => console.error('Error fetching stats:', err));
        }

        // Update every 1 second
        setInterval(updateStats, 1000);
        updateStats();
    </script>
</body>
</html>
        """
        return render_template_string(html)
