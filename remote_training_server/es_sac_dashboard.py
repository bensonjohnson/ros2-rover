#!/usr/bin/env python3
"""Flask Dashboard for ES-SAC Hybrid Trainer Monitoring."""

import threading
import time
import logging
import os
from flask import Flask, jsonify, render_template_string, Response
import torch
import cv2
import numpy as np

# Disable Flask banner
cli = logging.getLogger('flask.cli')
cli.setLevel(logging.ERROR)

class ESSACDashboard:
    """Flask-based dashboard for monitoring ES-SAC training."""

    def __init__(self, trainer, host='0.0.0.0', port=5000):
        self.trainer = trainer
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.server_thread = None

        # Register routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/api/stats', 'get_stats', self.get_stats)
        self.app.add_url_rule('/api/population', 'get_population', self.get_population)
        self.app.add_url_rule('/api/sac_metrics', 'get_sac_metrics', self.get_sac_metrics)
        self.app.add_url_rule('/api/system_resources', 'get_system_resources', self.get_system_resources)

    def start(self):
        """Start the dashboard in a background thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"📊 ES-SAC Dashboard running at http://{self.host}:{self.port}")

    def _run_server(self):
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def get_stats(self):
        """Return current training statistics as JSON."""
        try:
            stats = {
                # System Status
                'status': 'Running',
                'device': str(self.trainer.device),
                'inference_device': str(self.trainer.inference_device),
                'nats_connected': self.trainer.nc.is_connected if self.trainer.nc else False,

                # Training Progress
                'total_steps': self.trainer.total_steps,

                # Buffer
                'buffer_size': self.trainer.buffer.size,
                'buffer_capacity': self.trainer.buffer.capacity,
                'buffer_percent': (self.trainer.buffer.size / self.trainer.buffer.capacity) * 100,

                # ES Population
                'population_size': self.trainer.pop_manager.pop_size,
                'current_generation': self.trainer.pop_manager.generation,
                'next_id': self.trainer.pop_manager.next_id,

                # Active Rovers
                'active_rovers': len(self.trainer.active_requests),

                # Inference Count
                'inference_count': getattr(self.trainer, '_inference_count', 0),
            }
            return jsonify(stats)
        except Exception as e:
            print(f"Error in get_stats: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    def get_population(self):
        """Return population details as JSON."""
        try:
            population_data = []

            for individual in self.trainer.pop_manager.population:
                population_data.append({
                    'id': individual['id'],
                    'fitness': float(individual['fitness']) if individual['fitness'] != -float('inf') else None,
                    'source': individual['source'],
                    'evaluated': individual['fitness'] != -float('inf')
                })

            # Calculate statistics
            evaluated = [p for p in self.trainer.pop_manager.population if p['fitness'] != -float('inf')]

            if evaluated:
                fitnesses = [p['fitness'] for p in evaluated]
                stats = {
                    'best_fitness': float(max(fitnesses)),
                    'worst_fitness': float(min(fitnesses)),
                    'mean_fitness': float(np.mean(fitnesses)),
                    'std_fitness': float(np.std(fitnesses)),
                    'evaluated_count': len(evaluated),
                    'unevaluated_count': self.trainer.pop_manager.pop_size - len(evaluated),
                }
            else:
                stats = {
                    'best_fitness': None,
                    'worst_fitness': None,
                    'mean_fitness': None,
                    'std_fitness': None,
                    'evaluated_count': 0,
                    'unevaluated_count': self.trainer.pop_manager.pop_size,
                }

            return jsonify({
                'population': population_data,
                'stats': stats
            })
        except Exception as e:
            print(f"Error in get_population: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    def get_sac_metrics(self):
        """Return SAC training metrics from TensorBoard writer."""
        try:
            # Try to get latest metrics from the writer's event file
            # For simplicity, we'll track recent metrics in the trainer
            if not hasattr(self.trainer, 'recent_sac_metrics'):
                self.trainer.recent_sac_metrics = {
                    'actor_loss': [],
                    'critic_loss': [],
                    'steps': []
                }

            metrics = {
                'actor_loss': self.trainer.recent_sac_metrics.get('actor_loss', []),
                'critic_loss': self.trainer.recent_sac_metrics.get('critic_loss', []),
                'steps': self.trainer.recent_sac_metrics.get('steps', []),
            }

            return jsonify(metrics)
        except Exception as e:
            print(f"Error in get_sac_metrics: {e}")
            import traceback
            traceback.print_exc()
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

    def index(self):
        """Render the main dashboard page."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ES-SAC Hybrid Training Dashboard</title>
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
            --purple: #bb86fc;
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
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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

        .chart-container {
            position: relative;
            height: 300px;
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

        .metric-box.es {
            background-color: rgba(187, 134, 252, 0.1);
            border-left-color: var(--purple);
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

        .full-width-card {
            grid-column: 1 / -1;
        }

        @media (max-width: 768px) {
            .wide-card, .full-width-card {
                grid-column: span 1;
            }
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .population-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }

        .population-table th {
            background-color: rgba(88, 166, 255, 0.1);
            padding: 8px;
            text-align: left;
            border-bottom: 2px solid var(--card-border);
            color: var(--text-secondary);
        }

        .population-table td {
            padding: 6px 8px;
            border-bottom: 1px solid var(--card-border);
        }

        .population-table tr:hover {
            background-color: rgba(88, 166, 255, 0.05);
        }

        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .badge.init { background-color: rgba(139, 148, 158, 0.3); color: var(--text-secondary); }
        .badge.elite { background-color: rgba(210, 153, 34, 0.3); color: var(--warning); }
        .badge.mutation { background-color: rgba(88, 166, 255, 0.3); color: var(--accent-primary); }
        .badge.sac { background-color: rgba(187, 134, 252, 0.3); color: var(--purple); }
        .badge.evaluated { background-color: rgba(63, 185, 80, 0.3); color: var(--success); }
        .badge.pending { background-color: rgba(248, 81, 73, 0.3); color: var(--danger); }

        .mini-metric {
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 12px;
            background-color: rgba(88, 166, 255, 0.1);
            border-radius: 6px;
            font-size: 0.85rem;
        }

        .mini-metric .label {
            color: var(--text-secondary);
            font-size: 0.7rem;
            text-transform: uppercase;
        }

        .mini-metric .value {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 ES-SAC Hybrid Training Dashboard</h1>
        <p class="subtitle">Evolution Strategy + Soft Actor-Critic | Real-time Population & SAC Monitoring</p>
    </div>

    <div class="dashboard-grid">
        <!-- System Status -->
        <div class="card">
            <div class="card-title"><span class="icon">⚡</span> System Status</div>
            <div class="stat-row">
                <span class="stat-label">Status</span>
                <span id="status-text" class="stat-value success">Running</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Training Device</span>
                <span id="device" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Inference Device</span>
                <span id="inference-device" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">NATS Connection</span>
                <span id="nats-status" class="stat-value">...</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Active Rovers</span>
                <span id="active-rovers" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Inferences</span>
                <span id="inference-count" class="stat-value">0</span>
            </div>
        </div>

        <!-- Training Progress -->
        <div class="card">
            <div class="card-title"><span class="icon">📊</span> Training Progress</div>
            <div class="stat-row">
                <span class="stat-label">SAC Training Steps</span>
                <span id="total-steps" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">ES Generation</span>
                <span id="generation" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Population Size</span>
                <span id="population-size" class="stat-value">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Buffer Size</span>
                <span id="buffer-size" class="stat-value">0 / 0</span>
            </div>
            <div class="progress-bar">
                <div id="buffer-progress" class="progress-fill"></div>
            </div>
        </div>

        <!-- ES Population Stats -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">🧬</span> Population Statistics</div>
            <div class="metric-grid">
                <div class="metric-box es">
                    <div class="label">Best Fitness</div>
                    <div class="value" id="best-fitness">-</div>
                </div>
                <div class="metric-box es">
                    <div class="label">Mean Fitness</div>
                    <div class="value" id="mean-fitness">-</div>
                </div>
                <div class="metric-box es">
                    <div class="label">Worst Fitness</div>
                    <div class="value" id="worst-fitness">-</div>
                </div>
                <div class="metric-box es">
                    <div class="label">Std Dev</div>
                    <div class="value" id="std-fitness">-</div>
                </div>
            </div>
            <div style="margin-top: 16px;">
                <span class="mini-metric">
                    <div class="label">Evaluated</div>
                    <div class="value" id="evaluated-count">0</div>
                </span>
                <span class="mini-metric">
                    <div class="label">Pending</div>
                    <div class="value" id="unevaluated-count">0</div>
                </span>
            </div>
        </div>

        <!-- Population Table -->
        <div class="card full-width-card">
            <div class="card-title"><span class="icon">📋</span> Population Details</div>
            <div style="max-height: 400px; overflow-y: auto;">
                <table class="population-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Fitness</th>
                            <th>Source</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="population-tbody">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Fitness Evolution Chart -->
        <div class="card full-width-card">
            <div class="card-title"><span class="icon">📈</span> Fitness Evolution</div>
            <div class="chart-container">
                <canvas id="fitness-chart"></canvas>
            </div>
        </div>

        <!-- SAC Loss Curves -->
        <div class="card wide-card">
            <div class="card-title"><span class="icon">🎯</span> SAC Loss Curves</div>
            <div class="chart-container" style="height: 250px;">
                <canvas id="sac-loss-chart"></canvas>
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
    </div>

    <script>
        // Chart.js default configuration
        Chart.defaults.color = '#8b949e';
        Chart.defaults.borderColor = '#30363d';

        // Initialize charts
        let fitnessChart, sacLossChart;

        // Population history for fitness evolution
        let fitnessHistory = {
            generations: [],
            best: [],
            mean: [],
            worst: []
        };

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
                            grid: { color: '#30363d' }
                        },
                        y: {
                            grid: { color: '#30363d' }
                        }
                    }
                }
            };

            // Fitness Evolution Chart
            fitnessChart = new Chart(document.getElementById('fitness-chart'), {
                ...chartConfig,
                data: {
                    datasets: [
                        {
                            label: 'Best Fitness',
                            data: [],
                            borderColor: '#3fb950',
                            backgroundColor: 'rgba(63, 185, 80, 0.1)',
                            tension: 0.4,
                            fill: false,
                        },
                        {
                            label: 'Mean Fitness',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            tension: 0.4,
                            fill: false,
                        },
                        {
                            label: 'Worst Fitness',
                            data: [],
                            borderColor: '#f85149',
                            backgroundColor: 'rgba(248, 81, 73, 0.1)',
                            tension: 0.4,
                            fill: false,
                        }
                    ]
                },
                options: {
                    ...chartConfig.options,
                    scales: {
                        x: {
                            ...chartConfig.options.scales.x,
                            title: { display: true, text: 'Generation' }
                        },
                        y: {
                            ...chartConfig.options.scales.y,
                            title: { display: true, text: 'Fitness' }
                        }
                    }
                }
            });

            // SAC Loss Chart
            sacLossChart = new Chart(document.getElementById('sac-loss-chart'), {
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
                            borderColor: '#bb86fc',
                            backgroundColor: 'rgba(187, 134, 252, 0.1)',
                            tension: 0.4,
                        }
                    ]
                },
                options: {
                    ...chartConfig.options,
                    scales: {
                        x: {
                            ...chartConfig.options.scales.x,
                            title: { display: true, text: 'Training Steps' }
                        },
                        y: {
                            ...chartConfig.options.scales.y,
                            title: { display: true, text: 'Loss' }
                        }
                    }
                }
            });
        }

        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                // System Status
                document.getElementById('status-text').textContent = data.status;
                document.getElementById('device').textContent = data.device;
                document.getElementById('inference-device').textContent = data.inference_device;
                document.getElementById('nats-status').textContent = data.nats_connected ? '✓ Connected' : '✗ Disconnected';
                document.getElementById('nats-status').className = data.nats_connected ? 'stat-value success' : 'stat-value danger';
                document.getElementById('active-rovers').textContent = data.active_rovers;
                document.getElementById('inference-count').textContent = data.inference_count.toLocaleString();

                // Training Progress
                document.getElementById('total-steps').textContent = data.total_steps.toLocaleString();
                document.getElementById('generation').textContent = data.current_generation;
                document.getElementById('population-size').textContent = data.population_size;
                document.getElementById('buffer-size').textContent = `${data.buffer_size.toLocaleString()} / ${data.buffer_capacity.toLocaleString()}`;
                document.getElementById('buffer-progress').style.width = data.buffer_percent + '%';

            } catch (err) {
                console.error('Error fetching stats:', err);
            }
        }

        async function updatePopulation() {
            try {
                const response = await fetch('/api/population');
                const data = await response.json();

                // Update population stats
                const stats = data.stats;
                document.getElementById('best-fitness').textContent = stats.best_fitness !== null ? stats.best_fitness.toFixed(2) : '-';
                document.getElementById('mean-fitness').textContent = stats.mean_fitness !== null ? stats.mean_fitness.toFixed(2) : '-';
                document.getElementById('worst-fitness').textContent = stats.worst_fitness !== null ? stats.worst_fitness.toFixed(2) : '-';
                document.getElementById('std-fitness').textContent = stats.std_fitness !== null ? stats.std_fitness.toFixed(2) : '-';
                document.getElementById('evaluated-count').textContent = stats.evaluated_count;
                document.getElementById('unevaluated-count').textContent = stats.unevaluated_count;

                // Update population table
                const tbody = document.getElementById('population-tbody');
                tbody.innerHTML = '';

                // Sort by fitness (descending), unevaluated last
                const sorted = [...data.population].sort((a, b) => {
                    if (a.fitness === null && b.fitness === null) return 0;
                    if (a.fitness === null) return 1;
                    if (b.fitness === null) return -1;
                    return b.fitness - a.fitness;
                });

                sorted.forEach(individual => {
                    const row = document.createElement('tr');

                    const fitnessText = individual.fitness !== null ? individual.fitness.toFixed(2) : '-';

                    let sourceBadge = '';
                    if (individual.source === 'sac_injection') {
                        sourceBadge = '<span class="badge sac">SAC</span>';
                    } else if (individual.source === 'elite') {
                        sourceBadge = '<span class="badge elite">Elite</span>';
                    } else if (individual.source.startsWith('mut_')) {
                        sourceBadge = '<span class="badge mutation">Mutation</span>';
                    } else {
                        sourceBadge = '<span class="badge init">Init</span>';
                    }

                    const statusBadge = individual.evaluated ?
                        '<span class="badge evaluated">Evaluated</span>' :
                        '<span class="badge pending">Pending</span>';

                    row.innerHTML = `
                        <td>#${individual.id}</td>
                        <td><strong>${fitnessText}</strong></td>
                        <td>${sourceBadge}</td>
                        <td>${statusBadge}</td>
                    `;
                    tbody.appendChild(row);
                });

                // Update fitness history chart
                if (stats.best_fitness !== null) {
                    const currentGen = parseInt(document.getElementById('generation').textContent);

                    // Only add new data point if generation changed
                    if (fitnessHistory.generations.length === 0 ||
                        fitnessHistory.generations[fitnessHistory.generations.length - 1] !== currentGen) {
                        fitnessHistory.generations.push(currentGen);
                        fitnessHistory.best.push(stats.best_fitness);
                        fitnessHistory.mean.push(stats.mean_fitness);
                        fitnessHistory.worst.push(stats.worst_fitness);

                        // Keep last 50 generations
                        if (fitnessHistory.generations.length > 50) {
                            fitnessHistory.generations.shift();
                            fitnessHistory.best.shift();
                            fitnessHistory.mean.shift();
                            fitnessHistory.worst.shift();
                        }

                        // Update chart
                        const genData = fitnessHistory.generations;
                        fitnessChart.data.datasets[0].data = genData.map((g, i) => ({x: g, y: fitnessHistory.best[i]}));
                        fitnessChart.data.datasets[1].data = genData.map((g, i) => ({x: g, y: fitnessHistory.mean[i]}));
                        fitnessChart.data.datasets[2].data = genData.map((g, i) => ({x: g, y: fitnessHistory.worst[i]}));
                        fitnessChart.update();
                    }
                }

            } catch (err) {
                console.error('Error fetching population:', err);
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

        // Initialize
        initCharts();
        updateStats();
        updatePopulation();
        updateResources();

        // Update intervals
        setInterval(updateStats, 1000);  // Stats every 1s
        setInterval(updatePopulation, 1500);  // Population every 1.5s
        setInterval(updateResources, 2000);  // Resources every 2s
    </script>
</body>
</html>
        """
        return render_template_string(html)
