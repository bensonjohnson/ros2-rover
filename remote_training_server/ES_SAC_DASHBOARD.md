# ES-SAC Hybrid Training Dashboard

A real-time web dashboard for monitoring the ES-SAC (Evolution Strategy + Soft Actor-Critic) hybrid training server.

## Features

### 🧬 Evolution Strategy Monitoring
- **Population Statistics**: Real-time best, mean, worst fitness and standard deviation
- **Population Table**: Detailed view of all individuals with their:
  - ID
  - Fitness score
  - Source (Init, Elite, Mutation, SAC Injection)
  - Evaluation status (Evaluated/Pending)
- **Fitness Evolution Chart**: Track how fitness evolves across generations
- **Generation Tracking**: Current generation number and population size

### 🎯 SAC Training Metrics
- **Loss Curves**: Real-time actor and critic loss tracking
- **Training Progress**: SAC training steps counter
- **Buffer Status**: Replay buffer size and utilization

### ⚡ System Status
- **Device Information**: Training and inference device (GPU/CPU)
- **NATS Connection**: Connection status to the message broker
- **Active Rovers**: Number of rovers currently running episodes
- **Inference Count**: Total number of inferences performed

### 🖥️ System Resources
- **GPU Memory**: Allocated, reserved, and total VRAM with usage percentage
- **System RAM**: Used and total system memory
- **CPU Usage**: Current CPU utilization

## Accessing the Dashboard

### Default URL
```
http://localhost:5000
```

### Custom Port
You can specify a custom port when starting the server:
```bash
python3 v700_es_sac_trainer.py --dashboard_port 8080
```

### Remote Access
If running on a remote server, you can:

1. **SSH Tunnel** (Recommended for security):
   ```bash
   ssh -L 5000:localhost:5000 user@remote-server
   ```
   Then access `http://localhost:5000` on your local machine

2. **Direct Access** (if firewall allows):
   ```
   http://remote-server-ip:5000
   ```

## Dashboard Layout

### Top Section
- **System Status Card**: Connection and device info
- **Training Progress Card**: Steps, generation, population, buffer stats

### Middle Section
- **Population Statistics Card**: Fitness metrics (best, mean, worst, std dev)
- **Population Details Table**: Sortable table of all individuals

### Bottom Section
- **Fitness Evolution Chart**: Multi-line graph showing fitness trends
- **SAC Loss Curves**: Actor and critic loss over training steps
- **System Resources Card**: GPU/RAM/CPU monitoring

## Understanding the Display

### Population Status Badges
- 🟣 **SAC**: Individual injected from the SAC agent
- 🟡 **Elite**: Elite individual from previous generation
- 🔵 **Mutation**: Mutated offspring
- ⚪ **Init**: Initial random individual

### Evaluation Status
- 🟢 **Evaluated**: Fitness has been calculated
- 🔴 **Pending**: Waiting for evaluation

### Fitness Scores
- **Positive is better**: Higher fitness = better performance
- **Hybrid Score**: Fitness = Episode Reward + Critic Q-Score
- Individuals with `-` have not been evaluated yet

## Auto-Refresh Rates
- **Stats**: 1 second
- **Population**: 1.5 seconds
- **System Resources**: 2 seconds

## API Endpoints

The dashboard exposes several REST API endpoints:

- `GET /`: Dashboard UI (HTML)
- `GET /api/stats`: General statistics
- `GET /api/population`: Population details and fitness stats
- `GET /api/sac_metrics`: SAC training metrics history
- `GET /api/system_resources`: GPU/RAM/CPU usage

## Requirements

### Python Packages
- Flask
- PyTorch (for GPU monitoring)
- psutil (optional, for system resource monitoring)

All other dependencies are included via CDN (Chart.js).

## Tips

1. **Track Evolution**: Watch the Fitness Evolution chart to see if the population is improving over generations
2. **Monitor SAC Injection**: Look for purple "SAC" badges to see when the trained agent gets injected
3. **Population Diversity**: Check the std dev - low values might indicate loss of diversity
4. **Buffer Before Training**: SAC training starts when buffer > 1000 samples
5. **Performance**: If inference count isn't increasing, check rover connections

## Troubleshooting

### Dashboard won't load
- Check that port 5000 (or custom port) isn't already in use
- Verify the trainer started successfully
- Check firewall settings

### No data showing
- Wait for the buffer to fill (>1000 samples)
- Ensure rovers are connected and sending data
- Check NATS connection status (should be green)

### Charts not updating
- Check browser console for errors
- Try refreshing the page
- Ensure JavaScript is enabled

## Development

To modify the dashboard:

1. Edit `es_sac_dashboard.py` for backend/API changes
2. Modify the HTML template in the `index()` method for UI changes
3. Charts use Chart.js - see [Chart.js docs](https://www.chartjs.org/) for customization
4. Styling uses CSS custom properties (variables) - edit `:root` section to change theme

## Example Screenshots

### What You'll See

**Population Table**: All 10 individuals with their current fitness scores, sorted by performance

**Fitness Chart**: Three lines showing best/mean/worst fitness evolving over generations

**SAC Losses**: Real-time training loss curves for the meta-learner

**System Stats**: Live GPU memory, CPU usage, and training throughput

---

Built with Flask, Chart.js, and lots of ☕
