# Habitat Simulation Quick Start

## TL;DR

```bash
# 1. Install Habitat
conda install habitat-sim -c conda-forge -c aihabitat
pip install habitat-lab zstandard pyzmq

# 2. Download test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/

# 3. Start V620 trainer (Terminal 1)
cd /root/ros2-rover/remote_training_server
./start_map_elites_server.sh 5556 1000

# 4. Start Habitat sim (Terminal 2)
cd /root/ros2-rover/sim
./start_habitat_training.sh tcp://localhost:5556 1000 60.0 cuda
```

## What You Get

- **10-20x faster training** vs physical rover
- **URDF-accurate dimensions** (auto-loaded from robot definition)
- **Same ZeroMQ protocol** as physical rover (interchangeable!)
- **PyTorch on GPU** (no RKNN conversion needed)

## Key Features

### 🤖 URDF Integration (NEW!)

The simulator **automatically extracts robot dimensions** from your URDF:

```
✓ Loaded URDF dimensions:
  Agent radius: 93mm  (actual tank size!)
  Agent height: 90mm  (actual tank size!)
  Camera height: 123mm (actual RealSense mount!)
```

**No more manual configuration** - dimensions stay in sync with the physical robot.

### ⚡ Performance

Typical speeds on RTX 3090:
- Episode simulation: **10-20x realtime** (6-12s for 60s episode)
- Total throughput: **8-12 episodes/minute** vs 1 on rover
- Training 1000 episodes: **~1.5 hours** vs 16 hours

### 🎯 Accuracy

Matches physical rover:
- ✅ RealSense D435i (640×480, 69° HFOV)
- ✅ Tank dimensions (93mm radius, 90mm height)
- ✅ Max speed (0.18 m/s linear, 1.0 rad/s angular)
- ✅ Collision distance (120mm)
- ✅ Control rate (30Hz)

## File Structure

```
sim/
├── habitat_episode_runner.py      # Main simulation client
├── urdf_to_habitat.py             # URDF dimension extractor
├── habitat_config_from_urdf.yaml  # Auto-generated config
├── start_habitat_training.sh      # Startup script
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── ARCHITECTURE.md                # Protocol details
├── URDF_INTEGRATION.md            # URDF integration guide
└── QUICK_START.md                 # This file
```

## Troubleshooting

### "Habitat not found"
```bash
conda install habitat-sim -c conda-forge -c aihabitat
pip install habitat-lab
```

### "No scenes found"
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### "Connection refused"
Check V620 server is running:
```bash
netstat -tuln | grep 5556
```

### "CUDA out of memory"
Use CPU instead:
```bash
./start_habitat_training.sh tcp://localhost:5556 1000 60.0 cpu
```

## Next Steps

1. **Read README.md** for detailed usage
2. **Read URDF_INTEGRATION.md** to understand dimension extraction
3. **Read ARCHITECTURE.md** for protocol details
4. **Try hybrid training** (sim + rover in parallel!)

## Hybrid Training (Advanced)

Run **both sim and rover simultaneously** for best results:

```bash
# Terminal 1: Trainer
cd remote_training_server
./start_map_elites_server.sh 5556 1000

# Terminal 2: Physical rover
cd /root/ros2-rover
./start_map_elites_rover.sh

# Terminal 3: Habitat sim
cd sim
./start_habitat_training.sh tcp://10.0.0.200:5556 1000
```

The trainer will distribute models to **both** clients, mixing real and simulated episodes for maximum speed with real-world validation!
