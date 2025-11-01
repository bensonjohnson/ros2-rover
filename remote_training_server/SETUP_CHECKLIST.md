# V620 Setup Checklist

Quick reference checklist for setting up the V620 training server.

## Prerequisites

- [ ] V620 GPU machine with Ubuntu 22.04 or 24.04 (x86_64)
- [ ] Network connectivity between V620 and rover
- [ ] SSH access to both machines
- [ ] At least 50 GB free disk space
- [ ] Python 3.10 installed

## One-Time Setup

### V620 Server

- [ ] Copy `remote_training_server/` directory to V620
- [ ] Run `./setup_v620.sh`
- [ ] Verify installation completed successfully
- [ ] Test GPU: `rocm-smi`
- [ ] Test PyTorch: `python3 -c "import torch; print(torch.cuda.get_device_name(0))"`

### Rover

- [ ] Install RKNNLite runtime
- [ ] Install ZeroMQ: `pip3 install pyzmq`
- [ ] Build workspace: `colcon build --packages-select tractor_bringup tractor_control tractor_sensors`
- [ ] Test camera: `ros2 topic echo /camera/camera/color/image_raw`

## Network Configuration

- [ ] V620 firewall allows port 5555 (ZeroMQ)
- [ ] V620 firewall allows port 6006 (TensorBoard)
- [ ] Rover can ping V620: `ping V620_IP`
- [ ] V620 can SSH to rover: `ssh USER@ROVER_IP`

## Files Required

### V620 Server Files
```
remote_training_server/
├── v620_ppo_trainer.py          ✓ Training server
├── export_to_rknn.py            ✓ ONNX→RKNN converter
├── start_v620_server.sh         ✓ Startup script
├── deploy_model.sh              ✓ Deployment script
├── setup_v620.sh                ✓ Setup script
├── requirements.txt             ✓ Python dependencies
└── README.md                    ✓ Documentation
```

### Rover Files
```
src/tractor_bringup/tractor_bringup/
├── remote_training_collector.py     ✓ Data collector
└── remote_trained_inference.py      ✓ NPU inference

src/tractor_bringup/launch/
├── remote_training_collection.launch.py
└── remote_trained_inference.launch.py

Root directory/
├── start_remote_training_collection.sh
├── start_remote_trained_inference.sh
└── REMOTE_TRAINING_QUICKSTART.md
```

## First Run Test

### 1. Start V620 Server
```bash
cd ~/remote_training_server
./start_v620_server.sh
```

Expected output:
- ✓ All packages installed
- ✓ GPU detected: AMD Radeon Pro V620
- ✓ Port 5555 available
- ✓ TensorBoard started
- "Listening for rover data on port 5555"

### 2. Start Rover Collection
```bash
cd ~/Documents/ros2-rover
./start_remote_training_collection.sh tcp://V620_IP:5555
```

Expected output:
- ✓ Build complete
- ✓ USB power management configured
- ✓ ZeroMQ installed
- ✓ Connected to V620
- "Data collection running"

### 3. Verify Data Flow

On V620, check for incoming data:
```bash
tail -f logs/training_*.log
```

Should see:
```
Sent XXXX samples | Episode step: YY
```

### 4. Monitor TensorBoard

Open browser: `http://V620_IP:6006`

Should see:
- Graphs for episode_reward, policy_loss, value_loss
- Data points appearing as training progresses

## Troubleshooting Quick Fixes

### V620: "No GPU detected"
```bash
rocm-smi
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 -c "import torch; print(torch.cuda.is_available())"
```

### V620: "Port already in use"
```bash
lsof -i :5555
pkill -f v620_ppo_trainer
```

### Rover: "Cannot connect to V620"
```bash
# Check network
ping V620_IP

# Check firewall on V620
sudo ufw status
sudo ufw allow 5555

# Test ZeroMQ
python3 -c "import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.PUSH); s.connect('tcp://V620_IP:5555')"
```

### Rover: "ZeroMQ not installed"
```bash
pip3 install pyzmq
```

### Rover: "RKNNLite not found"
```bash
# Install RKNNLite (see installation guide)
pip3 show rknn-toolkit-lite2
```

## Performance Targets

After setup is complete, you should achieve:

### Data Collection
- [ ] 10 Hz RGB-D streaming to V620
- [ ] <100 ms latency between rover and V620
- [ ] ~1-2 MB/s network bandwidth usage

### Training (V620)
- [ ] ~2000 samples/sec processing speed
- [ ] <5 sec per PPO update (8192 samples)
- [ ] TensorBoard graphs updating in real-time
- [ ] GPU utilization >80%

### Inference (Rover)
- [ ] 10-30 FPS on RK3588 NPU
- [ ] 30-100 ms inference latency
- [ ] Smooth velocity commands

## Post-Setup Workflow

Once setup is verified:

1. Collect data: Drive rover with teleop for 20-30 min
2. Monitor training: Watch TensorBoard for improving rewards
3. Deploy model: After 30-50 updates, run `./deploy_model.sh`
4. Test inference: Run `./start_remote_trained_inference.sh`
5. Iterate: Collect more data, retrain, improve

## Support

- Detailed guide: `README.md`
- Quick start: `REMOTE_TRAINING_QUICKSTART.md`
- Installation: `setup_v620.sh`
- Startup: `start_v620_server.sh`

## Version Info

Track your versions for troubleshooting:

```bash
# V620
rocm-smi --showproductname
python3 --version
python3 -c "import torch; print(torch.__version__)"
python3 -c "import zmq; print(zmq.zmq_version())"

# Rover
cat /sys/kernel/debug/rknpu/version
python3 --version
ros2 --version
```
