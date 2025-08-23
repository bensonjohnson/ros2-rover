# 🚜 Tractor ES Training Simulation System

A comprehensive simulation environment for training Evolutionary Strategy (ES) models for autonomous robot navigation using PyBullet and ROCm-accelerated PyTorch.

## 🎯 Overview

This simulation system provides:

- **Realistic Physics Simulation**: PyBullet-based environment with accurate robot dynamics
- **Visual Feedback**: Real-time 3D visualization for training diagnostics
- **ROCm Acceleration**: GPU-accelerated training on AMD hardware
- **ES Training Pipeline**: Complete Evolutionary Strategy training workflow
- **Environment Variety**: Indoor, outdoor, and mixed environments
- **Model Export**: Automatic ONNX/RKNN conversion for robot deployment

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │   PyBullet  │  │        ES Training Pipeline         │  │
│  │ Environment │  │                                     │  │
│  │ Simulation  │  │  ┌───────────────────────────────┐  │  │
│  │             │  │  │ EvolutionaryStrategyTrainer   │  │  │
│  │             │  │  └───────────────────────────────┘  │  │
│  │             │  │                                     │  │
│  │             │  │  ┌───────────────────────────────┐  │  │
│  │             │  │  │    Reward Calculation         │  │  │
│  │             │  │  └───────────────────────────────┘  │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    ROCm PyTorch Backend                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make scripts executable
chmod +x setup_simulation.sh run_simulation.sh

# Run setup (builds Docker images)
./setup_simulation.sh
```

### 2. Run Simulation

```bash
# Run with GUI visualization
./run_simulation.sh

# Run with specific parameters
./run_simulation.sh --environment outdoor --population-size 20 --max-generations 200

# Run headless (no GUI)
./run_simulation.sh --no-gui
```

### 3. View Dashboard

```bash
# Start dashboard (if using docker-compose)
docker-compose -f docker-compose.simulation.yml up dashboard
```

Open `http://localhost:3000` in your browser.

## 📁 Project Structure

```
tractor_simulation/
├── tractor_simulation/
│   ├── bullet_simulation.py        # PyBullet environment
│   ├── es_simulation_trainer.py    # ES training integration
│   └── __init__.py
├── Dockerfile.pytorch-rocm         # ROCm-enabled Docker image
├── docker-compose.simulation.yml   # Multi-container setup
├── setup_simulation.sh             # Setup script
├── run_simulation.sh               # Run script
├── README_SIMULATION.md            # This file
└── dashboard/
    └── index.html                  # Web-based dashboard
```

## ⚙️ Configuration Options

### Environment Types

- `indoor`: Room-like environments with walls and obstacles
- `outdoor`: Open areas with random obstacles
- `mixed`: Combination of indoor and outdoor elements

### Training Parameters

- `--population-size`: ES population size (default: 10)
- `--sigma`: Parameter perturbation strength (default: 0.1)
- `--learning-rate`: Parameter update rate (default: 0.01)
- `--max-generations`: Training duration (default: 100)
- `--no-gui`: Run without visualization

## 🧠 ES Training Process

1. **Population Initialization**: Create parameter variations
2. **Fitness Evaluation**: Test each variation in simulation
3. **Natural Selection**: Select best performers
4. **Parameter Update**: Evolve toward better solutions
5. **Model Export**: Convert to ONNX/RKNN for robot deployment

## 📊 Monitoring & Visualization

### Real-time Dashboard Features

- Generation progress tracking
- Fitness score monitoring
- GPU/CPU resource usage
- Training parameter controls
- Live training logs
- Performance charts

### PyBullet Visualization

- 3D robot and environment view
- Real-time sensor data display
- Robot trajectory tracking
- Collision detection visualization

## 🏆 Training Optimization

### ROCm Acceleration

- Parallel fitness evaluation
- GPU-accelerated neural networks
- Efficient memory management
- Hardware-optimized PyTorch

### Adaptive Training

- Dynamic sigma adjustment
- Population diversity maintenance
- Convergence detection
- Automatic early stopping

## 📦 Output & Model Management

### Generated Files

- **Models**: `models/simulation/` - Trained PyTorch models
- **Logs**: `logs/simulation/` - Training statistics and metrics
- **Data**: `sim_data/` - Simulation datasets and buffers

### Model Formats

- **PyTorch**: Full model with training state
- **ONNX**: Portable format for conversion
- **RKNN**: Optimized for robot NPU deployment

## 🛠️ Development Workflow

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build Docker image
docker build -f Dockerfile.pytorch-rocm -t tractor-simulation:latest .
```

### 2. Custom Environment Creation

Modify `bullet_simulation.py` to add:
- New obstacle types
- Different robot models
- Custom sensors
- Advanced physics properties

### 3. Training Customization

Modify `es_simulation_trainer.py` to adjust:
- Reward functions
- Training parameters
- Evaluation metrics
- Convergence criteria

## 🔧 Troubleshooting

### Common Issues

1. **Docker Permission Errors**
   ```bash
   sudo usermod -aG docker $USER
   # Then log out and back in
   ```

2. **ROCm Not Detected**
   ```bash
   # Check ROCm installation
   rocminfo
   
   # Verify GPU drivers
   lsmod | grep amdgpu
   ```

3. **GUI Not Working**
   ```bash
   # Allow Docker to access X11
   xhost +local:docker
   ```

### Performance Tuning

- **Increase Population Size**: Better exploration but slower training
- **Adjust Sigma**: Higher values = more exploration, lower = exploitation
- **Modify Learning Rate**: Faster convergence vs. stability trade-off
- **Use Headless Mode**: For maximum training speed

## 📈 Best Practices

### Training Recommendations

1. **Start Simple**: Begin with indoor environments
2. **Monitor Convergence**: Watch for fitness plateaus
3. **Save Regularly**: Models are saved every 10 generations
4. **Experiment with Parameters**: Find optimal settings for your use case

### Model Deployment

1. **Export to ONNX**: Automatic conversion after training
2. **Convert to RKNN**: Use RKNN toolkit for robot deployment
3. **Test on Hardware**: Validate performance on actual robot
4. **Iterate**: Use real-world performance to improve simulation

## 🤝 Integration with ROS2 Rover

The trained models can be directly used with your existing ROS2 rover system:

1. **Model Conversion**: ONNX → RKNN using existing scripts
2. **Deployment**: Copy RKNN model to robot
3. **Inference**: Use existing NPU exploration nodes
4. **Validation**: Test in real-world scenarios

## 📚 Additional Resources

- [PyBullet Documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)
- [PyTorch ROCm Support](https://pytorch.org/blog/pytorch-for-amd-rocm-platform/)
- [Evolutionary Strategies Paper](https://arxiv.org/abs/1703.03864)

## 📞 Support

For issues or questions, please:
1. Check the existing documentation
2. Review error logs in `logs/simulation/`
3. Open an issue with detailed information
