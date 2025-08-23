#!/bin/bash

# Setup script for ES Training Simulation Environment
# Prepares Docker environment and runs simulation

set -e  # Exit on any error

echo "🚀 Setting up ES Training Simulation Environment"
echo "=============================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠ Warning: This setup is optimized for Linux systems with ROCm support"
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker Compose found: $(docker-compose --version)"

# Check ROCm support
echo "Checking ROCm support..."
if lsmod | grep -q amdgpu; then
    echo "✓ AMD GPU drivers detected"
else
    echo "⚠ AMD GPU drivers not detected. ROCm acceleration may not work."
fi

# Check for ROCm packages
if command -v rocminfo &> /dev/null; then
    echo "✓ ROCm tools found"
    rocminfo | head -20
else
    echo "⚠ ROCm tools not found. Please install ROCm 6.4.1 for full acceleration support."
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models/simulation
mkdir -p logs/simulation
mkdir -p sim_data
mkdir -p dashboard

echo "✓ Directories created"

# Check if user is in docker group
if groups $USER | grep -q docker; then
    echo "✓ User is in docker group"
else
    echo "⚠ User is not in docker group. You may need to run with sudo or add user to docker group:"
    echo "  sudo usermod -aG docker $USER"
    echo "  (Then log out and back in)"
fi

# Build Docker images
echo "Building Docker images..."
echo "This may take 10-30 minutes depending on your internet connection and system performance."

# Build the main simulation image
docker build -f Dockerfile.pytorch-rocm -t tractor-simulation:latest .

echo "✓ Docker images built successfully"

# Test the setup
echo "Running basic test..."
docker run --rm tractor-simulation:latest python -c "
import torch
import pybullet
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('ROCm available:', hasattr(torch.version, 'hip') and torch.version.hip is not None)
if hasattr(torch.version, 'hip'):
    print('ROCm version:', torch.version.hip)
print('PyBullet imported successfully')
print('Setup test completed successfully!')
"

echo "🎉 Setup completed successfully!"

echo ""
echo "To run the simulation:"
echo "  ./run_simulation.sh"
echo ""
echo "To run with specific parameters:"
echo "  docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add=video \\"
echo "    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=\\$DISPLAY \\"
echo "    tractor-simulation:latest \\"
echo "    python src/tractor_simulation/tractor_simulation/es_simulation_trainer.py --help"
echo ""
echo "For headless training:"
echo "  docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add=video \\"
echo "    tractor-simulation:latest \\"
echo "    python src/tractor_simulation/tractor_simulation/es_simulation_trainer.py --no-gui"
