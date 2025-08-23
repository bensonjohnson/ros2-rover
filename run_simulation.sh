#!/bin/bash

# Run script for ES Training Simulation
# Launches the simulation with proper ROCm and GUI support

set -e  # Exit on any error

echo "🚜 Running ES Training Simulation"
echo "================================="

# Default parameters
GUI_ENABLED=true
ENVIRONMENT="indoor"
POPULATION_SIZE=10
FORCE_SEQUENTIAL=false
MAX_GENERATIONS=100
SIGMA=0.1
LEARNING_RATE=0.01

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-gui)
            GUI_ENABLED=false
            shift
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --population-size)
            POPULATION_SIZE="$2"
            shift 2
            ;;
        --max-generations)
            MAX_GENERATIONS="$2"
            shift 2
            ;;
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --force-sequential)
            FORCE_SEQUENTIAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-gui              Run without GUI (headless)"
            echo "  --environment TYPE    Environment type: indoor, outdoor, mixed (default: indoor)"
            echo "  --population-size N   ES population size (default: 10)"
            echo "  --max-generations N   Maximum generations (default: 100)"
            echo "  --sigma VALUE         ES sigma parameter (default: 0.1)"
            echo "  --learning-rate VALUE Learning rate (default: 0.01)"
            echo "  --force-sequential    Disable parallel processing for debugging"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Docker image exists
if ! docker image inspect tractor-simulation:latest &>/dev/null; then
    echo "❌ Simulation Docker image not found. Please run setup first:"
    echo "  ./setup_simulation.sh"
    exit 1
fi

echo "✓ Using tractor-simulation:latest Docker image"

# Prepare Docker run arguments
DOCKER_ARGS=(
    "docker" "run" "-it" "--rm"
    "--name" "tractor-simulation-run"
    "--device=/dev/kfd"
    "--device=/dev/dri"
    "--group-add=video"
    "--security-opt" "seccomp=unconfined"
    "--cap-add=SYS_PTRACE"
    "--shm-size=2g"
    "--ulimit" "memlock=-1"
    "--ulimit" "stack=67108864"
    "-e" "HIP_VISIBLE_DEVICES=0"
    "-e" "HSA_OVERRIDE_GFX_VERSION=10.3.0"
    "-e" "ROCM_VERSION=5.4"
    "-e" "AMD_SERIALIZE_KERNEL=1"
    "-e" "TORCH_USE_HIP_DSA=1"
    "-e" "HIP_LAUNCH_BLOCKING=1"
    "-e" "GPU_MAX_ALLOC_PERCENT=90"
    "-e" "HSA_ENABLE_SDMA=0"
    "-e" "CUDA_LAUNCH_BLOCKING=1"
    "-e" "PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128"
    "-e" "HIP_FORCE_DEV_KERNARG=1"
    "-e" "HIP_DB=0x1"
    "-e" "HIP_TRACE_API=1"
    "-e" "PYTORCH_DISABLE_CUDA_MEMORY_CACHING=1"
    "-v" "$(pwd)/models:/workspace/models"
    "-v" "$(pwd)/logs:/workspace/logs"
    "-v" "$(pwd)/sim_data:/workspace/sim_data"
)

# Add GUI support if enabled
if [ "$GUI_ENABLED" = true ]; then
    echo "✓ GUI enabled - connecting to display ${DISPLAY:-:0}"
    DOCKER_ARGS+=(
        "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
        "-e" "DISPLAY=${DISPLAY:-:0}"
        "-e" "LIBGL_ALWAYS_INDIRECT=0"
        "-e" "LIBGL_ALWAYS_SOFTWARE=0"
    )
else
    echo "✓ Running in headless mode"
fi

# Add the image and command
DOCKER_ARGS+=(
    "tractor-simulation:latest"
    "python" "src/tractor_simulation/tractor_simulation/es_simulation_trainer.py"
    "--population-size" "$POPULATION_SIZE"
    "--sigma" "$SIGMA"
    "--learning-rate" "$LEARNING_RATE"
    "--max-generations" "$MAX_GENERATIONS"
    "--environment" "$ENVIRONMENT"
)

# Add GUI flag if disabled
if [ "$GUI_ENABLED" = false ]; then
    DOCKER_ARGS+=("--no-gui")
fi

# Add sequential flag if enabled
if [ "$FORCE_SEQUENTIAL" = true ]; then
    DOCKER_ARGS+=("--force-sequential")
fi


# Show the command that will be executed
echo ""
echo "Executing command:"
echo "  ${DOCKER_ARGS[*]}"
echo ""

# Run the simulation
echo "🚀 Starting simulation..."
echo "========================"

"${DOCKER_ARGS[@]}"

echo ""
echo "🏁 Simulation completed!"
echo "======================="

# Show where results are stored
echo ""
echo "Results are stored in:"
echo "  - Models: $(pwd)/models/simulation/"
echo "  - Logs: $(pwd)/logs/simulation/"
echo "  - Simulation data: $(pwd)/sim_data/"
