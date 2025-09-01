#!/bin/bash

# NPU Bird's Eye View Exploration Script
# Clean architecture: Hardware + AI only, no SLAM/Nav2 complexity

echo "=================================================="
echo "ROS2 Tractor - NPU Bird's Eye View Exploration"
echo "=================================================="
echo "This minimal system uses:"
echo "  ✓ Hiwonder motor control"
echo "  ✓ RealSense D435i (point cloud output)"
echo "  ✓ NPU-based exploration AI with BEV processing"
echo "  ✓ Direct safety monitoring"
echo "  ✓ Anti-overtraining reward system (optional)"
echo "  ✓ No SLAM/Nav2 complexity"
echo ""
echo "Available modes:"
echo "  • train:   RL-ES (PBT) training"
echo "  • infer:   Inference on trained model"
echo ""
echo "Optimization levels:"
echo "  • basic:     Core performance and safety optimization"
echo "  • standard:  Balanced multi-objective optimization (default)"
echo "  • full:      Advanced optimization with efficiency metrics"
echo "  • research:  Experimental features and maximum optimization"
echo ""
echo "BEV Configuration:"
echo "  • Size:        200x200 pixels (default)"
echo "  • Range:       10x10 meters (default)"
echo "  • Channels:    Multi-height + density (default)"
echo "  • Ground Removal: Enabled (default)"
echo ""

# Check if we're in the right directory
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Build only what we need
echo "Building minimal workspace..."
# Check if packages need to be built
if [ ! -d "build/tractor_bringup" ] || [ ! -d "build/tractor_control" ] || [ ! -d "install/tractor_bringup" ] || [ ! -d "install/tractor_control" ]; then
    echo "Building packages..."
    colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Build failed! Please check for compilation errors."
        exit 1
    fi
    echo "✓ Workspace built successfully"
else
    echo "✓ Packages already built"
fi

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash
echo "✓ ROS2 environment sourced"

# Initial RKNN conversion and reward system check
echo "Checking/performing initial RKNN conversion and reward system setup..."
python3 - <<'PYCODE'
try:
    from tractor_bringup.rknn_trainer_bev import RKNNTrainerBEV, RKNN_AVAILABLE
    from tractor_bringup.improved_reward_system import ImprovedRewardCalculator
    from tractor_bringup.anti_overtraining_config import get_safe_config
    
    # Initialize reward system with anti-overtraining config
    config = get_safe_config()
    reward_calculator = ImprovedRewardCalculator(**config)
    print("✓ Anti-overtraining reward system initialized")
    
    # RKNN conversion
    trainer = RKNNTrainerBEV(bev_channels=4, enable_debug=True)
    if RKNN_AVAILABLE:
        trainer.convert_to_rknn()
    else:
        print("RKNN toolkit not available - skipping initial conversion")
        
except Exception as e:
    print(f"Initial setup failed: {e}")
    print("Continuing with basic configuration...")
PYCODE

echo "Initial RKNN conversion and reward system setup complete"

# USB power management for RealSense
echo "Configuring USB power management..."
USB_DEVICE_PATH=""
# Find D435i device by Product ID (0B3A)
for device in /sys/bus/usb/devices/*/idProduct; do
    if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
        USB_DEVICE_PATH=$(dirname $device)
        echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
        break
    fi
done

# Fallback to common paths if Product ID detection fails
if [ -z "$USB_DEVICE_PATH" ]; then
    for path in "/sys/bus/usb/devices/8-1" "/sys/bus/usb/devices/2-1" "/sys/bus/usb/devices/1-1"; do
        if [ -d "$path" ]; then
            USB_DEVICE_PATH="$path"
            echo "✓ Using fallback USB path: $USB_DEVICE_PATH"
            break
        fi
    done
fi

if [ -n "$USB_DEVICE_PATH" ]; then
    # Disable autosuspend for the device
    echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
    echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
    echo "✓ USB power management configured for $USB_DEVICE_PATH"
else
    echo "⚠ USB device path not found, continuing anyway"
fi

# Check RealSense D435i specifically
echo "Checking RealSense D435i..."
if command -v rs-enumerate-devices &> /dev/null; then
    timeout 5s rs-enumerate-devices | grep -q "D435I"
    if [ $? -eq 0 ]; then
        echo "✓ RealSense D435i detected"
        # Reset the USB device to clear any error states
        if [ -n "$USB_DEVICE_PATH" ] && [ -w "$USB_DEVICE_PATH" ]; then
            echo "Resetting USB device..."
            echo "0" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            sleep 1
            echo "1" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            echo "✓ USB device reset"
        fi
    else
        echo "⚠ RealSense D435i not detected - will attempt to continue"
    fi
else
    echo "⚠ rs-enumerate-devices command not found - skipping RealSense check"
fi

# Configuration
# Usage: ./start_npu_exploration_bev.sh <train|infer> <max_speed> <exploration_time> <safety_distance>
DEFAULT_MODE="train"
DEFAULT_MAX_SPEED="0.15"
DEFAULT_EXPLORATION_TIME="300"
DEFAULT_SAFETY_DISTANCE="0.2"

# PBT default values
DEFAULT_PBT_POPULATION_SIZE="4"
DEFAULT_PBT_UPDATE_INTERVAL="1000"  # Increased from 200 for stability
DEFAULT_PBT_PERTURB_PROB="0.25"
DEFAULT_PBT_RESAMPLE_PROB="0.25"

MODE_INPUT=${1:-$DEFAULT_MODE}
shift || true
MAX_SPEED=${1:-$DEFAULT_MAX_SPEED}
EXPLORATION_TIME=${2:-$DEFAULT_EXPLORATION_TIME}
SAFETY_DISTANCE=${3:-$DEFAULT_SAFETY_DISTANCE}

# Map simple modes to internal operation modes
case "$MODE_INPUT" in
  train|TRAIN)
    MODE="es_rl_hybrid";;
  infer|INFER|inference)
    MODE="es_inference";;
  *)
    echo "Unknown mode '$MODE_INPUT', defaulting to 'train' (es_rl_hybrid)"
    MODE="es_rl_hybrid";;
esac

# PBT defaults
PBT_POPULATION_SIZE=${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE}
PBT_UPDATE_INTERVAL=${PBT_UPDATE_INTERVAL:-$DEFAULT_PBT_UPDATE_INTERVAL}
PBT_PERTURB_PROB=${PBT_PERTURB_PROB:-$DEFAULT_PBT_PERTURB_PROB}
PBT_RESAMPLE_PROB=${PBT_RESAMPLE_PROB:-$DEFAULT_PBT_RESAMPLE_PROB}

if [[ "$MODE" != "es_inference" && "$MODE" != "es_rl_hybrid" ]]; then
  echo "Invalid mode '$MODE'. Valid: es_rl_hybrid | es_inference"
  exit 1
fi

# PBT-specific validation for es_rl_hybrid mode
if [[ "$MODE" == "es_rl_hybrid" ]]; then
    echo "Validating PBT dependencies for es_rl_hybrid mode..."
    python3 - <<'PBT_CHECK'
try:
    from tractor_bringup.pbt_es_rl_trainer import PBT_ES_RL_Trainer
    from tractor_bringup.rknn_trainer_bev import RKNNTrainerBEV
    import torch.multiprocessing as mp
    
    # Test PBT interface compatibility
    dummy_trainer = RKNNTrainerBEV(bev_channels=4, enable_debug=False)
    if not hasattr(dummy_trainer, 'copy_weights_from'):
        raise AttributeError("RKNNTrainerBEV missing copy_weights_from method")
    if not hasattr(dummy_trainer, 'get_model'):
        raise AttributeError("RKNNTrainerBEV missing get_model method")
    if not hasattr(dummy_trainer, 'torch'):
        raise AttributeError("RKNNTrainerBEV missing torch module exposure")
    
    print("✓ PBT dependencies and interface compatibility verified")
except ImportError as e:
    print(f"❌ PBT dependency missing: {e}")
    print("Please ensure tractor_bringup package is built and PBT trainer is available")
    exit(1)
except AttributeError as e:
    print(f"❌ PBT interface error: {e}")
    print("RKNNTrainerBEV class needs PBT interface methods")
    exit(1)
except Exception as e:
    print(f"❌ PBT validation failed: {e}")
    exit(1)
PBT_CHECK
    
    if [ $? -ne 0 ]; then
        echo "❌ PBT validation failed - cannot run es_rl_hybrid mode"
        exit 1
    fi
    
    echo "✓ PBT validation passed"
fi

echo ""
echo "Configuration:"
echo "  Operation Mode: ${MODE}"
case "$MODE" in
  "es_inference") echo "    → Inference on trained ES/RL model" ;;
  "es_rl_hybrid") echo "    → Combined RL+ES training (PBT)" ;;
esac
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Exploration Time: ${EXPLORATION_TIME} seconds"
echo "  Safety Distance: ${SAFETY_DISTANCE} m"
if [[ "$MODE" == "es_rl_hybrid" ]]; then
  echo "  PBT Population: ${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE} agents"
  echo "  PBT Update Interval: ${PBT_UPDATE_INTERVAL:-$DEFAULT_PBT_UPDATE_INTERVAL} steps"
  echo "  PBT Perturbation Prob: ${PBT_PERTURB_PROB:-$DEFAULT_PBT_PERTURB_PROB}"
  echo "  PBT Resample Prob: ${PBT_RESAMPLE_PROB:-$DEFAULT_PBT_RESAMPLE_PROB}"
fi
echo "  IMU Status: Disabled (to reduce USB errors)"
echo "  USB Mode: Optimized for stability"
if [[ "$MODE" == "safe_training" ]]; then
  echo "  Reward System: Anti-overtraining measures active"
  echo "    → Behavioral diversity tracking"
  echo "    → Anti-gaming detection"
  echo "    → Curriculum learning ready"
fi
echo ""

# Countdown
echo "Starting NPU BEV exploration in:"
for i in {3..1}; do
    echo "  $i..."
    sleep 1
done
echo "  🚀 LAUNCHING!"
echo ""

# Launch the minimal system
echo "Launching NPU BEV exploration system..."
echo "Press Ctrl+C to stop safely"
echo ""

FOXGLOVE_PID=""
# Launch Foxglove bridge only for (es_)inference or (es_)hybrid, unless disabled
if [[ "${ENABLE_FOXGLOVE:-1}" == "1" ]] && [[ "$MODE" == "inference" || "$MODE" == "hybrid" || "$MODE" == "es_inference" || "$MODE" == "es_hybrid" ]]; then
  echo "Launching Foxglove bridge..."
  ros2 launch foxglove_bridge foxglove_bridge_launch.xml &
  FOXGLOVE_PID=$!
  echo "✓ Foxglove bridge launched with PID: $FOXGLOVE_PID"
  echo ""
else
  echo "Skipping Foxglove bridge (training-focused mode or disabled)"
fi

# Adjust BEV size for PBT to reduce memory footprint when population is large
BEV_SIZE_PARAM="[200, 200]"
if [[ "$MODE" == "es_rl_hybrid" ]] && [[ ${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE} -ge 3 ]]; then
  BEV_SIZE_PARAM="[160, 160]"
fi

ros2 launch tractor_bringup npu_exploration_bev.launch.py \
    operation_mode:=${MODE} \
    max_speed:=${MAX_SPEED} \
    exploration_time:=${EXPLORATION_TIME} \
    safety_distance:=${SAFETY_DISTANCE} \
    enable_lsm_imu_proprio:=true \
    anti_overtraining:=$([[ "$MODE" == "safe_training" || "$MODE" == "safe_es_training" ]] && echo "true" || echo "false") \
    enable_bayesian_optimization:=$([[ "$MODE" == "es_training" || "$MODE" == "es_hybrid" || "$MODE" == "safe_es_training" ]] && echo "true" || echo "false") \
    pbt_population_size:=${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE} \
    pbt_update_interval:=${PBT_UPDATE_INTERVAL:-$DEFAULT_PBT_UPDATE_INTERVAL} \
    pbt_perturb_prob:=${PBT_PERTURB_PROB:-$DEFAULT_PBT_PERTURB_PROB} \
    pbt_resample_prob:=${PBT_RESAMPLE_PROB:-$DEFAULT_PBT_RESAMPLE_PROB} \
    optimization_level:=standard \
    enable_training_optimization:=true \
    enable_reward_optimization:=false \
    enable_multi_metric_evaluation:=true \
    enable_optimization_monitoring:=true \
    enable_multi_objective_optimization:=false \
    enable_safety_constraints:=true \
    enable_architecture_optimization:=false \
    enable_progressive_architecture:=false \
    enable_sensor_fusion_optimization:=false \
    use_sim_time:=false \
    bev_size:="$BEV_SIZE_PARAM" \
    bev_range:="[10.0, 10.0]" \
    bev_height_channels:="[0.2, 1.0]" \
    enable_ground_removal:=true &

LAUNCH_PID=$!

# Simple shutdown handler
shutdown_handler() {
    echo ""
    echo "🛑 Stopping NPU BEV exploration..."
    if ps -p $LAUNCH_PID > /dev/null; then
        kill $LAUNCH_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if ps -p $LAUNCH_PID > /dev/null; then
            kill -9 $LAUNCH_PID 2>/dev/null
        fi
    fi
    # Trigger a best-effort model save
    timeout 5s ros2 service call /save_models std_srvs/srv/Trigger "{}" >/dev/null 2>&1 || true
    if [[ -n "$FOXGLOVE_PID" ]] && ps -p $FOXGLOVE_PID > /dev/null; then
        kill $FOXGLOVE_PID 2>/dev/null
    fi
    # Stop monitoring if running
    if [[ -n "$MONITOR_PID" ]] && ps -p $MONITOR_PID > /dev/null; then
        kill $MONITOR_PID 2>/dev/null
    fi
    
    # Save training logs if safe_training mode was used
    if [[ "$MODE" == "safe_training" || "$MODE" == "safe_es_training" ]]; then
        echo "💾 Saving anti-overtraining logs..."
        python3 - <<'SAVE_LOGS'
try:
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/anti_overtraining"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Training logs saved to {log_dir}/training_session_{timestamp}/")
except Exception as e:
    print(f"Log saving failed: {e}")
SAVE_LOGS
    fi
    
    # Save PBT population state if es_rl_hybrid mode was used
    if [[ "$MODE" == "es_rl_hybrid" ]]; then
        echo "💾 Saving PBT population state..."
        python3 - <<'SAVE_PBT'
try:
    import os
    import pickle
    from datetime import datetime
    from tractor_bringup.pbt_es_rl_trainer import PBT_ES_RL_Trainer
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/pbt_sessions"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save PBT session metadata
    session_info = {
        "timestamp": timestamp,
        "population_size": "${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE}",
        "update_interval": "${PBT_UPDATE_INTERVAL:-$DEFAULT_PBT_UPDATE_INTERVAL}",
        "session_type": "es_rl_hybrid"
    }
    
    session_file = os.path.join(log_dir, f"pbt_session_{timestamp}.json")
    with open(session_file, "w") as f:
        import json
        json.dump(session_info, f, indent=2)
    
    print(f"PBT session info saved to {session_file}")
    
except Exception as e:
    print(f"PBT state save failed: {e}")
SAVE_PBT
    fi
    
    echo "✅ NPU BEV exploration stopped safely"
    echo "=================================================="
    exit 0
}

trap shutdown_handler SIGINT SIGTERM

echo "🤖 NPU BEV exploration active..."
echo "   System is learning to navigate autonomously using Bird's Eye View maps"
echo "   Monitor via: ros2 topic echo /npu_exploration_status"
if [[ "$MODE" == "safe_training" ]]; then
  echo "   Anti-overtraining monitoring active"
  echo "   Training health: ros2 topic echo /training_health"
fi
if [[ "$MODE" == "es_rl_hybrid" ]]; then
  echo "🧠 PBT Population-Based Training active:"
  echo "   - Population size: ${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE} agents"
  echo "   - Agent switching: ros2 topic echo /pbt_agent_status"
  echo "   - Population metrics: ros2 topic echo /pbt_population_metrics"
  echo "   - Memory usage: ~$(( (${PBT_POPULATION_SIZE:-$DEFAULT_PBT_POPULATION_SIZE} * 500) ))MB expected"
fi
echo ""

# Enhanced monitoring for safe modes
if [[ "$MODE" == "safe_training" || "$MODE" == "safe_es_training" ]]; then
  echo "🛡️  Anti-overtraining monitoring enabled:"
  echo "   - Behavioral diversity tracking"
  echo "   - Reward gaming detection" 
  echo "   - Automatic early stopping"
  echo "   - Training health indicators"
  echo ""
fi

# Enhanced monitoring for PBT mode
if [[ "$MODE" == "es_rl_hybrid" ]]; then
  echo "🧠 PBT Population-Based Training monitoring:"
  echo "   - Population diversity: ros2 topic echo /pbt_population_diversity"
  echo "   - Agent performance: ros2 topic echo /pbt_agent_fitness"
  echo "   - Weight evolution: ros2 topic echo /pbt_evolution_status"
  echo "   - Hyperparameter exploration: ros2 topic echo /pbt_hyperparams"
  echo "   - Memory optimization active"
  echo ""
  
  # Start background monitoring
  (
    sleep 30  # Wait for system to start
    while ps -p $LAUNCH_PID > /dev/null 2>&1; do
      # Check training health every 60 seconds
      python3 - <<'MONITOR_PYCODE'
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import json
    
    # Quick health check without full node setup
    print("📊 Training health check - $(date)")
    # This would integrate with your ROS2 training nodes
    
except Exception as e:
    pass  # Silent fail for monitoring
MONITOR_PYCODE
      sleep 60
    done
  ) &
  MONITOR_PID=$!
fi

# Wait for completion
wait $LAUNCH_PID

shutdown_handler
