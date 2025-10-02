#!/bin/bash

set -euo pipefail

DEFAULT_MAX_SPEED="0.18"
DEFAULT_SAFETY_DISTANCE="0.25"
DEFAULT_BUILD="on"
DEFAULT_BAG_REPLAY="off"
DEFAULT_BAG_PATH=""
DEFAULT_BAG_RATE="1.0"
DEFAULT_BAG_LOOP="false"
DEFAULT_LOG_DIR="log"
DEFAULT_OBS_HEIGHT="128"
DEFAULT_OBS_WIDTH="128"
DEFAULT_PPO_UPDATE_INTERVAL="20.0"

MAX_SPEED="$DEFAULT_MAX_SPEED"
SAFETY_DISTANCE="$DEFAULT_SAFETY_DISTANCE"
DO_BUILD="$DEFAULT_BUILD"
BAG_REPLAY="$DEFAULT_BAG_REPLAY"
BAG_PATH="$DEFAULT_BAG_PATH"
BAG_RATE="$DEFAULT_BAG_RATE"
BAG_LOOP="$DEFAULT_BAG_LOOP"
LOG_DIR="$DEFAULT_LOG_DIR"
PPO_UPDATE_INTERVAL="$DEFAULT_PPO_UPDATE_INTERVAL"
OBS_HEIGHT="$DEFAULT_OBS_HEIGHT"
OBS_WIDTH="$DEFAULT_OBS_WIDTH"
SESSION_MODE="explore"  # explore | train

ensure_workspace() {
  if [ ! -f "install/setup.bash" ]; then
    echo "Error: run this script from the ros2-rover workspace root"
    echo "Current directory: $(pwd)"
    exit 1
  fi
}

toggle_build() {
  if [ "$DO_BUILD" = "on" ]; then
    DO_BUILD="off"
  else
    DO_BUILD="on"
  fi
}

toggle_bag_replay() {
  if [ "$BAG_REPLAY" = "on" ]; then
    BAG_REPLAY="off"
  else
    BAG_REPLAY="on"
  fi
}

prompt_value() {
  local prompt="$1"
  local current="$2"
  local varname="$3"
  read -r -p "$prompt [$current]: " input
  if [ -n "$input" ]; then
    eval "$varname=\"$input\""
  fi
}

draw_menu() {
  clear
  cat <<MENU
==================================================
 ROS2 Tractor - RTAB Control Menu
==================================================
 (1) Max speed (m/s) ............... $MAX_SPEED
 (2) Safety distance (m) ........... $SAFETY_DISTANCE
 (3) Observation height (px) ....... $OBS_HEIGHT
 (4) Observation width (px) ........ $OBS_WIDTH
 (5) PPO update interval (s) ....... $PPO_UPDATE_INTERVAL
 (6) Colcon build on start ......... ${DO_BUILD^^}
 (7) Bag replay .................... ${BAG_REPLAY^^}
MENU
  if [ "$BAG_REPLAY" = "on" ]; then
    cat <<MENU
 (8) Bag path ..................... ${BAG_PATH:-<unset>}
 (9) Bag rate scale ................ $BAG_RATE
 (A) Bag loop playback ............ ${BAG_LOOP^^}
MENU
  fi
  cat <<MENU
 (S) Start exploration (live sensors)
 (T) Start offline training (bag replay)
 (Q) Quit without starting
--------------------------------------------------
 Select option: 
MENU
}

run_colcon() {
  echo "Building workspace (tractor_bringup + tractor_control)..."
  colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release
}

configure_realsense_usb() {
  echo "Configuring USB power management..."
  USB_DEVICE_PATH=""
  USB_HUB_PATH=""

  # Find D435i device by Product ID (0B3A)
  for device in /sys/bus/usb/devices/*/idProduct; do
    if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
      USB_DEVICE_PATH=$(dirname $device)
      echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
      # Extract hub path (e.g., 8-1 from 8-1-4)
      USB_HUB_PATH=$(echo "$USB_DEVICE_PATH" | grep -oP '/sys/bus/usb/devices/\d+-\d+' || echo "")
      break
    fi
  done

  # Fallback to common paths if Product ID detection fails
  if [ -z "$USB_DEVICE_PATH" ]; then
    for path in "/sys/bus/usb/devices/8-1-4" "/sys/bus/usb/devices/8-1-3" "/sys/bus/usb/devices/8-1-2" "/sys/bus/usb/devices/8-1"; do
      if [ -d "$path" ]; then
        USB_DEVICE_PATH="$path"
        USB_HUB_PATH="/sys/bus/usb/devices/8-1"
        echo "✓ Using fallback USB path: $USB_DEVICE_PATH"
        break
      fi
    done
  fi

  # Reset the USB hub first to clear all states
  if [ -n "$USB_HUB_PATH" ] && [ -d "$USB_HUB_PATH" ]; then
    echo "Resetting USB hub at $USB_HUB_PATH..."
    echo "0" | sudo tee $USB_HUB_PATH/authorized > /dev/null 2>&1
    sleep 1
    echo "1" | sudo tee $USB_HUB_PATH/authorized > /dev/null 2>&1
    sleep 3
    echo "✓ USB hub reset complete"
  fi

  # Configure power management for the camera device
  if [ -n "$USB_DEVICE_PATH" ]; then
    # Wait for device to re-enumerate after hub reset
    sleep 2

    # Disable autosuspend for the device
    echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
    echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
    echo "✓ USB power management configured for $USB_DEVICE_PATH"
  else
    echo "⚠ USB device path not found, continuing anyway"
  fi

  echo "Checking RealSense D435i..."
  if command -v rs-enumerate-devices &> /dev/null; then
    if timeout 15s rs-enumerate-devices 2>/dev/null | grep -q "D435I"; then
      echo "✓ RealSense D435i detected"
    else
      echo "⚠ RealSense D435i not detected - check USB connection"
    fi
  else
    echo "⚠ rs-enumerate-devices not found - install librealsense2-tools"
  fi
}

launch_exploration() {
  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/rtab_exploration_$(date +%Y%m%d_%H%M%S).log"

  # Configure RealSense USB before launching
  configure_realsense_usb

  # Temporarily disable strict error checking for ROS sourcing
  set +u
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  set -u

  echo "Launching RTAB exploration stack..."
  {
    printf '[INFO] %s starting RTAB exploration max_speed=%s safety_distance=%s obs=%sx%s ppo_update_interval=%s\n' "$(date)" "$MAX_SPEED" "$SAFETY_DISTANCE" "$OBS_HEIGHT" "$OBS_WIDTH" "$PPO_UPDATE_INTERVAL"
    ros2 launch tractor_bringup npu_exploration_ppo.launch.py \
      max_speed:=$MAX_SPEED \
      safety_distance:=$SAFETY_DISTANCE \
      observation_height:=$OBS_HEIGHT \
      observation_width:=$OBS_WIDTH \
      ppo_update_interval_sec:=$PPO_UPDATE_INTERVAL
  } | tee "$log_file"
}

launch_training() {
  # Temporarily disable strict error checking for ROS sourcing
  set +u
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  set -u

  echo "Launching RTAB offline training stack..."
  ros2 launch tractor_bringup rtab_offline_training.launch.py
}

start_bag_replay() {
  if [ "$BAG_REPLAY" != "on" ] || [ -z "$BAG_PATH" ]; then
    BAG_REPLAY_PID=""
    return
  fi
  echo "Starting bag replay from $BAG_PATH (rate=$BAG_RATE loop=$BAG_LOOP)"
  ros2 run tractor_bringup rtab_bag_replay.py --ros-args \
    -p bag_path:="$BAG_PATH" \
    -p rate_scale:="$BAG_RATE" \
    -p loop:="$BAG_LOOP" \
    >/tmp/rtab_bag_replay.log 2>&1 &
  BAG_REPLAY_PID=$!
  sleep 1
  if ! ps -p "$BAG_REPLAY_PID" > /dev/null 2>&1; then
    echo "⚠ Failed to start bag replay; check /tmp/rtab_bag_replay.log"
    BAG_REPLAY_PID=""
  fi
}

stop_bag_replay() {
  if [ -n "$1" ] && ps -p "$1" > /dev/null 2>&1; then
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
  fi
}

ensure_workspace

while true; do
  draw_menu
  read -r choice
  case "${choice^^}" in
    1)
      prompt_value "Enter max speed" "$MAX_SPEED" MAX_SPEED
      ;;
    2)
      prompt_value "Enter safety distance" "$SAFETY_DISTANCE" SAFETY_DISTANCE
      ;;
    3)
      prompt_value "Enter observation height (pixels)" "$OBS_HEIGHT" OBS_HEIGHT
      ;;
    4)
      prompt_value "Enter observation width (pixels)" "$OBS_WIDTH" OBS_WIDTH
      ;;
    5)
      prompt_value "Enter PPO update interval" "$PPO_UPDATE_INTERVAL" PPO_UPDATE_INTERVAL
      ;;
    6)
      toggle_build
      ;;
    7)
      toggle_bag_replay
      ;;
    8)
      if [ "$BAG_REPLAY" = "on" ]; then
        prompt_value "Enter bag path" "$BAG_PATH" BAG_PATH
      fi
      ;;
    9)
      if [ "$BAG_REPLAY" = "on" ]; then
        prompt_value "Enter rate scale" "$BAG_RATE" BAG_RATE
      fi
      ;;
    A)
      if [ "$BAG_REPLAY" = "on" ]; then
        BAG_LOOP=$([ "$BAG_LOOP" = "true" ] && echo "false" || echo "true")
      fi
      ;;
    S)
      SESSION_MODE="explore"
      break
      ;;
    T)
      SESSION_MODE="train"
      if [ "$BAG_REPLAY" = "off" ]; then
        echo "Bag replay must be enabled for offline training."
        BAG_REPLAY="on"
      fi
      if [ -z "$BAG_PATH" ]; then
        prompt_value "Enter bag path" "$BAG_PATH" BAG_PATH
      fi
      if [ -z "$BAG_PATH" ]; then
        echo "Bag path required for training."
        sleep 1
        continue
      fi
      break
      ;;
    Q)
      echo "Exiting without starting."
      exit 0
      ;;
    *)
      ;;
  esac
  sleep 0.2
done

if [ "$DO_BUILD" = "on" ]; then
  run_colcon
else
  echo "Skipping colcon build"
fi

BAG_REPLAY_PID=""
trap 'echo; echo "🛑 Stopping session..."; [ -n "$MAIN_PID" ] && kill "$MAIN_PID" 2>/dev/null; stop_bag_replay "$BAG_REPLAY_PID"; exit 0' SIGINT SIGTERM

start_bag_replay

if [ "$SESSION_MODE" = "explore" ]; then
  launch_exploration &
else
  launch_training &
fi
MAIN_PID=$!
wait $MAIN_PID
EXIT_CODE=$?
stop_bag_replay "$BAG_REPLAY_PID"
exit $EXIT_CODE
