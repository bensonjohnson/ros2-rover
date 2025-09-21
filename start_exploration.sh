#!/bin/bash

set -e

DEFAULT_MODE="rtab"          # rtab | bev
DEFAULT_MAX_SPEED="0.18"
DEFAULT_SAFETY_DISTANCE="0.25"
DEFAULT_BUILD="on"
DEFAULT_BAG_REPLAY="off"
DEFAULT_BAG_PATH=""
DEFAULT_BAG_RATE="1.0"
DEFAULT_BAG_LOOP="false"
DEFAULT_LOG_DIR="log"

MODE="$DEFAULT_MODE"
MAX_SPEED="$DEFAULT_MAX_SPEED"
SAFETY_DISTANCE="$DEFAULT_SAFETY_DISTANCE"
DO_BUILD="$DEFAULT_BUILD"
BAG_REPLAY="$DEFAULT_BAG_REPLAY"
BAG_PATH="$DEFAULT_BAG_PATH"
BAG_RATE="$DEFAULT_BAG_RATE"
BAG_LOOP="$DEFAULT_BAG_LOOP"
LOG_DIR="$DEFAULT_LOG_DIR"

function ensure_workspace() {
  if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover workspace root"
    echo "Current directory: $(pwd)"
    exit 1
  fi
}

function toggle_mode() {
  if [ "$MODE" = "rtab" ]; then
    MODE="bev"
  else
    MODE="rtab"
  fi
}

function toggle_build() {
  if [ "$DO_BUILD" = "on" ]; then
    DO_BUILD="off"
  else
    DO_BUILD="on"
  fi
}

function toggle_bag_replay() {
  if [ "$BAG_REPLAY" = "on" ]; then
    BAG_REPLAY="off"
  else
    BAG_REPLAY="on"
  fi
}

function prompt_value() {
  local prompt="$1"
  local current="$2"
  local varname="$3"
  read -r -p "$prompt [$current]: " input
  if [ -n "$input" ]; then
    eval "$varname=\"$input\""
  fi
}

function draw_menu() {
  clear
  cat <<MENU
==================================================
 ROS2 Tractor - Exploration Launcher
==================================================
 (1) Exploration mode .............. ${MODE^^}
 (2) Max speed (m/s) ............... $MAX_SPEED
 (3) Safety distance (m) ........... $SAFETY_DISTANCE
 (4) Colcon build on start ......... ${DO_BUILD^^}
 (5) Bag replay .................... ${BAG_REPLAY^^}
MENU
  if [ "$BAG_REPLAY" = "on" ]; then
    cat <<MENU
  (6) Bag path ..................... ${BAG_PATH:-<unset>}
  (7) Bag rate scale ................ $BAG_RATE
  (8) Bag loop playback ............ ${BAG_LOOP^^}
MENU
  fi
  cat <<MENU
 (S) Start exploration
 (Q) Quit without starting
--------------------------------------------------
 Select option: 
MENU
}

function run_colcon() {
  echo "Building workspace (tractor_bringup + tractor_control)..."
  colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release
}

function launch_stack() {
  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/$(printf '%s_exploration_%s.log' "$MODE" "$(date +%Y%m%d_%H%M%S)")"

  source /opt/ros/jazzy/setup.bash
  source install/setup.bash

  local args=(
    ros2 launch tractor_bringup npu_exploration_ppo.launch.py
    max_speed:=$MAX_SPEED
    safety_distance:=$SAFETY_DISTANCE
  )

  if [ "$MODE" = "rtab" ]; then
    args+=(use_rtab_observation:=true)
  else
    args+=(use_rtab_observation:=false)
  fi

  echo "Launching exploration stack (mode=$MODE) ..."
  {
    printf '[INFO] %s starting exploration mode=%s max_speed=%s safety_distance=%s\n' "$(date)" "$MODE" "$MAX_SPEED" "$SAFETY_DISTANCE"
    "${args[@]}"
  } | tee "$log_file"
}

function start_bag_replay() {
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

function stop_bag_replay() {
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
      toggle_mode
      ;;
    2)
      prompt_value "Enter max speed" "$MAX_SPEED" MAX_SPEED
      ;;
    3)
      prompt_value "Enter safety distance" "$SAFETY_DISTANCE" SAFETY_DISTANCE
      ;;
    4)
      toggle_build
      ;;
    5)
      toggle_bag_replay
      ;;
    6)
      if [ "$BAG_REPLAY" = "on" ]; then
        prompt_value "Enter bag path" "$BAG_PATH" BAG_PATH
      fi
      ;;
    7)
      if [ "$BAG_REPLAY" = "on" ]; then
        prompt_value "Enter rate scale" "$BAG_RATE" BAG_RATE
      fi
      ;;
    8)
      if [ "$BAG_REPLAY" = "on" ]; then
        if [ "$BAG_LOOP" = "true" ]; then
          BAG_LOOP="false"
        else
          BAG_LOOP="true"
        fi
      fi
      ;;
    S)
      break
      ;;
    Q)
      echo "Exiting without starting."
      exit 0
      ;;
    *)
      # Additional shortcuts when bag replay enabled
      if [ "$BAG_REPLAY" = "on" ]; then
        case "${choice^^}" in
          P)
            prompt_value "Enter bag path" "$BAG_PATH" BAG_PATH
            ;;
          T)
            prompt_value "Topics (comma separated, optional)" "${TOPICS:-}" TOPICS
            ;;
          *)
            ;;
        esac
      fi
      ;;
  esac
  if [ "${choice^^}" = "" ]; then
    continue
  fi
  if [[ "${choice^^}" =~ ^[0-9A-Z]$ ]]; then
    continue
  fi
  sleep 0.2
done

if [ "$DO_BUILD" = "on" ]; then
  run_colcon
else
  echo "Skipping colcon build"
fi

BAG_REPLAY_PID=""
trap 'echo; echo "🛑 Stopping exploration..."; [ -n "$MAIN_PID" ] && kill "$MAIN_PID" 2>/dev/null; stop_bag_replay "$BAG_REPLAY_PID"; exit 0' SIGINT SIGTERM

start_bag_replay

launch_stack &
MAIN_PID=$!
wait $MAIN_PID
EXIT_CODE=$?
stop_bag_replay "$BAG_REPLAY_PID"
exit $EXIT_CODE
