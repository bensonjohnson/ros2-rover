#!/bin/bash
# Track polarity test
# Drives left-channel then right-channel commands so you can confirm
# which physical track responds to which logical channel.
# Prop the rover up first.

set -e

if [ ! -f "install/setup.bash" ]; then
  echo "Run from the ros2-rover directory."
  exit 1
fi

source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Starting motor driver..."
ros2 run tractor_control hiwonder_motor_driver > /tmp/track_polarity_motor.log 2>&1 &
DRIVER_PID=$!
cleanup() {
  echo "Stopping..."
  ros2 topic pub --once /track_cmd std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0]}" >/dev/null 2>&1 || true
  kill "$DRIVER_PID" 2>/dev/null || true
  wait "$DRIVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the driver to come up and the /track_cmd topic to exist.
for i in $(seq 1 20); do
  if ros2 topic list 2>/dev/null | grep -q "^/track_cmd$"; then
    break
  fi
  sleep 0.5
done
if ! ros2 topic list 2>/dev/null | grep -q "^/track_cmd$"; then
  echo "Motor driver did not come up. Log:"
  tail -30 /tmp/track_polarity_motor.log
  exit 1
fi
echo "Motor driver up. Topic /track_cmd is live."
sleep 1

run_cmd() {
  local label="$1"
  local data="$2"
  echo
  echo "=== $label ==="
  echo "Publishing $data for 3 seconds. Watch the rover."
  end=$(( SECONDS + 3 ))
  while [ $SECONDS -lt $end ]; do
    ros2 topic pub --once /track_cmd std_msgs/msg/Float32MultiArray "{data: $data}" >/dev/null 2>&1
    sleep 0.1
  done
  ros2 topic pub --once /track_cmd std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0]}" >/dev/null 2>&1
  sleep 1
}

run_cmd "TEST 1  data=[+0.6, 0.0]  (left channel only, forward)"  "[0.6, 0.0]"
run_cmd "TEST 2  data=[0.0, +0.6]  (right channel only, forward)" "[0.0, 0.6]"
run_cmd "TEST 3  data=[+0.6, -0.6] (in-place turn, channel 0 fwd, channel 1 rev)" "[0.6, -0.6]"

echo
echo "Done. Expected results:"
echo "  TEST 1: physical LEFT track spins forward  -> labels correct"
echo "  TEST 2: physical RIGHT track spins forward -> labels correct"
echo "  TEST 3: rover should rotate to the RIGHT (CW from above) -> labels correct"
echo
echo "If any of the above produced the opposite physical motion,"
echo "set invert_steering:=True in the NoMaD launch (or swap the cables)."
