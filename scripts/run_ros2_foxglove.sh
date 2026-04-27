#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43}"
MAX_SPEED="${2:-2.0}"
SCAN_TOPIC="${SCAN_TOPIC:-/scan}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"
DRIVE_TOPIC="${DRIVE_TOPIC:-/drive}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"

cd "$ROOT_DIR"

echo "[team7] repo: $ROOT_DIR"
echo "[team7] run:  $RUN_DIR"
echo "[team7] cap:  ${MAX_SPEED} m/s"
echo "[team7] topics: scan=${SCAN_TOPIC} odom=${ODOM_TOPIC} drive=${DRIVE_TOPIC}"

if [[ ! -f "$RUN_DIR/final_model.zip" ]]; then
  echo "Could not find model: $RUN_DIR/final_model.zip" >&2
  exit 1
fi

if [[ ! -f "$RUN_DIR/config.yaml" ]]; then
  echo "Could not find config: $RUN_DIR/config.yaml" >&2
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is not on PATH. Source ROS2 first, for example:" >&2
  echo "  source /opt/ros/humble/setup.bash" >&2
  exit 1
fi

echo "[team7] installing local Python package if needed..."
python3 -m pip install -e . >/tmp/team7_f1tenth_rl_pip.log 2>&1 || {
  cat /tmp/team7_f1tenth_rl_pip.log >&2
  exit 1
}

echo "[team7] launching RL inference node..."
python3 -u f1tenth_rl/ros2/inference_node.py --ros-args \
  -p model_path:="$RUN_DIR/final_model.zip" \
  -p config_path:="$RUN_DIR/config.yaml" \
  -p use_onnx:=false \
  -p max_speed:="$MAX_SPEED" \
  -p scan_topic:="$SCAN_TOPIC" \
  -p odom_topic:="$ODOM_TOPIC" \
  -p drive_topic:="$DRIVE_TOPIC" &

NODE_PID=$!
echo "[team7] RL node pid: $NODE_PID"

if ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
  echo "[team7] launching foxglove_bridge..."
  ros2 run foxglove_bridge foxglove_bridge --ros-args \
    -p port:="$FOXGLOVE_PORT" \
    -p address:=0.0.0.0 &
  BRIDGE_PID=$!
  echo "Foxglove bridge running on ws://localhost:$FOXGLOVE_PORT"
else
  BRIDGE_PID=""
  echo "foxglove_bridge is not installed; RL node is running without Foxglove bridge."
  echo "Install with: sudo apt install ros-humble-foxglove-bridge"
fi

echo "[team7] running. In another terminal, check:"
echo "  ros2 topic echo ${SCAN_TOPIC} --once"
echo "  ros2 topic echo ${ODOM_TOPIC} --once"
echo "  ros2 topic echo ${DRIVE_TOPIC} --once"
echo "[team7] Press Ctrl-C to stop."

trap 'kill "$NODE_PID" ${BRIDGE_PID:+"$BRIDGE_PID"} 2>/dev/null || true' EXIT
wait "$NODE_PID"
