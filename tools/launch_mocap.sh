#!/usr/bin/env bash
set -eo pipefail

. ./tools/setup_mocap.sh

export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

python scripts/nominal_frame_publisher.py &
publisher_pid=$!

ros2 run motion_capture_tracking motion_capture_tracking_node \
  --ros-args \
  --remap __node:=motion_capture_tracking \
  --params-file ros_ws/src/motion_capture_tracking/motion_capture_tracking/config/cfg.yaml &
tracking_pid=$!

cleanup() {
  kill "$publisher_pid" "$tracking_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

rviz2 -d tools/mocap.rviz
