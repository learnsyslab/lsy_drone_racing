
#!/usr/bin/env bash
mkdir -p ros_ws/src

if [ ! -d ros_ws/src/motion_capture_tracking/.git ]; then
  echo "[Pixi activation] Cloning motion_capture_tracking..."
  git clone --recurse-submodules https://github.com/learnsyslab/motion_capture_tracking ros_ws/src/motion_capture_tracking
fi

if [ ! -x ros_ws/install/motion_capture_tracking/lib/motion_capture_tracking/motion_capture_tracking_node ] || [ ! -d ros_ws/install/drone_racing_msgs ]; then
  echo "[Pixi activation] Running colcon build..."
  (cd ros_ws && colcon build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
fi

. ./ros_ws/install/setup.sh