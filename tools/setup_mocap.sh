
#!/usr/bin/env bash
mkdir -p ros_ws/src

if [ ! -d ros_ws/src/motion_capture_tracking/.git ]; then
  echo "[Pixi activation] Cloning motion_capture_tracking..."
  git clone --recurse-submodules https://github.com/learnsyslab/motion_capture_tracking ros_ws/src/motion_capture_tracking
fi

if [ ! -f ros_ws/install/setup.sh ] || [ ! -d ros_ws/install/drone_racing_msgs ]; then
  echo "[Pixi activation] Running colcon build..."
  # One build at a time when several terminals activate together
  (cd ros_ws && flock . colcon build --packages-skip-build-finished --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
fi

. ./ros_ws/install/setup.sh