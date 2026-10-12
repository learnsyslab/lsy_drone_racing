"""ROS 2 node for publishing nominal gate, obstacle, and drone-init poses in RViz.

When launching this node, add a `MarkerArray` display in RViz and subscribe it to the
`/nominal_frames` topic. Each object is shown with a small coordinate system at its center and a
text label (e.g. `nominal_gate1`).
"""

import os
from pathlib import Path

os.environ["SCIPY_ARRAY_API"] = "1"

import argparse

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

from lsy_drone_racing.utils import load_config

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


class NominalFramePublisherNode(Node):
    """Publish nominal gate and obstacle markers from a TOML track config."""

    def __init__(self, config_name: str = "level2.toml"):
        """Create the publisher node.

        Args:
            config_name: Name of the track config file inside the repository-level config dir.
        """
        super().__init__("nominal_frame_publisher")
        self.publisher = self.create_publisher(MarkerArray, "nominal_frames", 10)
        self.config = load_config(Path(__file__).resolve().parents[1] / "config" / config_name)

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Publishing nominal frames from {config_name} on /nominal_frames")

    def get_poses(
        self,
    ) -> tuple[
        list[tuple[list[float], list[float]]],
        list[list[float]],
        list[list[float]],
        list[tuple[list[float], list[float]]],
    ]:
        """Return the nominal gate, obstacle, and drone poses from the configuration."""
        gate_poses = [
            (gate["pos"], (R.from_euler("xyz", gate["rpy"])).as_quat().tolist())
            for gate in self.config.env.track.gates
        ]

        gate_marker_orientations = [
            (R.from_euler("xyz", gate["rpy"]) * R.from_euler("xyz", [0, 1.5708, 0]))
            .as_quat()
            .tolist()
            for gate in self.config.env.track.gates
        ]

        obstacle_poses = [obstacle["pos"] for obstacle in self.config.env.track.obstacles]
        drone_poses = [
            (drone["pos"], (R.from_euler("xyz", drone["rpy"])).as_quat().tolist())
            for drone in self.config.env.track.drones
        ]
        return gate_poses, gate_marker_orientations, obstacle_poses, drone_poses

    def _make_marker(
        self,
        marker_id: int,
        namespace: str,
        marker_type: int,
        frame_id: str,
        position: list[float],
        orientation: list[float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(orientation[0])
        marker.pose.orientation.y = float(orientation[1])
        marker.pose.orientation.z = float(orientation[2])
        marker.pose.orientation.w = float(orientation[3])
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def _make_axes(
        self,
        marker_id: int,
        namespace: str,
        position: list[float],
        orientation: list[float],
        length: float = 0.15,
    ) -> list[Marker]:
        """Create a small coordinate system (x=red, y=green, z=blue) at the given pose."""
        rot = R.from_quat(orientation)
        # Arrow markers point along their local x-axis. Rotate them onto the y- and z-axes.
        axis_rotations = [R.identity(), R.from_euler("z", 1.5708), R.from_euler("y", -1.5708)]
        colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
        return [
            self._make_marker(
                marker_id=3 * marker_id + axis,
                namespace=namespace,
                marker_type=Marker.ARROW,
                frame_id="world",
                position=position,
                orientation=(rot * axis_rot).as_quat().tolist(),
                scale=(length, 0.015, 0.015),
                color=color,
            )
            for axis, (axis_rot, color) in enumerate(zip(axis_rotations, colors))
        ]

    def _make_text(
        self, marker_id: int, namespace: str, text: str, position: list[float]
    ) -> Marker:
        """Create a camera-facing text label at the given position."""
        marker = self._make_marker(
            marker_id=marker_id,
            namespace=namespace,
            marker_type=Marker.TEXT_VIEW_FACING,
            frame_id="world",
            position=position,
            orientation=IDENTITY_QUAT,
            scale=(0.0, 0.0, 0.08),
            color=(1.0, 1.0, 1.0, 1.0),
        )
        marker.text = text
        return marker

    def timer_callback(self):
        """Publish the nominal track objects at a fixed rate."""
        msg = MarkerArray()
        gate_poses, marker_orientations, obstacle_poses, drone_poses = self.get_poses()

        for index, ((position, orientation), marker_orientation) in enumerate(
            zip(gate_poses, marker_orientations)
        ):
            msg.markers.append(
                self._make_marker(
                    marker_id=index,
                    namespace="gates",
                    marker_type=Marker.CUBE,
                    frame_id="world",
                    position=position,
                    orientation=marker_orientation,
                    scale=(0.72, 0.72, 0.02),
                    color=(0.2, 0.8, 1.0, 0.5),
                )
            )
            msg.markers.extend(self._make_axes(index, "gate_axes", position, orientation))
            label_position = [position[0], position[1], position[2] + 0.45]
            msg.markers.append(
                self._make_text(index, "gate_labels", f"nominal_gate{index + 1}", label_position)
            )

        for index, position in enumerate(obstacle_poses):
            center = [position[0], position[1], 1.52 / 2]
            msg.markers.append(
                self._make_marker(
                    marker_id=index,
                    namespace="obstacles",
                    marker_type=Marker.CYLINDER,
                    frame_id="world",
                    position=center,
                    orientation=IDENTITY_QUAT,
                    scale=(0.03, 0.03, 1.5),
                    color=(1.0, 0.6, 0.2, 0.8),
                )
            )
            msg.markers.extend(self._make_axes(index, "obstacle_axes", position, IDENTITY_QUAT))
            label_position = [position[0], position[1], position[2] + 0.1]
            msg.markers.append(
                self._make_text(
                    index, "obstacle_labels", f"nominal_obstacle{index + 1}", label_position
                )
            )

        for index, (position, orientation) in enumerate(drone_poses):
            msg.markers.extend(self._make_axes(index, "drone_init_axes", position, orientation))
            label_position = [position[0], position[1], position[2] + 0.2]
            msg.markers.append(
                self._make_text(
                    index, "drone_init_labels", f"nominal_drone{index + 1}", label_position
                )
            )

        self.publisher.publish(msg)


def main():
    """Run the nominal frame publisher node."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="level2.toml")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = NominalFramePublisherNode(config_name=args.config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
