"""ROS 2 node for publishing nominal gate, obstacle, and drone-init poses in RViz.

When launching this node, add a `MarkerArray` display in RViz and subscribe it to the
`/nominal_frame_publisher` topic.
"""

import os
from pathlib import Path

os.environ["SCIPY_ARRAY_API"] = "1"

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from lsy_drone_racing.utils import load_config


class NominalFramePublisherNode(Node):
    """Publish nominal gate and obstacle markers from a TOML track config."""

    def __init__(self, config_name: str = "level2.toml"):
        """Create the publisher node.

        Args:
            config_name: Name of the track config file inside the repository-level config dir.
        """
        super().__init__("nominal_frame_publisher")
        self.publisher = self.create_publisher(MarkerArray, "nominal_frame_publisher", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.config = load_config(Path(__file__).resolve().parents[1] / "config" / config_name)

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _make_transform(
        self, frame_id: str, child_frame_id: str, position: list[float], orientation: list[float]
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = frame_id
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.x = float(orientation[0])
        transform.transform.rotation.y = float(orientation[1])
        transform.transform.rotation.z = float(orientation[2])
        transform.transform.rotation.w = float(orientation[3])
        return transform

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

    def timer_callback(self):
        """Publish the nominal track objects at a fixed rate."""
        msg = MarkerArray()
        transforms: list[TransformStamped] = []
        gate_poses, marker_orientations, obstacle_poses, drone_poses = self.get_poses()

        for marker_id, ((position, orientation), marker_orientation) in enumerate(
            zip(gate_poses, marker_orientations)
        ):
            gate = self.config.env.track.gates[marker_id]
            frame_id = gate.get("name", f"g_{marker_id + 1}")
            transforms.append(
                self._make_transform(
                    frame_id="world",
                    child_frame_id=frame_id,
                    position=position,
                    orientation=orientation,
                )
            )
            msg.markers.append(
                self._make_marker(
                    marker_id=marker_id,
                    namespace="gates",
                    marker_type=Marker.CUBE,
                    frame_id="world",
                    position=position,
                    orientation=marker_orientation,
                    scale=(0.72, 0.72, 0.02),
                    color=(0.2, 0.8, 1.0, 0.5),
                )
            )

        obstacle_offset = len(gate_poses)
        for index, position in enumerate(obstacle_poses):
            obstacle = self.config.env.track.obstacles[index]
            frame_id = obstacle.get("name", f"o_{index + 1}")
            marker_obstacle_position = [position[0], position[1], 1.52 / 2]
            transforms.append(
                self._make_transform(
                    frame_id="world",
                    child_frame_id=frame_id,
                    position=position,
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            msg.markers.append(
                self._make_marker(
                    marker_id=obstacle_offset + index,
                    namespace="obstacles",
                    marker_type=Marker.CYLINDER,
                    frame_id="world",
                    position=marker_obstacle_position,
                    orientation=[0.0, 0.0, 0.0, 1.0],
                    scale=(0.03, 0.03, 1.5),
                    color=(1.0, 0.6, 0.2, 0.8),
                )
            )

        drone_offset = obstacle_offset + len(obstacle_poses)
        for index, (position, orientation) in enumerate(drone_poses):
            drone = self.config.env.track.drones[index]
            frame_id = drone.get("name", f"drone_init_{index + 1}")
            marker_position = [position[0], position[1], position[2] + 0.05]
            transforms.append(
                self._make_transform(
                    frame_id="world",
                    child_frame_id=frame_id,
                    position=position,
                    orientation=orientation,
                )
            )
            msg.markers.append(
                self._make_marker(
                    marker_id=drone_offset + index,
                    namespace="drone_init",
                    marker_type=Marker.ARROW,
                    frame_id="world",
                    position=marker_position,
                    orientation=orientation,
                    scale=(0.25, 0.05, 0.05),
                    color=(0.3, 1.0, 0.3, 0.9),
                )
            )

        self.publisher.publish(msg)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)


def main():
    """Run the nominal frame publisher node."""
    rclpy.init()
    node = NominalFramePublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
