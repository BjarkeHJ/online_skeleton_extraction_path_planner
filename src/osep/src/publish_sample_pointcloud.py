#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os

class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')
        self.declare_parameter('topic_name', 'sample_pointcloud')
        self.declare_parameter('filename', 'static_cloud.pcd')
        topic = self.get_parameter('topic_name').get_parameter_value().string_value
        filename = self.get_parameter('filename').get_parameter_value().string_value

        # Hardcoded results directory (same as saver)
        results_dir = "/workspaces/isaac_ros-dev/src/osep/results"
        self.pcd_path = os.path.join(results_dir, filename)

        if not os.path.exists(self.pcd_path):
            raise FileNotFoundError(f"PCD file not found: {self.pcd_path}")

        self.publisher_ = self.create_publisher(PointCloud2, topic, 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"Publishing {self.pcd_path} on topic: {topic}")

        # Load the point cloud once
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        self.points = np.asarray(pcd.points, dtype=np.float32)
        self.colors = None
        if pcd.has_colors():
            self.colors = np.asarray(pcd.colors, dtype=np.float32)
            self.get_logger().info("Loaded colors from PCD.")
        self.frame_id = "odom"
        self.get_logger().info(f"Loaded {self.points.shape[0]} points from {self.pcd_path}")
        self.get_logger().info(f"Points dtype: {self.points.dtype}")
        self.get_logger().info(f"Any NaN: {np.isnan(self.points).any()}, Any Inf: {np.isinf(self.points).any()}")

    def timer_callback(self):
        if self.points.size == 0:
            self.get_logger().warn("No points to publish.")
            return

        if self.colors is not None and self.colors.shape == self.points.shape:
            # Convert RGB float [0,1] to packed float32
            rgb_uint8 = (self.colors * 255).astype(np.uint8)
            rgb_packed = np.left_shift(rgb_uint8[:, 0], 16) + \
                         np.left_shift(rgb_uint8[:, 1], 8) + \
                         rgb_uint8[:, 2]
            points_with_rgb = np.zeros(self.points.shape[0], dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
            ])
            points_with_rgb['x'] = self.points[:, 0]
            points_with_rgb['y'] = self.points[:, 1]
            points_with_rgb['z'] = self.points[:, 2]
            points_with_rgb['rgb'] = rgb_packed.view(np.float32)
            msg = pc2.create_cloud(
                header=self.make_header(),
                fields=[
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                ],
                points=points_with_rgb
            )
        else:
            msg = pc2.create_cloud_xyz32(
                header=self.make_header(),
                points=self.points
            )
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published pointcloud with {self.points.shape[0]} points.")

    def make_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header

def main(args=None):
    rclpy.init(args=args)
    node = PCDPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()