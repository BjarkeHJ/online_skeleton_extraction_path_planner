#!/usr/bin/env python3
import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from ament_index_python.packages import get_package_share_directory

class CloudSaver(Node):
    def __init__(self):
        super().__init__('cloud_saver')

        # Declare parameters
        self.declare_parameter('topic_name', '/osep/tsdf/static_pointcloud')
        self.declare_parameter('filename', 'static_cloud.pcd')
        topic = self.get_parameter('topic_name').get_parameter_value().string_value
        filename = self.get_parameter('filename').get_parameter_value().string_value

        if not filename.endswith('.pcd'):
            raise ValueError(f"filename must end with '.pcd', got: {filename}")

        # Hardcoded results directory
        results_dir = "/workspaces/isaac_ros-dev/src/osep/results"
        os.makedirs(results_dir, exist_ok=True)
        self.save_path = os.path.join(results_dir, filename)

        self.create_subscription(PointCloud2, topic, self.callback, 10)
        self.get_logger().info(f"Subscribed to topic: {topic}")
        self.get_logger().info(f"Will save to: {self.save_path}")

    def callback(self, msg):
        points = np.asarray([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ], dtype=np.float64)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(self.save_path, cloud)
        self.get_logger().info(f"Saved {self.save_path} with {len(points)} points")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CloudSaver())
    rclpy.shutdown()

if __name__ == "__main__":
    main()