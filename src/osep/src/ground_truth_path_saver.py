#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import csv
import os
import sys

class PathSaver(Node):
    def __init__(self):
        super().__init__('path_saver')
        self.declare_parameter('topic', '/osep/ground_truth')
        self.declare_parameter('result_file', 'ground_truth_path.csv')
        topic = self.get_parameter('topic').get_parameter_value().string_value
        result_file = self.get_parameter('result_file').get_parameter_value().string_value
        self.result_path = os.path.join(
            "/workspaces/isaac_ros-dev/src/osep/results", result_file)
        self.sub = self.create_subscription(Path, topic, self.path_callback, 10)
        self.saved = False
        self.get_logger().info(f"Waiting for path on {topic}...")

    def path_callback(self, msg):
        if self.saved:
            return
        poses = []
        for pose_stamped in msg.poses:
            t = pose_stamped.header.stamp.sec + pose_stamped.header.stamp.nanosec * 1e-9
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z
            qx = pose_stamped.pose.orientation.x
            qy = pose_stamped.pose.orientation.y
            qz = pose_stamped.pose.orientation.z
            qw = pose_stamped.pose.orientation.w
            poses.append((t, x, y, z, qx, qy, qz, qw))
        with open(self.result_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            writer.writerows(poses)
        self.get_logger().info(f"Saved {len(poses)} poses to {self.result_path}")
        self.saved = True
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = PathSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()