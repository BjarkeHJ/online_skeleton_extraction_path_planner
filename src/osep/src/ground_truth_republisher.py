#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import csv
import os
import time

class PathReplayer(Node):
    def __init__(self):
        super().__init__('path_replayer')
        self.declare_parameter('csv_file', 'ground_truth_path.csv')
        self.declare_parameter('pose_topic', '/osep/replayed_path')
        self.declare_parameter('path_topic', '/osep/replayed_path_sofar')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('stride', 1)

        self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.playback_rate = self.get_parameter('playback_rate').get_parameter_value().double_value
        self.stride = self.get_parameter('stride').get_parameter_value().integer_value

        self.csv_path = os.path.join("/workspaces/isaac_ros-dev/src/osep/results", self.csv_file)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV path file not found: {self.csv_path}")

        # Load the path
        self.path = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.path.append({
                    'time': float(row['time']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'qx': float(row['qx']),
                    'qy': float(row['qy']),
                    'qz': float(row['qz']),
                    'qw': float(row['qw']),
                })

        if self.stride > 1:
            self.path = self.path[::self.stride]

        if len(self.path) < 2:
            raise RuntimeError("Path file must contain at least two points after striding.")

        self.get_logger().info(f"Loaded {len(self.path)} poses from {self.csv_path} (stride={self.stride})")

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.idx = 0
        self.start_wall_time = time.time()
        self.start_path_time = self.path[0]['time']
        self.path_so_far = []

        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        if self.idx >= len(self.path):
            self.get_logger().info("Finished replaying path.")
            return

        elapsed = (time.time() - self.start_wall_time) * self.playback_rate
        target_time = self.path[self.idx]['time'] - self.start_path_time

        if elapsed < target_time:
            return

        pose = self.path[self.idx]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = pose['x']
        msg.pose.position.y = pose['y']
        msg.pose.position.z = pose['z']
        msg.pose.orientation.x = pose['qx']
        msg.pose.orientation.y = pose['qy']
        msg.pose.orientation.z = pose['qz']
        msg.pose.orientation.w = pose['qw']
        self.pose_pub.publish(msg)

        # Add to path so far and publish Path
        self.path_so_far.append(msg)
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = self.frame_id
        path_msg.poses = self.path_so_far.copy()
        self.path_pub.publish(path_msg)

        self.idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = PathReplayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()