#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import math

def get_yaw_from_quaternion(q):
    # q = [x, y, z, w]
    siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return math.atan2(siny_cosp, cosy_cosp)

def is_in_pyramid(pt, drone_pos, drone_yaw, pyramid_length, pyramid_width, pyramid_height):
    rel = pt - drone_pos
    c, s = np.cos(-drone_yaw), np.sin(-drone_yaw)
    x = c * rel[0] - s * rel[1]
    y = s * rel[0] + c * rel[1]
    z = rel[2]
    if x < 0 or x > pyramid_length:
        return False
    half_w = (pyramid_width / 2) * (x / pyramid_length)
    half_h = (pyramid_height / 2) * (x / pyramid_length)
    if abs(y) > half_w or abs(z) > half_h:
        return False
    return True

def raycast_occluded(pt, drone_pos, points_set, voxel_size):
    direction = pt - drone_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return False
    direction = direction / dist
    steps = int(dist / voxel_size)
    target_voxel = tuple(np.floor(pt / voxel_size).astype(int))
    for i in range(1, steps):
        pos = drone_pos + direction * (i * voxel_size)
        voxel = tuple(np.floor(pos / voxel_size).astype(int))
        if voxel in points_set and voxel != target_voxel:
            return True
    return False

class PCDProcessorPublisher(Node):
    def __init__(self):
        super().__init__('pcd_processor_publisher')
        self.declare_parameter('topic_name', 'sample_pointcloud')
        self.declare_parameter('filename', 'static_cloud.pcd')
        self.declare_parameter('voxel_size', 0.25)
        self.declare_parameter('pyramid_length', 20.0)
        self.declare_parameter('pyramid_width', 15.0)
        self.declare_parameter('pyramid_height', 15.0)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('camera_frame', 'base_link')
        self.declare_parameter('detection_distance', 20.0)

        topic = self.get_parameter('topic_name').get_parameter_value().string_value
        filename = self.get_parameter('filename').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.pyramid_length = self.get_parameter('pyramid_length').get_parameter_value().double_value
        self.pyramid_width = self.get_parameter('pyramid_width').get_parameter_value().double_value
        self.pyramid_height = self.get_parameter('pyramid_height').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.detection_distance = self.get_parameter('detection_distance').get_parameter_value().double_value

        results_dir = "/workspaces/isaac_ros-dev/src/osep/results"
        self.pcd_path = os.path.join(results_dir, filename)

        if not os.path.exists(self.pcd_path):
            raise FileNotFoundError(f"PCD file not found: {self.pcd_path}")

        self.publisher_ = self.create_publisher(PointCloud2, topic, 1)
        self.marker_pub = self.create_publisher(Marker, "pyramid_marker", 1)
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.get_logger().info(f"Publishing {self.pcd_path} on topic: {topic}")

        # Load the point cloud once
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        self.points = np.asarray(pcd.points, dtype=np.float32)
        self.colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (self.points.shape[0], 1))  # All red

        self.get_logger().info(f"Loaded {self.points.shape[0]} points from {self.pcd_path}")

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def timer_callback(self):
        try:
            now = rclpy.time.Time()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.frame_id, self.camera_frame, now)
            drone_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            q = trans.transform.rotation
            drone_quat = [q.x, q.y, q.z, q.w]
            drone_yaw = get_yaw_from_quaternion(drone_quat)
        except Exception as e:
            self.get_logger().warn(f"TF not available: {e}")
            return

        # Publish the pyramid marker for visualization
        self.publish_pyramid_marker(drone_pos, drone_yaw)

        # Build a set of occupied voxels for fast lookup
        points_voxels = set(tuple(np.floor(pt / self.voxel_size).astype(int)) for pt in self.points)

        vecs = self.points - drone_pos
        dists = np.linalg.norm(vecs, axis=1)
        within_dist = dists <= self.detection_distance
        indices = np.where(within_dist)[0]
        # Print number of points within distance
        self.get_logger().info(f"Points within distance: {len(indices)}")


        for i in indices:
            pt = self.points[i]
            if not is_in_pyramid(pt, drone_pos, drone_yaw, self.pyramid_length, self.pyramid_width, self.pyramid_height):
                continue
            if raycast_occluded(pt, drone_pos, points_voxels, self.voxel_size):
                continue
            # Only update if not already green
            if not np.allclose(self.colors[i], [0.0, 1.0, 0.0]):
                self.colors[i] = [0.0, 1.0, 0.0]  # Set to green

        # Pack colors and publish
        rgb_uint8 = (self.colors * 255).astype(np.uint8)
        rgb_packed = (rgb_uint8[:, 0].astype(np.uint32) << 16) | \
                    (rgb_uint8[:, 1].astype(np.uint32) << 8)  | \
                    (rgb_uint8[:, 2].astype(np.uint32))

        # reinterpret as float32 for ROS PointCloud2
        rgb_float = rgb_packed.view(np.float32)

        points_with_rgb = np.zeros(self.points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
        ])
        points_with_rgb['x'] = self.points[:, 0]
        points_with_rgb['y'] = self.points[:, 1]
        points_with_rgb['z'] = self.points[:, 2]
        points_with_rgb['rgb'] = rgb_float

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
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published processed pointcloud with {self.points.shape[0]} points.")


    def publish_pyramid_marker(self, drone_pos, drone_yaw):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pyramid"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width

        # Color: blue
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Pyramid corners in local camera frame
        L = self.pyramid_length
        W = self.pyramid_width
        H = self.pyramid_height
        corners = np.array([
            [0, 0, 0],  # tip (drone)
            [L, -W/2, -H/2],  # base corners
            [L, -W/2, H/2],
            [L, W/2, H/2],
            [L, W/2, -H/2],
        ])
        # Rotate and translate to world
        c, s = np.cos(drone_yaw), np.sin(drone_yaw)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        corners_world = (R @ corners.T).T + drone_pos

        # Define lines (tip to base, and base edges)
        lines = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # tip to base
            (1, 2), (2, 3), (3, 4), (4, 1)   # base edges
        ]
        for i1, i2 in lines:
            marker.points.append(self.make_point(corners_world[i1]))
            marker.points.append(self.make_point(corners_world[i2]))

        self.marker_pub.publish(marker)

    def make_point(self, arr):
        p = Point()
        p.x, p.y, p.z = arr.tolist()
        return p

    def make_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header

def main(args=None):
    rclpy.init(args=args)
    node = PCDProcessorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()