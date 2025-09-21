#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import collections
import numpy as np
from sklearn.mixture import GaussianMixture
import sensor_msgs_py.point_cloud2 as pc2

class Skeletonizer:
    def __init__(self, voxel_size=1.0, super_voxel_factor=4.0,
                 max_edge_points=10, dot_threshold=0.8, min_dist_factor=5.0, max_clusters=20,
                 merge_radius_factor=10.0):
        self.voxel_size = voxel_size
        self.super_voxel_size = super_voxel_factor * voxel_size
        self.max_edge_points = max_edge_points
        self.dot_threshold = dot_threshold
        self.min_dist_factor = min_dist_factor
        self.max_clusters = max_clusters
        self.merge_radius_factor = merge_radius_factor

    def _quantize(self, x, y, z):
        return (int(np.floor(x / self.super_voxel_size)),
                int(np.floor(y / self.super_voxel_size)),
                int(np.floor(z / self.super_voxel_size)))

    def cluster_detection(self, points):
        print("🔄 Fitting GMM models (iterative elbow detection)...")
        bics, models = [], []
        threshold = 0.02  # 2% relative improvement cutoff
        elbow_idx = None
        for k in range(1, self.max_clusters + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(points)
            bics.append(gmm.bic(points))
            models.append(gmm)
            if k > 1:
                improvement = -(bics[-1] - bics[-2]) / abs(bics[-2])
                if improvement < threshold:
                    elbow_idx = k - 1
                    break
        if elbow_idx is None:
            elbow_idx = int(np.argmin(bics)) + 1
        bics = np.array(bics)
        best_k = elbow_idx
        best_gmm = models[best_k - 1]
        labels = best_gmm.predict(points)
        print(f"Cluster sizes: {np.bincount(labels)}")
        return labels, best_k, best_gmm
    
    def extract_edges_and_centroids(self, points, labels, best_k):
        edge_color = np.array([1, 0, 0])  # Red
        centroid_color = np.array([0, 0.5, 1])  # Blue-ish for centroid
        raw_edge_points = []
        raw_edge_clusters = []  # List of lists of cluster indices
        raw_centroids = []
        for k in range(best_k):
            cluster_mask = (labels == k)
            cluster_points = points[cluster_mask]
            if len(cluster_points) == 0:
                continue
            centroid, edge_indices = self._process_cluster(cluster_points)
            for ei in edge_indices:
                if 0 <= ei < len(cluster_points):
                    raw_edge_points.append(cluster_points[ei])
                    raw_edge_clusters.append([k])
            raw_centroids.append(centroid)
        return np.array(raw_edge_points), raw_edge_clusters, np.array(raw_centroids), edge_color, centroid_color
    
    def _process_cluster(self, points):
        # 1. Supervoxel population map
        supervoxel_counts = collections.Counter(self._quantize(x, y, z) for x, y, z in points)
        # 2. Weighted centroid
        centroid = np.zeros(3)
        total_weight = 0.0
        for pt in points:
            v = self._quantize(*pt)
            weight = 1.0 / supervoxel_counts[v]
            centroid += weight * pt
            total_weight += weight
        if total_weight > 0:
            centroid /= total_weight
        # 3. Distances from centroid
        dists = np.linalg.norm(points - centroid, axis=1)
        idx_dist = sorted(enumerate(dists), key=lambda x: -x[1])
        # 4. Select up to max_edge_points with unique directions
        selected_indices = []
        selected_dirs = []
        selected_dist = []
        min_dist = self.min_dist_factor * self.voxel_size
        for idx, dist in idx_dist:
            if dist < min_dist:
                continue

            dir_vec = points[idx] - centroid
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            dir_vec /= norm

            if selected_dirs:
                # Compute dot products with all selected directions
                dots = np.dot(selected_dirs, dir_vec)
                max_idx = np.argmax(dots)

                if dots[max_idx] > self.dot_threshold:
                    # ✅ Compare distance with the matching selected distance (not just last)
                    if np.isclose(dist, selected_dist[max_idx], atol= 2 * self.voxel_size):
                        # Require the candidate to be close to ALL selected points
                        if all(
                            np.linalg.norm(points[idx] - points[si]) > 4 * self.voxel_size
                            for si in selected_indices
                        ):
                            selected_indices.append(idx)
                            selected_dirs.append(dir_vec)
                            selected_dist.append(dist)
                            continue
                        else:
                            # Too close to an existing point → skip
                            continue
                    else:
                        # Direction too similar but distance not close → reject
                        continue

            # If we get here, either no similar direction or candidate passed checks
            selected_indices.append(idx)
            selected_dirs.append(dir_vec)
            selected_dist.append(dist)

            if len(selected_indices) >= self.max_edge_points:
                break

        return centroid, selected_indices

class RealTimeSkeletonizerNode(Node):
    def __init__(self):
        super().__init__('realtime_skeletonizer')
        self.declare_parameter('static_input_topic', 'osep/tsdf/static_pointcloud')
        self.declare_parameter('output_topic', 'osep/tsdf/skeleton')

        self.input_topic = self.get_parameter('static_input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.skel = Skeletonizer()

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, 1)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 1)

        self.last_point_count = 0
        self.last_msg = None

    def callback(self, msg):
        points = np.asarray(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points.dtype.fields is not None:
            points = np.stack([points['x'], points['y'], points['z']], axis=-1)
        if len(points) == 0:
            self.get_logger().warn("Received empty point cloud.")
            return

        # Early return if point count matches last published
        if points.shape[0] == self.last_point_count and self.last_msg is not None:
            self.pub.publish(self.last_msg)
            return

        labels, best_k, _ = self.skel.cluster_detection(points)
        raw_edge_points, raw_edge_clusters, raw_centroids, edge_color, centroid_color = self.skel.extract_edges_and_centroids(points, labels, best_k)

        # Combine edge points and centroids
        combined_points = np.vstack([raw_edge_points, raw_centroids])
        edge_rgb = np.array([255, 0, 0], dtype=np.uint8)
        centroid_rgb = np.array([0, 128, 255], dtype=np.uint8)
        combined_colors = np.vstack([
            np.tile(edge_rgb, (len(raw_edge_points), 1)),
            np.tile(centroid_rgb, (len(raw_centroids), 1))
        ])

        def pack_rgb(r, g, b):
            rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
            return np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]

        if combined_points.shape[0] > 0:
            structured_combined = np.zeros(combined_points.shape[0], dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('rgb', np.float32)
            ])
            structured_combined['x'] = combined_points[:, 0]
            structured_combined['y'] = combined_points[:, 1]
            structured_combined['z'] = combined_points[:, 2]
            structured_combined['rgb'] = [pack_rgb(*c) for c in combined_colors]

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            header = msg.header
            pc2_msg = pc2.create_cloud(header, fields, structured_combined)
            self.pub.publish(pc2_msg)
            self.last_point_count = points.shape[0]
            self.last_msg = pc2_msg
            self.get_logger().info(f"Published {len(raw_edge_points)} edge points and {len(raw_centroids)} centroids.")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeSkeletonizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()