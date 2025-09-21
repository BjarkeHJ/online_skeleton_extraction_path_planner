#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.mixture import GaussianMixture
import sensor_msgs_py.point_cloud2 as pc2


class Skeletonizer:
    def __init__(self, voxel_size=1.0, super_voxel_factor=4.0,
                 max_edge_points=10, dot_threshold=0.8, min_dist_factor=10.0, max_clusters=20,
                 merge_radius_factor=5.0):
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

    def merge_points_within_clusters(self, merged_edge_points, merged_clusters, points):
        """
        For each cluster, merge points that are within 2x merge_radius of each other.
        Returns new merged points and their cluster lists.
        """
        merged_edge_points = np.asarray(merged_edge_points)
        final_points = []
        final_clusters = []
        for k in set(i for clist in merged_clusters for i in clist):
            idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            if len(idxs) == 0:
                continue
            pts = merged_edge_points[idxs]
            db = DBSCAN(eps=2 * self.merge_radius_factor * self.voxel_size, min_samples=1).fit(pts)
            for label in np.unique(db.labels_):
                group = pts[db.labels_ == label]
                group_idxs = np.array(idxs)[db.labels_ == label]
                group_clusters = [merged_clusters[i] for i in group_idxs]
                merged_cluster = sorted(set(i for sublist in group_clusters for i in sublist))
                group_centroid = np.mean(group, axis=0)
                dists = np.linalg.norm(points - group_centroid, axis=1)
                best_idx = np.argmin(dists)
                final_points.append(points[best_idx])
                final_clusters.append(merged_cluster)
        return np.array(final_points), final_clusters
    
    def densify_skeleton(self, merged_edge_points, merged_clusters, points, labels, 
                        max_dist=5, min_points_for_skeleton=10):
        
        densified = {}
        
        for k in set(i for clist in merged_clusters for i in clist):
            # Get all points in this cluster
            cluster_mask = (labels == k)
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < min_points_for_skeleton:
                continue
                
            # Get edge points for this cluster
            idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            if len(idxs) < 2:
                continue
                
            edge_pts = np.array([merged_edge_points[i] for i in idxs])
            
            # Start with edge points as key skeleton points
            skel_points = set(tuple(pt) for pt in edge_pts)
            
            # Build minimum spanning tree of edge points
            n = len(edge_pts)
            dist_matrix = np.full((n, n), np.inf)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(edge_pts[i] - edge_pts[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = d
            
            mst = minimum_spanning_tree(dist_matrix)
            mst_edges = np.array(mst.nonzero()).T
            
            # Interpolate between all connected edge points
            for i, j in mst_edges:
                p1, p2 = edge_pts[int(i)], edge_pts[int(j)]
                d = np.linalg.norm(p2 - p1)
                n_steps = max(1, int(np.ceil(d / (max_dist * self.voxel_size))))
                for t in range(1, n_steps):
                    interp = p1 + (p2 - p1) * (t / n_steps)
                    skel_points.add(tuple(interp))
            
            if skel_points:
                densified[k] = np.array(list(skel_points))
        
        return densified
        

class RealTimeSkeletonizerNode(Node):
    def __init__(self):
        super().__init__('realtime_skeletonizer')
        self.declare_parameter('static_input_topic', 'osep/tsdf/static_pointcloud')
        self.declare_parameter('output_topic', 'osep/tsdf/skeleton')

        self.input_topic = self.get_parameter('static_input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.skel = Skeletonizer()

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, 1)
        self.skeleton_pub = self.create_publisher(PointCloud2, self.output_topic, 1)
        self.centroids_pub = self.create_publisher(PointCloud2, self.output_topic + "/centroids", 1)

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
            self.centroids_pub.publish(self.last_msg)
            return

        labels, best_k, _ = self.skel.cluster_detection(points)
        raw_edge_points, raw_edge_clusters, raw_centroids, edge_color, centroid_color = self.skel.extract_edges_and_centroids(points, labels, best_k)
        merged_edge_points, merged_clusters = self.skel.merge_points_within_clusters(raw_edge_points, raw_edge_clusters, points)
        densified = self.skel.densify_skeleton(merged_edge_points, merged_clusters, points, labels, max_dist=5)

        # Combine edge points and centroids
        combined_points = np.vstack([merged_edge_points, raw_centroids])
        edge_rgb = np.array([255, 0, 0], dtype=np.uint8)
        centroid_rgb = np.array([0, 128, 255], dtype=np.uint8)
        combined_colors = np.vstack([
            np.tile(edge_rgb, (len(merged_edge_points), 1)),
            np.tile(centroid_rgb, (len(raw_centroids), 1))
        ])

        def pack_rgb(r, g, b):
            rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
            return np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]

        # Publish edge points and centroids (combined)
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
            self.centroids_pub.publish(pc2_msg)
            self.last_point_count = points.shape[0]
            self.last_msg = pc2_msg
            self.get_logger().info(f"Published {len(raw_edge_points)} edge points and {len(raw_centroids)} centroids.")

        # Publish densified skeleton with unique color per cluster
        if densified:
            all_skel_points = []
            all_skel_colors = []
            rng = np.random.default_rng(42)
            color_map = rng.integers(0, 255, size=(max(densified.keys())+1, 3), dtype=np.uint8)
            for k, pts in densified.items():
                all_skel_points.append(pts)
                all_skel_colors.append(np.tile(color_map[k], (pts.shape[0], 1)))
            all_skel_points = np.vstack(all_skel_points)
            all_skel_colors = np.vstack(all_skel_colors)
            structured_skel = np.zeros(all_skel_points.shape[0], dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('rgb', np.float32)
            ])
            structured_skel['x'] = all_skel_points[:, 0]
            structured_skel['y'] = all_skel_points[:, 1]
            structured_skel['z'] = all_skel_points[:, 2]
            structured_skel['rgb'] = [pack_rgb(*c) for c in all_skel_colors]
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            header = msg.header
            skel_msg = pc2.create_cloud(header, fields, structured_skel)
            self.skeleton_pub.publish(skel_msg)
            self.get_logger().info(f"Published densified skeleton with {all_skel_points.shape[0]} points.")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeSkeletonizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()