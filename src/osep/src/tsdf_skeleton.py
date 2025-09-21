#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt


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
     
    def filter_lonely_points(self, points, min_cluster_size=10, eps_factor=3.0):
        """
        Remove points that are not part of a large connected component.
        Uses DBSCAN to find connected structures.
        """
        if len(points) == 0:
            return points
        # eps is the neighborhood size, set relative to voxel size
        eps = eps_factor * self.voxel_size
        db = DBSCAN(eps=eps, min_samples=1).fit(points)
        labels, counts = np.unique(db.labels_, return_counts=True)
        # Only keep clusters with enough points
        large_clusters = labels[counts >= min_cluster_size]
        mask = np.isin(db.labels_, large_clusters)
        filtered_points = points[mask]
        removed = len(points) - len(filtered_points)
        if removed > 0:
            print(f"Filtered out {removed} lonely points (remaining: {len(filtered_points)})")
        return filtered_points

    def _quantize(self, x, y, z):
        return (int(np.floor(x / self.super_voxel_size)),
                int(np.floor(y / self.super_voxel_size)),
                int(np.floor(z / self.super_voxel_size)))

    def _full_dilation(self, points, dilation_voxels=1):
        """
        Perform full 3D dilation: for each point, add all neighbors within dilation_voxels in each axis.
        Returns unique set of dilated points.
        """
        offsets = np.array([
            [dx, dy, dz]
            for dx in range(-dilation_voxels, dilation_voxels + 1)
            for dy in range(-dilation_voxels, dilation_voxels + 1)
            for dz in range(-dilation_voxels, dilation_voxels + 1)
        ]) * self.voxel_size

        # Broadcast-add offsets to all points
        dilated = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        # Remove duplicates
        dilated_unique = np.unique(dilated, axis=0)
        return dilated_unique

    def cluster_detection(self, points, dilation_voxels=1, min_cluster_size=50):
        print("🔄 Fitting GMM models (iterative elbow detection)...")
        # 1. Dilation for GMM fitting only
        if dilation_voxels > 0:
            dilated_points = self._full_dilation(points, dilation_voxels)
            print(f"Applied full dilation with {dilation_voxels} voxels. Dilation points: {len(dilated_points)}")
        else:
            dilated_points = points

        bics, models = [], []
        threshold = 0.02  # 2% relative improvement cutoff
        elbow_idx = None
        for k in range(1, self.max_clusters + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(dilated_points)
            bics.append(gmm.bic(dilated_points))
            models.append(gmm)
            if k > 1:
                improvement = -(bics[-1] - bics[-2]) / abs(bics[-2])
                if improvement < threshold:
                    elbow_idx = k - 1
                    print(f"Elbow detected at k={elbow_idx} (breaking point, improvement={improvement:.4f})")
                    break
        if elbow_idx is None:
            elbow_idx = int(np.argmin(bics)) + 1
        bics = np.array(bics)
        best_k = elbow_idx
        best_gmm = models[best_k - 1]

        # 2. Assign each original point to the nearest GMM component
        labels = best_gmm.predict(points)

        # 3. Optionally: split each cluster into connected components (on original points)
        final_labels = np.full_like(labels, -1)
        next_label = 0
        for k in range(best_k):
            mask = (labels == k)
            if np.sum(mask) == 0:
                continue
            db = DBSCAN(eps=3.0 * self.voxel_size, min_samples=1).fit(points[mask])
            sub_labels = db.labels_
            for sub in np.unique(sub_labels):
                sub_mask = (sub_labels == sub)
                cluster_indices = np.where(mask)[0][sub_mask]
                if len(cluster_indices) < min_cluster_size:
                    # Too small, skip (leave as -1)
                    continue
                final_labels[cluster_indices] = next_label
                next_label += 1
        labels = final_labels
        best_k = next_label

        print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")
        return labels, best_k

    def extract_edges_and_centroids(self, points, labels, best_k):
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
        return np.array(raw_edge_points), raw_edge_clusters, np.array(raw_centroids)
    
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

    def _identify_edge_points_in_skeleton(self, skeleton_points, merged_edge_points, tol=1e-3):
        """
        Identify which skeleton points are original edge points
        """
        if len(skeleton_points) == 0 or len(merged_edge_points) == 0:
            return np.zeros(len(skeleton_points), dtype=bool)
        
        # Find nearest edge point for each skeleton point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(merged_edge_points)
        distances, _ = nn.kneighbors(skeleton_points)
        
        # Points that are very close to an edge point are considered edge points
        return distances.flatten() < tol
        
    def merge_skeleton_points(self, densified, merged_edge_points, merged_clusters):
        """
        Merge edge points in the skeleton, including those from different clusters if nearby
        Returns: merged_densified, updated_merged_edge_points, updated_merged_clusters
        """
        from sklearn.cluster import DBSCAN
        
        merged_densified = {}
        
        # First, collect ALL edge points from all clusters with their cluster associations
        all_edge_points = []
        all_edge_clusters = []
        point_to_original_clusters = {}  # Track which clusters each point belongs to
        
        for k, skel_pts in densified.items():
            if len(skel_pts) < 2:
                merged_densified[k] = skel_pts
                continue
            
            # Identify which skeleton points are original edge points
            cluster_edge_idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            cluster_edge_pts = merged_edge_points[cluster_edge_idxs] if cluster_edge_idxs else np.array([])
            
            edge_mask = self._identify_edge_points_in_skeleton(skel_pts, cluster_edge_pts)
            edge_points = skel_pts[edge_mask]
            
            # Store edge points with their cluster information
            for pt in edge_points:
                pt_tuple = tuple(pt)
                all_edge_points.append(pt)
                all_edge_clusters.append([k])  # Start with single cluster
                
                # Track which clusters this point originally belongs to
                if pt_tuple not in point_to_original_clusters:
                    point_to_original_clusters[pt_tuple] = set()
                point_to_original_clusters[pt_tuple].add(k)
        
        if not all_edge_points:
            return densified, merged_edge_points, merged_clusters
        
        # Merge ALL edge points regardless of cluster (including inter-cluster merging)
        all_edge_points = np.array(all_edge_points)
        db = DBSCAN(eps=self.merge_radius_factor * self.voxel_size, min_samples=1).fit(all_edge_points)
        
        updated_merged_edge_points = []
        updated_merged_clusters = []
        
        # Process each merged group
        for label in np.unique(db.labels_):
            group_indices = np.where(db.labels_ == label)[0]
            group_points = all_edge_points[group_indices]
            
            # Merge the points
            merged_point = np.mean(group_points, axis=0)
            updated_merged_edge_points.append(merged_point)
            
            # Combine cluster associations from all points in this group
            merged_cluster_set = set()
            for idx in group_indices:
                pt_tuple = tuple(all_edge_points[idx])
                if pt_tuple in point_to_original_clusters:
                    merged_cluster_set.update(point_to_original_clusters[pt_tuple])
            
            updated_merged_clusters.append(sorted(merged_cluster_set))
        
        # Now update each cluster's skeleton with the merged edge points
        for k in densified.keys():
            skel_pts = densified[k]
            
            # Identify which points are edge points in this cluster
            cluster_edge_idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            cluster_edge_pts = merged_edge_points[cluster_edge_idxs] if cluster_edge_idxs else np.array([])
            
            edge_mask = self._identify_edge_points_in_skeleton(skel_pts, cluster_edge_pts)
            edge_points = skel_pts[edge_mask]
            interpolated_points = skel_pts[~edge_mask]
            
            # Find the corresponding merged edge points for this cluster
            cluster_merged_edge_points = []
            for i, clusters in enumerate(updated_merged_clusters):
                if k in clusters:
                    cluster_merged_edge_points.append(updated_merged_edge_points[i])
            
            if cluster_merged_edge_points:
                # Replace edge points with merged versions
                final_points = np.vstack([cluster_merged_edge_points, interpolated_points])
                merged_densified[k] = final_points
            else:
                merged_densified[k] = skel_pts
        
        # Convert to numpy arrays
        updated_merged_edge_points = np.array(updated_merged_edge_points)
        
        return merged_densified, updated_merged_edge_points, updated_merged_clusters
    
    def extend_single_cluster_endpoints(self, merged_densified, merged_edge_points, merged_clusters, voxel_factor=2.5):
        """
        Extend single-cluster edge points by 1 voxel size in the direction from the closest densified point (excluding itself) to the edge point.
        """
        extended_densified = merged_densified.copy()

        for k, skel_pts in merged_densified.items():
            # Find edge points that belong only to this cluster
            single_cluster_edge_points = []
            for i, clusters in enumerate(merged_clusters):
                if len(clusters) == 1 and clusters[0] == k:
                    single_cluster_edge_points.append(merged_edge_points[i])

            if not single_cluster_edge_points or len(skel_pts) < 2:
                continue

            single_cluster_edge_points = np.array(single_cluster_edge_points)
            extended_points = []

            # For each edge point, find the closest other densified point (not itself)
            for edge_point in single_cluster_edge_points:
                # Exclude the edge point itself from the search
                other_skel_pts = skel_pts[np.any(np.abs(skel_pts - edge_point) > 1e-8, axis=1)]
                if len(other_skel_pts) == 0:
                    continue
                dists = np.linalg.norm(other_skel_pts - edge_point, axis=1)
                closest_pt = other_skel_pts[np.argmin(dists)]
                direction = edge_point - closest_pt
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                    extension_point = edge_point + direction * voxel_factor * self.voxel_size
                    extended_points.append(extension_point)
            if extended_points:
                extended_points = np.array(extended_points)
                current_points = extended_densified[k]
                extended_densified[k] = np.vstack([current_points, extended_points])

        return extended_densified

        
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
        self.last_skeleton_msg = None
        self.last_centroids_msg = None

    @staticmethod
    def distinct_colors(n):
        """Return n visually distinct RGB colors as uint8."""
        cmap = plt.get_cmap('tab20' if n <= 20 else 'hsv')
        colors = (np.array([cmap(i / n)[:3] for i in range(n)]) * 255).astype(np.uint8)
        return colors
    @staticmethod
    def pack_rgb(r, g, b):
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]

    def callback(self, msg):
        points = np.asarray(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points.dtype.fields is not None:
            points = np.stack([points['x'], points['y'], points['z']], axis=-1)

        # Early return if point count matches last published
        if (
            points.shape[0] == self.last_point_count
            and self.last_centroids_msg is not None
            and self.last_skeleton_msg is not None
        ):
            self.centroids_pub.publish(self.last_centroids_msg)
            self.skeleton_pub.publish(self.last_skeleton_msg)
            return

        points = self.skel.filter_lonely_points(points, min_cluster_size=100, eps_factor=5.0)
        if len(points) == 0:
            self.get_logger().warn("Received empty point cloud after filtering.")
            return

        labels, best_k = self.skel.cluster_detection(points, dilation_voxels=1, min_cluster_size=50)
        raw_edge_points, raw_edge_clusters, raw_centroids = self.skel.extract_edges_and_centroids(points, labels, best_k)
        merged_edge_points, merged_clusters = self.skel.merge_points_within_clusters(raw_edge_points, raw_edge_clusters, points)
        densified = self.skel.densify_skeleton(merged_edge_points, merged_clusters, points, labels, max_dist=5)
        merged_densified, updated_merged_edge_points, updated_merged_clusters = self.skel.merge_skeleton_points(
            densified, merged_edge_points, merged_clusters
        )
        extended_densified = self.skel.extend_single_cluster_endpoints(
            merged_densified,  updated_merged_edge_points, updated_merged_clusters, voxel_factor=2.5
        )

        # Combine edge points and centroids
        combined_points = np.vstack([merged_edge_points, raw_centroids])
        edge_rgb = np.array([255, 0, 0], dtype=np.uint8)
        centroid_rgb = np.array([0, 128, 255], dtype=np.uint8)
        combined_colors = np.vstack([
            np.tile(edge_rgb, (len(merged_edge_points), 1)),
            np.tile(centroid_rgb, (len(raw_centroids), 1))
        ])

        # Publish edge points and centroids (combined)
        if combined_points.shape[0] > 0:
            structured_combined = np.zeros(combined_points.shape[0], dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('rgb', np.float32)
            ])
            structured_combined['x'] = combined_points[:, 0]
            structured_combined['y'] = combined_points[:, 1]
            structured_combined['z'] = combined_points[:, 2]
            structured_combined['rgb'] = [self.pack_rgb(*c) for c in combined_colors]

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
            self.last_centroids_msg = pc2_msg
            self.get_logger().info(f"Published {len(raw_edge_points)} edge points and {len(raw_centroids)} centroids.")

        # Publish densified skeleton with unique color per cluster
        if extended_densified:
            all_skel_points = []
            all_skel_colors = []
            color_map = self.distinct_colors(max(extended_densified.keys()) + 1)
            for k, pts in extended_densified.items():
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
            structured_skel['rgb'] = [self.pack_rgb(*c) for c in all_skel_colors]
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            header = msg.header
            skel_msg = pc2.create_cloud(header, fields, structured_skel)
            self.skeleton_pub.publish(skel_msg)
            self.last_skeleton_msg = skel_msg
            self.get_logger().info(f"Published densified skeleton with {all_skel_points.shape[0]} points.")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeSkeletonizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()