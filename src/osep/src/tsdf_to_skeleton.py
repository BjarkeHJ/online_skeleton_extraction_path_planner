#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt

from sklearn.neighbors import radius_neighbors_graph
from scipy.optimize import linear_sum_assignment
import colorsys


class Skeletonizer:
    def __init__(self, voxel_size=1.0, super_voxel_factor=4.0,
                 max_edge_points=10, dot_threshold=0.8, min_dist_factor=8.0, max_dist_percentage=0.25, max_clusters=20,
                 merge_radius_factor=5.0):
        self.voxel_size = voxel_size
        self.super_voxel_size = super_voxel_factor * voxel_size
        self.max_edge_points = max_edge_points
        self.dot_threshold = dot_threshold
        self.min_dist_factor = min_dist_factor
        self.max_dist_percentage = max_dist_percentage
        self.max_clusters = max_clusters
        self.merge_radius_factor = merge_radius_factor

        self.last_k = 0
        self.k_stability_counter = 0
        self.k_stable_epochs = 0
        self.k_min_switch = 2
        self.k_max_switch = 5
        self.k_hysteresis_factor_up = 0.10   # Easier to increase
        self.k_hysteresis_factor_down = 0.5  # Harder to decrease
     
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

    def full_dilation(self, points, dilation_voxels=1):
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

    def _update_stable_k(self, detected_k):
        if not hasattr(self, "previous_k"):
            self.previous_k = None

        if self.last_k == 0:
            self.last_k = detected_k
            self.k_stability_counter = 0
            self.k_stable_epochs = 1
            self.previous_k = detected_k
        elif detected_k == self.last_k:
            self.k_stability_counter = 0
            self.k_stable_epochs += 1
            self.previous_k = detected_k
        else:
            if detected_k == self.previous_k:
                self.k_stability_counter += 1
            else:
                self.k_stability_counter = 1
                self.previous_k = detected_k

            # Use different hysteresis factors for up/down
            if detected_k > self.last_k:
                hysteresis_factor = self.k_hysteresis_factor_up
            else:
                hysteresis_factor = self.k_hysteresis_factor_down

            dynamic_threshold = min(
                self.k_max_switch,
                max(self.k_min_switch, int(hysteresis_factor * self.k_stable_epochs))
            )
            if self.k_stability_counter >= dynamic_threshold:
                print(f"Switching cluster count from {self.last_k} to {detected_k} after {self.k_stability_counter} consecutive detections (threshold was {dynamic_threshold})")
                self.last_k = detected_k
                self.k_stable_epochs = 1
                self.k_stability_counter = 0

        print(f"Stable cluster count: {self.last_k} (detected: {detected_k}, stable_epochs: {self.k_stable_epochs}, switch_counter: {self.k_stability_counter})")
        return self.last_k

    def cluster_detection(self, points, dilated_points, min_cluster_size=50):
        print("🔄 Fitting GMM models (iterative elbow detection)...")
        bics, models = [], []
        threshold = 0.01  # 1% relative improvement cutoff
        elbow_idx = None
        skip_processing_publishes_last_msg = False
        for k in range(1, self.max_clusters + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(dilated_points)
            bics.append(gmm.bic(dilated_points))
            models.append(gmm)
            if k > 1:
                improvement = -(bics[-1] - bics[-2]) / abs(bics[-2])
                if improvement < threshold and elbow_idx is None:
                    elbow_idx = k - 1
                    print(f"Elbow detected at k={elbow_idx} (breaking point, improvement={improvement:.4f})")
                    continue
                if (k >= self.last_k) and (elbow_idx is not None):
                    break
        if elbow_idx is None:
            elbow_idx = int(np.argmin(bics)) + 1
        bics = np.array(bics)
        detected_k = elbow_idx
        best_k = self._update_stable_k(detected_k)

        best_gmm = models[best_k - 1]

        # Assign labels for both original and dilated points
        labels = best_gmm.predict(points)
        dilated_labels = best_gmm.predict(dilated_points)
        if best_k != detected_k:
            print(f"Using stable cluster count k={best_k} instead of detected k={detected_k}")
            skip_processing_publishes_last_msg = True
            return labels, dilated_labels, best_k, skip_processing_publishes_last_msg

        # Optionally: split each cluster into connected components (on original points)
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
                    continue
                final_labels[cluster_indices] = next_label
                next_label += 1
        labels = final_labels
        best_k = next_label

        print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")
        return labels, dilated_labels, best_k, skip_processing_publishes_last_msg

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
        max_dist = np.max(dists)
        min_dist = max(min_dist, self.max_dist_percentage * max_dist)

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
    
    
    def _adaptive_thin_path(self, path_pts, voxel_size,
                        step_len_straight=5.0,
                        step_len_curve=1.5,
                        curvature_threshold=0.15):
        """
        Thin a polyline adaptively: keep more points in curves, fewer in straight segments.
        Args:
            path_pts: (N, 3) array of points along the path
            voxel_size: float, base voxel size
            step_len_straight: float, spacing in straight regions (in voxel_size units)
            step_len_curve: float, spacing in curved regions (in voxel_size units)
            curvature_threshold: float, angle threshold (radians) to detect curves
        Returns:
            np.ndarray of thinned points
        """
        if len(path_pts) < 2:
            return path_pts

        step_straight = step_len_straight * voxel_size
        step_curve = step_len_curve * voxel_size

        thinned = [path_pts[0]]
        acc = 0.0
        for i in range(1, len(path_pts) - 1):
            a, b, c = path_pts[i - 1], path_pts[i], path_pts[i + 1]
            v1 = b - a
            v2 = c - b
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
            else:
                angle = 0.0

            seg = np.linalg.norm(b - thinned[-1])
            acc += seg

            step = step_curve if angle > curvature_threshold else step_straight

            if acc >= step:
                thinned.append(b)
                acc = 0.0

        if len(path_pts) > 1:
            thinned.append(path_pts[-1])

        thinned = np.array(thinned)
        if len(thinned) > 1:
            thinned = np.unique(thinned, axis=0)
        return thinned


    def densify_skeleton(self,
                        merged_edge_points, merged_clusters,
                        points, labels,
                        max_graph_radius_factor=3.5,
                        min_points_for_skeleton=10,
                        step_len_straight=5.0,
                        step_len_curve=1.5,
                        curvature_threshold=0.15):
        """
        Densify skeleton by following the *occupied* pointcloud via geodesic paths.
        - For each cluster, build a radius-neighbors graph (on that cluster's points).
        - Compute geodesic (graph) distances between edge points.
        - Build MST on those geodesic distances.
        - Recover actual shortest paths for each MST edge and stitch them.
        Args:
            merged_edge_points: (M, 3) final edge points after intra-cluster merging
            merged_clusters: list of lists, cluster associations per edge point
            points, labels: use *dilated* points/labels for routing (pass in dilated_* from caller)
            max_graph_radius_factor: radius (in voxels) for graph edges = factor * self.voxel_size
            min_points_for_skeleton: minimum cluster size to attempt skeletonization
            step_len_straight: spacing in straight regions (in voxel_size units)
            step_len_curve: spacing in curved regions (in voxel_size units)
            curvature_threshold: angle threshold (radians) to detect curves
        Returns:
            dict: cluster_id -> (N_i, 3) np.array of skeleton points following the cloud
        """
        densified = {}

        # Map: for each cluster, collect its edge points (from merged_edge_points & merged_clusters)
        cluster_to_edgepts = {}
        for i, clist in enumerate(merged_clusters):
            for k in clist:
                cluster_to_edgepts.setdefault(k, []).append(merged_edge_points[i])

        # For each cluster, build geodesic skeleton
        for k in cluster_to_edgepts.keys():
            # Get cluster points from (preferably) dilated cloud
            cluster_mask = (labels == k)
            cluster_points = points[cluster_mask]
            if len(cluster_points) < min_points_for_skeleton:
                continue

            edge_pts = np.array(cluster_to_edgepts[k], dtype=float)
            if edge_pts.shape[0] < 2:
                continue

            # Map edge points to nearest nodes (indices) in the cluster graph
            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(cluster_points)
            edge_dists, edge_idxs = nbrs.kneighbors(edge_pts, return_distance=True)
            edge_node_indices = edge_idxs.flatten()

            # Build a radius graph on cluster points (weighted by Euclidean distance)
            graph_radius = max_graph_radius_factor * self.voxel_size
            G = radius_neighbors_graph(cluster_points, radius=graph_radius, mode='distance', include_self=False)
            if G.nnz == 0:
                # graph disconnected under this radius; skip
                continue

            # Geodesic distances between edge nodes (all-pairs via Dijkstra from each edge node)
            nE = len(edge_node_indices)
            pairwise_geo = np.full((nE, nE), np.inf, dtype=float)
            # Also keep predecessors to reconstruct actual paths
            predecessors_all = {}

            for i, src in enumerate(edge_node_indices):
                dist_i, predecessors = dijkstra(G, directed=False, indices=src, return_predecessors=True)
                predecessors_all[src] = predecessors
                for j, dst in enumerate(edge_node_indices):
                    pairwise_geo[i, j] = dist_i[dst]

            # Some pairs may be disconnected (inf); keep only finite ones
            # Build MST over geodesic distances
            # (Set diagonal to 0 and symmetrize)
            np.fill_diagonal(pairwise_geo, 0.0)
            pairwise_geo = np.minimum(pairwise_geo, pairwise_geo.T)
            mst = minimum_spanning_tree(pairwise_geo)
            mst_edges = np.array(mst.nonzero()).T  # edges on index range [0..nE)

            # Recover actual node sequences for each MST edge using predecessors
            path_node_indices = []
            for ii, jj in mst_edges:
                src_node = int(edge_node_indices[int(ii)])
                dst_node = int(edge_node_indices[int(jj)])

                # Run Dijkstra once from src_node with predecessors to ensure consistency
                dist_src, predecessors = dijkstra(G, directed=False, indices=src_node, return_predecessors=True)
                if not np.isfinite(dist_src[dst_node]):
                    # No path; skip this edge
                    continue

                # Reconstruct path nodes by backtracking predecessors
                path = []
                cur = dst_node
                while cur != -9999 and cur != -1 and cur != src_node:
                    path.append(cur)
                    cur = predecessors[cur]
                path.append(src_node)
                path = path[::-1]  # src -> ... -> dst
                path_node_indices.append(path)

            if not path_node_indices:
                continue

            # Concatenate unique nodes from all edge paths
            all_nodes = []
            for seq in path_node_indices:
                all_nodes.extend(seq)

            if not all_nodes:
                continue

            # Turn node indices into coordinates
            path_pts = cluster_points[np.array(all_nodes, dtype=int), :]

            # Adaptive thinning based on curvature
            thinned = self._adaptive_thin_path(
                path_pts,
                self.voxel_size,
                step_len_straight=step_len_straight,
                step_len_curve=step_len_curve,
                curvature_threshold=curvature_threshold
            )

            if len(thinned) >= 2:
                densified[k] = thinned

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
        
    def merge_edge_skeleton_points(self, densified, merged_edge_points, merged_clusters):
        """
        Merge edge points in the skeleton, including those from different clusters if nearby
        Returns: merged_densified, updated_merged_edge_points, updated_merged_clusters
        """        
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

    def extend_single_cluster_endpoints(self, merged_densified, merged_edge_points, merged_clusters, voxel_factor=5.0):
        """
        Extend single-cluster edge points by voxel_factor size in the direction from the closest densified point (excluding itself) to the edge point.
        Ensures the original edge point is also present in the skeleton.
        Adds points every 2 meters up to voxel_factor * voxel_size.
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

            for edge_point in single_cluster_edge_points:
                # Ensure the edge point is present in the skeleton
                if not np.any(np.all(np.isclose(skel_pts, edge_point, atol=1e-8), axis=1)):
                    extended_points.append(edge_point)
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
                    max_dist = voxel_factor * self.voxel_size
                    n_points = int(np.floor(max_dist / 2.0))
                    for i in range(1, n_points + 1):
                        extension_point = edge_point + direction * (i * 2.0)
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
        self.declare_parameter('voxel_size', 1.0)

        self.input_topic = self.get_parameter('static_input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value

        self.skel = Skeletonizer(voxel_size=self.voxel_size, super_voxel_factor=4.0,
                 max_edge_points=10, dot_threshold=0.7, min_dist_factor=5.0, max_dist_percentage=0.25, max_clusters=20,
                 merge_radius_factor=5.0)

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, 1)
        self.skeleton_pub = self.create_publisher(PointCloud2, self.output_topic, 1)
        self.centroids_pub = self.create_publisher(PointCloud2, self.output_topic + "/centroids", 1)

        self.last_point_count = 0
        self.last_skeleton_msg = None
        self.last_centroids_msg = None

        self.tick = 0
        self.next_cluster_id = 0
        self.tracks = {} # id -> {"centroid": np.array([x,y,z]), "age": int, "miss": int}
        self.id_to_color = {}
        self.max_miss = 15

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

    def ensure_color(self, cid): #bhj
        if cid not in self.id_to_color:
            golden = 0.61803398875
            h = (cid * golden) % 1.0          # well-spaced hue around the circle
            s = 0.92
            # small deterministic value wobble to separate near hues in dense id ranges
            v_base = 0.98
            v = v_base - 0.08 * (((cid * 97) % 5) / 4.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            self.id_to_color[cid] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        return self.id_to_color[cid]

    def assign_stable_id(self, curr_centroids, dist_gate=5.0): #bhj
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[i]["centroid"] for i in track_ids]) if track_ids else np.empty((0,3))

        valid_curr = [(i,c) for i,c in enumerate(curr_centroids) if c is not None]
        if not valid_curr:
            for i in track_ids:
                self.tracks[i]["miss"] += 1
            return {}
        
        idxs_curr = [i for i,_ in valid_curr]
        curr_mat = np.vstack([c for _,c in valid_curr])

        if len(track_ids) and len(curr_mat):
            dists = np.linalg.norm(curr_mat[:,None,:] - track_centroids[None,:,:], axis=2)
            row_ind, col_ind = linear_sum_assignment(dists)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        
        k_to_id = {}
        assigned_tracks = set()
        assigned_curr = set()

        for r, c in zip(row_ind, col_ind):
            if dists[r, c] <= dist_gate:
                k = idxs_curr[r]
                tid = track_ids[c]
                k_to_id[k] = tid
                assigned_tracks.add(tid)
                assigned_curr.add(k)

                self.tracks[tid]["centroid"] = curr_mat[r]
                self.tracks[tid]["age"] += 1
                self.tracks[tid]["miss"] = 0
        
        # new tracks / unmatched clusters
        for k, c in valid_curr:
            if k in assigned_curr:
                continue
            tid = self.next_cluster_id
            self.next_cluster_id += 1
            self.tracks[tid] = {"centroid": c, "age": 1, "miss": 0}
            k_to_id[k] = tid
        
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]["miss"] += 1
        
        stale = [tid for tid,v in self.tracks.items() if v["miss"] > self.max_miss]
        for tid in stale:
            del self.tracks[tid]
        
        return k_to_id


    def pack_points_with_colors(self, points, colors):
        """Pack points and uint8 RGB colors into a structured array for PointCloud2."""
        structured = np.zeros(points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('rgb', np.float32)
        ])
        structured['x'] = points[:, 0]
        structured['y'] = points[:, 1]
        structured['z'] = points[:, 2]
        structured['rgb'] = [self.pack_rgb(*c) for c in colors]
        return structured

    def publish_pointcloud(self, points, colors, header, publisher):
        """Create and publish a PointCloud2 message from points and colors."""
        if points.shape[0] == 0:
            return
        structured = self.pack_points_with_colors(points, colors)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg = pc2.create_cloud(header, fields, structured)
        publisher.publish(pc2_msg)
        return pc2_msg

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
        else:
            self.last_point_count = points.shape[0]

        points = self.skel.filter_lonely_points(points, min_cluster_size=50, eps_factor=5.0)
        if len(points) == 0:
            self.get_logger().warn("Received empty point cloud after filtering.")
            return

        dilated_points = self.skel.full_dilation(points, dilation_voxels=1)
        labels, dilated_labels, best_k, skip_processing_publishes_last_msg = self.skel.cluster_detection(points, dilated_points, min_cluster_size=50)
        
        if skip_processing_publishes_last_msg:
            if self.last_centroids_msg is not None:
                self.centroids_pub.publish(self.last_centroids_msg)
            if self.last_skeleton_msg is not None:
                self.skeleton_pub.publish(self.last_skeleton_msg)
            return

        # --- Stable IDs using raw_centroids ---
        raw_edge_points, raw_edge_clusters, raw_centroids = self.skel.extract_edges_and_centroids(points, labels, best_k)
        k_to_id = self.assign_stable_id(raw_centroids, dist_gate=5.0 * self.skel.voxel_size)
        global_labels = np.full_like(labels, -1)
        for k in range(best_k):
            pid = k_to_id.get(k, None)
            if pid is not None:
                global_labels[labels == k] = pid
        # --------------------------------------

        merged_edge_points, merged_clusters = self.skel.merge_points_within_clusters(raw_edge_points, raw_edge_clusters, points)
        densified = self.skel.densify_skeleton(
            merged_edge_points, merged_clusters, points, labels,
            max_graph_radius_factor=3.5,
            min_points_for_skeleton=10,
            step_len_straight=4.0,
            step_len_curve=2.0,
            curvature_threshold=0.5
        )
        merged_densified, updated_merged_edge_points, updated_merged_clusters = self.skel.merge_edge_skeleton_points(
            densified, merged_edge_points, merged_clusters
        )
        extended_densified = self.skel.extend_single_cluster_endpoints(
            merged_densified, updated_merged_edge_points, updated_merged_clusters, voxel_factor=4.0
        )

        # Combine edge points and centroids for visualization
        if merged_edge_points.size == 0:
            merged_edge_points = np.empty((0, 3))
        if raw_centroids.size == 0:
            raw_centroids = np.empty((0, 3))

        combined_points = np.vstack([merged_edge_points, raw_centroids])
        edge_rgb = np.array([255, 0, 0], dtype=np.uint8)
        centroid_rgb = np.array([0, 128, 255], dtype=np.uint8)
        combined_colors = np.vstack([
            np.tile(edge_rgb, (len(merged_edge_points), 1)),
            np.tile(centroid_rgb, (len(raw_centroids), 1))
        ])

        # Publish edge points and centroids (combined)
        if combined_points.shape[0] > 0:
            pc2_msg = self.publish_pointcloud(
                combined_points, combined_colors, msg.header, self.centroids_pub
            )
            self.last_centroids_msg = pc2_msg

        # Publish densified skeleton with unique color per cluster (using stable IDs)
        if extended_densified:
            all_skel_points = []
            all_skel_colors = []
            for k, pts in extended_densified.items():
                pid = k_to_id.get(k, None)
                if pid is None:
                    continue
                all_skel_points.append(pts)
                all_skel_colors.append(np.tile(self.ensure_color(pid), (pts.shape[0], 1)))
            if all_skel_points:
                all_skel_points = np.vstack(all_skel_points)
                all_skel_colors = np.vstack(all_skel_colors)
                skel_msg = self.publish_pointcloud(
                    all_skel_points, all_skel_colors, msg.header, self.skeleton_pub
                )
                self.last_skeleton_msg = skel_msg

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeSkeletonizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()