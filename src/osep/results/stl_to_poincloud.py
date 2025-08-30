import open3d as o3d
import os
import numpy as np
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))
stl_dir = os.path.join(os.path.dirname(script_dir), "stl_models")
output_dir = script_dir
voxel_size = 0.25  # Adjust as needed

neighbor_offsets = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
])

for fname in os.listdir(stl_dir):
    if not fname.lower().endswith(".stl"):
        continue
    stl_path = os.path.join(stl_dir, fname)
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if not mesh.has_triangles():
        print(f"Skipping {fname}: no triangles.")
        continue

    # Voxelize mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print(f"Skipping {fname}: no voxels (try smaller voxel_size).")
        continue

    # Build set of voxel indices for fast lookup
    voxel_indices = set(tuple(voxel.grid_index) for voxel in voxels)
    voxel_indices_arr = np.array(list(voxel_indices))
    min_idx = voxel_indices_arr.min(axis=0) - 1
    max_idx = voxel_indices_arr.max(axis=0) + 2

    # Build a 3D grid: 1=filled, 0=empty, 2=exterior air
    grid_shape = tuple(max_idx - min_idx)
    grid = np.zeros(grid_shape, dtype=np.uint8)
    for idx in voxel_indices:
        grid_idx = tuple(np.array(idx) - min_idx)
        grid[grid_idx] = 1

    # Flood fill exterior air
    queue = deque()
    visited = set()
    for x in [0, grid_shape[0]-1]:
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                queue.append((x, y, z))
    for y in [0, grid_shape[1]-1]:
        for x in range(grid_shape[0]):
            for z in range(grid_shape[2]):
                queue.append((x, y, z))
    for z in [0, grid_shape[2]-1]:
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                queue.append((x, y, z))
    while queue:
        idx = queue.popleft()
        if idx in visited:
            continue
        visited.add(idx)
        if grid[idx] != 0:
            continue
        grid[idx] = 2  # Mark as exterior air
        for offset in neighbor_offsets:
            nidx = tuple(np.array(idx) + offset)
            if all(0 <= nidx[d] < grid_shape[d] for d in range(3)):
                queue.append(nidx)

    # Keep only voxels adjacent to exterior air
    surface_points = []
    for idx in voxel_indices:
        grid_idx = tuple(np.array(idx) - min_idx)
        is_shell = False
        for offset in neighbor_offsets:
            nidx = tuple(np.array(grid_idx) + offset)
            if all(0 <= nidx[d] < grid_shape[d] for d in range(3)):
                if grid[nidx] == 2:
                    is_shell = True
                    break
        if is_shell:
            pt = np.array(idx) * voxel_size + voxel_grid.origin + voxel_size / 2
            surface_points.append(pt)

    if not surface_points:
        print(f"Skipping {fname}: no shell voxels found.")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)

    # Save as PCD
    out_name = os.path.splitext(fname)[0] + f"_outer_shell_voxels_{voxel_size}.pcd"
    out_path = os.path.join(output_dir, out_name)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved {out_path} with {len(surface_points)} outer shell points.")