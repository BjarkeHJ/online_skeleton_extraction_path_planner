import open3d as o3d
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
stl_dir = os.path.join(os.path.dirname(script_dir), "stl_models")
output_dir = script_dir
voxel_size = 0.1  # Adjust as needed

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

    # Use voxel centers as points
    points = [voxel.grid_index * voxel_size + voxel_grid.origin + voxel_size / 2 for voxel in voxels]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save as PCD
    out_name = os.path.splitext(fname)[0] + f"_voxels_{voxel_size}.pcd"
    out_path = os.path.join(output_dir, out_name)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved {out_path} with {len(points)} points.")