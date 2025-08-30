#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <filesystem>

namespace fs = std::filesystem;

// Helper for hashing 3D indices
struct VoxelIndex {
    int x, y, z;
    bool operator==(const VoxelIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};
namespace std {
    template <>
    struct hash<VoxelIndex> {
        std::size_t operator()(const VoxelIndex& k) const {
            return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
        }
    };
}

// Uniformly sample points on a triangle mesh
pcl::PointCloud<pcl::PointXYZ>::Ptr sample_mesh_surface(const aiMesh* mesh, int samples_per_triangle) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
        const aiFace& face = mesh->mFaces[f];
        if (face.mNumIndices != 3) continue;
        aiVector3D v0 = mesh->mVertices[face.mIndices[0]];
        aiVector3D v1 = mesh->mVertices[face.mIndices[1]];
        aiVector3D v2 = mesh->mVertices[face.mIndices[2]];
        for (int i = 0; i < samples_per_triangle; ++i) {
            float r1 = std::sqrt(dist(gen));
            float r2 = dist(gen);
            float a = 1 - r1;
            float b = r1 * (1 - r2);
            float c = r1 * r2;
            float x = a * v0.x + b * v1.x + c * v2.x;
            float y = a * v0.y + b * v1.y + c * v2.y;
            float z = a * v0.z + b * v1.z + c * v2.z;
            cloud->points.emplace_back(x, y, z);
        }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

int main(int argc, char** argv) {
    std::string script_dir = fs::path(argv[0]).parent_path();
    std::string stl_dir = (fs::path(script_dir) / "../stl_models").string();
    std::string output_dir = script_dir + "/";

    float voxel_size = 0.1f;
    if (argc > 1) {
        voxel_size = std::stof(argv[1]);
        std::cout << "Using voxel size: " << voxel_size << std::endl;
    }

    std::vector<std::vector<int>> neighbor_offsets = {
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}
    };

    for (const auto& entry : fs::directory_iterator(stl_dir)) {
        if (entry.path().extension() != ".stl") continue;
        std::cout << "Processing: " << entry.path().filename() << std::endl;

        // Load mesh with Assimp
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(entry.path().string(), aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
        if (!scene || !scene->HasMeshes()) continue;

        // Sample mesh surface
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
            const aiMesh* mesh = scene->mMeshes[m];
            auto sampled = sample_mesh_surface(mesh, 2000); // 2000 samples per triangle
            *cloud += *sampled;
        }
        if (cloud->empty()) continue;

        // Find min point (origin)
        float min_x = std::numeric_limits<float>::max(), min_y = min_x, min_z = min_x;
        for (const auto& pt : cloud->points) {
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            min_z = std::min(min_z, pt.z);
        }

        // Voxelization
        std::unordered_set<VoxelIndex> voxel_indices;
        int min_ix=INT_MAX, min_iy=INT_MAX, min_iz=INT_MAX;
        int max_ix=INT_MIN, max_iy=INT_MIN, max_iz=INT_MIN;
        for (const auto& pt : cloud->points) {
            int ix = static_cast<int>(std::floor((pt.x - min_x) / voxel_size));
            int iy = static_cast<int>(std::floor((pt.y - min_y) / voxel_size));
            int iz = static_cast<int>(std::floor((pt.z - min_z) / voxel_size));
            voxel_indices.insert({ix, iy, iz});
            min_ix = std::min(min_ix, ix); min_iy = std::min(min_iy, iy); min_iz = std::min(min_iz, iz);
            max_ix = std::max(max_ix, ix); max_iy = std::max(max_iy, iy); max_iz = std::max(max_iz, iz);
        }
        min_ix--; min_iy--; min_iz--;
        max_ix++; max_iy++; max_iz++;

        int sx = max_ix - min_ix + 1, sy = max_iy - min_iy + 1, sz = max_iz - min_iz + 1;
        std::vector<std::vector<std::vector<uint8_t>>> grid(
            sx, std::vector<std::vector<uint8_t>>(sy, std::vector<uint8_t>(sz, 0)));

        for (const auto& idx : voxel_indices) {
            int gx = idx.x - min_ix, gy = idx.y - min_iy, gz = idx.z - min_iz;
            grid[gx][gy][gz] = 1;
        }

        // Flood fill exterior air
        std::queue<std::tuple<int,int,int>> q;
        for (int x : {0, sx-1}) for (int y=0; y<sy; ++y) for (int z=0; z<sz; ++z) q.emplace(x, y, z);
        for (int y : {0, sy-1}) for (int x=0; x<sx; ++x) for (int z=0; z<sz; ++z) q.emplace(x, y, z);
        for (int z : {0, sz-1}) for (int x=0; x<sx; ++x) for (int y=0; y<sy; ++y) q.emplace(x, y, z);

        while (!q.empty()) {
            auto [x, y, z] = q.front(); q.pop();
            if (x < 0 || x >= sx || y < 0 || y >= sy || z < 0 || z >= sz) continue;
            if (grid[x][y][z] != 0) continue;
            grid[x][y][z] = 2;
            for (const auto& off : neighbor_offsets) {
                int nx = x + off[0], ny = y + off[1], nz = z + off[2];
                q.emplace(nx, ny, nz);
            }
        }

        // Keep only voxels adjacent to exterior air
        pcl::PointCloud<pcl::PointXYZ>::Ptr shell_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : voxel_indices) {
            int gx = idx.x - min_ix, gy = idx.y - min_iy, gz = idx.z - min_iz;
            bool is_shell = false;
            for (const auto& off : neighbor_offsets) {
                int nx = gx + off[0], ny = gy + off[1], nz = gz + off[2];
                if (nx < 0 || nx >= sx || ny < 0 || ny >= sy || nz < 0 || nz >= sz || grid[nx][ny][nz] == 2) {
                    is_shell = true;
                    break;
                }
            }
            if (is_shell) {
                float x = min_x + (idx.x + 0.5f) * voxel_size;
                float y = min_y + (idx.y + 0.5f) * voxel_size;
                float z = min_z + (idx.z + 0.5f) * voxel_size;
                shell_cloud->points.emplace_back(x, y, z);
            }
        }

        if (shell_cloud->empty()) {
            std::cout << "No shell voxels found for " << entry.path().filename() << std::endl;
            continue;
        }

        char voxel_size_str[16];
        std::snprintf(voxel_size_str, sizeof(voxel_size_str), "%.1f", voxel_size);
        std::string out_name = entry.path().stem().string() + "_outer_shell_voxels_" + voxel_size_str + ".pcd";
        std::string out_path = output_dir + out_name;
        pcl::io::savePCDFileBinary(out_path, *shell_cloud);
        std::cout << "Saved " << out_path << " with " << shell_cloud->size() << " outer shell points." << std::endl;
    }
    return 0;
}