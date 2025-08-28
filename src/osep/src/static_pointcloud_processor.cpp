#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <map>
#include <set>
#include <Eigen/Core>
#include <tuple>
#include <cmath>

class StaticPointcloudPostprocessNode : public rclcpp::Node
{
public:
    StaticPointcloudPostprocessNode()
    : Node("static_pointcloud_postprocess_node")
    {
        this->declare_parameter<std::string>("static_input_topic", "osep/tsdf/static_pointcloud");
        this->declare_parameter<std::string>("output_topic", "osep/tsdf/upsampled_static_pointcloud");
        this->declare_parameter<double>("voxel_size", 1.0);  // 1.0 m by default
        this->declare_parameter<int>("upsample_N", 4);       // Default upsampling factor

        static_input_topic_ = this->get_parameter("static_input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        voxel_size_ = this->get_parameter("voxel_size").as_double();
        upsample_N_ = this->get_parameter("upsample_N").as_int();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            static_input_topic_, 1,
            std::bind(&StaticPointcloudPostprocessNode::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 1);
    }

private:
    std::string static_input_topic_;
    std::string output_topic_;
    double voxel_size_;
    int upsample_N_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    size_t last_point_count_ = 0;
    sensor_msgs::msg::PointCloud2 last_msg_;

    // Quantize a point to voxel index
    static inline std::tuple<int, int, int> quantize(float x, float y, float z, float res) {
        return std::make_tuple(
            static_cast<int>(std::floor(x / res)),
            static_cast<int>(std::floor(y / res)),
            static_cast<int>(std::floor(z / res)));
    }

    // Get voxel center from index
    static inline Eigen::Vector3f voxel_center(int ix, int iy, int iz, float res) {
        return Eigen::Vector3f(
            (ix + 0.5f) * res,
            (iy + 0.5f) * res,
            (iz + 0.5f) * res
        );
    }

    // Helper: Upsample a voxel face into a grid of points
    void upsample_voxel_face(
        const Eigen::Vector3f& voxel_center,
        float voxel_size,
        int upsample_N,
        int face, // 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>& out_points)
    {
        float half = voxel_size / 2.0f;
        float step = voxel_size / upsample_N;
        Eigen::Vector3f face_center = voxel_center;
        Eigen::Vector3f u, v, normal;
        switch(face) {
            case 0: face_center.x() += half; u = {0, 1, 0}; v = {0, 0, 1}; normal = {1, 0, 0}; break; // +x
            case 1: face_center.x() -= half; u = {0, 1, 0}; v = {0, 0, 1}; normal = {-1, 0, 0}; break; // -x
            case 2: face_center.y() += half; u = {1, 0, 0}; v = {0, 0, 1}; normal = {0, 1, 0}; break; // +y
            case 3: face_center.y() -= half; u = {1, 0, 0}; v = {0, 0, 1}; normal = {0, -1, 0}; break; // -y
            case 4: face_center.z() += half; u = {1, 0, 0}; v = {0, 1, 0}; normal = {0, 0, 1}; break; // +z
            case 5: face_center.z() -= half; u = {1, 0, 0}; v = {0, 1, 0}; normal = {0, 0, -1}; break; // -z
        }
        // The lower-left corner of the face
        Eigen::Vector3f corner = face_center - 0.5f * voxel_size * u - 0.5f * voxel_size * v;
        for (int i = 0; i < upsample_N; ++i) {
            for (int j = 0; j < upsample_N; ++j) {
                // Center each mini-voxel on the face, then move it half a step toward the voxel center
                Eigen::Vector3f pt = corner + (i + 0.5f) * step * u + (j + 0.5f) * step * v - 0.5f * step * normal;
                out_points.emplace_back(pt.x(), pt.y(), pt.z());
            }
        }
    }

    // Helper: Find connected voxel groups and keep only large ones
    std::set<std::tuple<int,int,int>> filter_sparse_voxel_groups(
        const std::set<std::tuple<int,int,int>>& occupied_voxels,
        int min_group_size)
    {
        std::set<std::tuple<int,int,int>> result;
        std::set<std::tuple<int,int,int>> visited;

        // Generate all 26 neighbor offsets (excluding 0,0,0)
        std::vector<std::tuple<int,int,int>> neighbors;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    neighbors.emplace_back(dx, dy, dz);
                }
            }
        }

        for (const auto& voxel : occupied_voxels) {
            if (visited.count(voxel)) continue;
            // BFS for connected component
            std::vector<std::tuple<int,int,int>> queue = {voxel};
            std::vector<std::tuple<int,int,int>> group;
            visited.insert(voxel);
            while (!queue.empty()) {
                auto v = queue.back();
                queue.pop_back();
                group.push_back(v);
                int ix = std::get<0>(v);
                int iy = std::get<1>(v);
                int iz = std::get<2>(v);
                for (const auto& n : neighbors) {
                    int nx = ix + std::get<0>(n);
                    int ny = iy + std::get<1>(n);
                    int nz = iz + std::get<2>(n);
                    auto nidx = std::make_tuple(nx, ny, nz);
                    if (occupied_voxels.count(nidx) && !visited.count(nidx)) {
                        visited.insert(nidx);
                        queue.push_back(nidx);
                    }
                }
            }
            if (group.size() >= static_cast<size_t>(min_group_size)) {
                result.insert(group.begin(), group.end());
            }
        }
        return result;
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Only process if the new point cloud has more points than the last one
        size_t current_count = msg->width * msg->height;
        if (current_count <= last_point_count_) {
            if (!last_msg_.data.empty()) {
                pub_->publish(last_msg_);
            }
            return;
        }
        // Write that we are processing as information
        RCLCPP_INFO(this->get_logger(), "Processing point cloud with %zu points", current_count);
        last_point_count_ = current_count;

        // 1. Voxelize input at voxel_size_
        std::map<std::tuple<int,int,int>, int> voxel_map;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            auto idx = quantize(*iter_x, *iter_y, *iter_z, voxel_size_);
            voxel_map[idx]++;
        }

        // 2. Find all occupied voxels (at least 1 point)
        std::set<std::tuple<int,int,int>> occupied_voxels;
        for (const auto& kv : voxel_map) {
            occupied_voxels.insert(kv.first);
        }

        // 3. Filter out small groups (keep only groups with at least 50 voxels)
        occupied_voxels = filter_sparse_voxel_groups(occupied_voxels, 50);

        // 4. For each voxel, check each face for neighbors
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> shell_points;
        const int dx[6] = {1, -1, 0, 0, 0, 0};
        const int dy[6] = {0, 0, 1, -1, 0, 0};
        const int dz[6] = {0, 0, 0, 0, 1, -1};
        for (const auto& idx : occupied_voxels) {
            int ix = std::get<0>(idx);
            int iy = std::get<1>(idx);
            int iz = std::get<2>(idx);
            Eigen::Vector3f center = voxel_center(ix, iy, iz, voxel_size_);
            for (int face = 0; face < 6; ++face) {
                auto nidx = std::make_tuple(ix + dx[face], iy + dy[face], iz + dz[face]);
                if (occupied_voxels.count(nidx) == 0) {
                    upsample_voxel_face(center, voxel_size_, upsample_N_, face, shell_points);
                }
            }
        }

        // 5. Output as point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_shell(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_shell->points = shell_points;
        cloud_shell->width = static_cast<uint32_t>(shell_points.size());
        cloud_shell->height = 1;
        cloud_shell->is_dense = true;

        // Apply voxel filter to remove duplicates/overlaps
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_shell);
        sor.setLeafSize(voxel_size_ / upsample_N_, voxel_size_ / upsample_N_, voxel_size_ / upsample_N_);
        sor.filter(*cloud_filtered);

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header = msg->header;
        pub_->publish(output);
        last_msg_ = output;
        last_point_count_ = msg->width * msg->height;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticPointcloudPostprocessNode>());
    rclcpp::shutdown();
    return 0;
}