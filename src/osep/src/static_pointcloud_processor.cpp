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
        this->declare_parameter<std::string>("output_topic", "osep/tsdf/shell_static_pointcloud");
        this->declare_parameter<double>("voxel_size", 1.0);  // 1.0 m by default
        this->declare_parameter<int>("upsample_N", 1);       // Number of upsample-opening cycles

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

    // Upsample voxels by factor 2
    std::set<std::tuple<int,int,int>> upsample_voxels_by_2(
        const std::set<std::tuple<int,int,int>>& voxels)
    {
        std::set<std::tuple<int,int,int>> upsampled;
        for (const auto& v : voxels) {
            int x = std::get<0>(v) * 2;
            int y = std::get<1>(v) * 2;
            int z = std::get<2>(v) * 2;
            for (int dx = 0; dx < 2; ++dx)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dz = 0; dz < 2; ++dz)
                        upsampled.insert({x+dx, y+dy, z+dz});
        }
        return upsampled;
    }

    std::set<std::tuple<int,int,int>> full_erosion(
        const std::set<std::tuple<int,int,int>>& voxels,
        int erosion_threshold)
    {
        std::set<std::tuple<int,int,int>> eroded;
        for (const auto& v : voxels) {
            int present = 0;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz)
                        if (!(dx == 0 && dy == 0 && dz == 0) &&
                            voxels.count({std::get<0>(v)+dx, std::get<1>(v)+dy, std::get<2>(v)+dz}))
                            ++present;
            if (present >= erosion_threshold)
                eroded.insert(v);
        }
        return eroded;
    }

    std::set<std::tuple<int,int,int>> full_dilation(
    const std::set<std::tuple<int,int,int>>& voxels)
    {
        std::set<std::tuple<int,int,int>> dilated;
        for (const auto& v : voxels) {
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz)
                        if (!(dx == 0 && dy == 0 && dz == 0)) // exclude the center if you want only neighbors
                            dilated.insert({std::get<0>(v)+dx, std::get<1>(v)+dy, std::get<2>(v)+dz});
            // Optionally, include the center voxel itself:
            dilated.insert(v);
        }
        return dilated;
    }

    std::set<std::tuple<int,int,int>> selective_dilation(
    const std::set<std::tuple<int,int,int>>& eroded)
    {
        std::set<std::tuple<int,int,int>> dilated = eroded; // Start with existing voxels

        // For each voxel in the eroded set, check its 6 neighbors
        for (const auto& v : eroded) {
            static const int dxs[6] = {1, -1, 0, 0, 0, 0};
            static const int dys[6] = {0, 0, 1, -1, 0, 0};
            static const int dzs[6] = {0, 0, 0, 0, 1, -1};
            for (int i = 0; i < 6; ++i) {
                std::tuple<int,int,int> neighbor{
                    std::get<0>(v)+dxs[i],
                    std::get<1>(v)+dys[i],
                    std::get<2>(v)+dzs[i]
                };
                // Only consider empty neighbors
                if (!eroded.count(neighbor)) {
                    // Count how many of the 6 neighbors of this empty voxel are filled
                    int filled_neighbors = 0;
                    for (int j = 0; j < 6; ++j) {
                        std::tuple<int,int,int> nn{
                            std::get<0>(neighbor)+dxs[j],
                            std::get<1>(neighbor)+dys[j],
                            std::get<2>(neighbor)+dzs[j]
                        };
                        if (eroded.count(nn)) ++filled_neighbors;
                    }
                    if (filled_neighbors >= 2) {
                        dilated.insert(neighbor);
                    }
                }
            }
        }
        return dilated;
    }

    // Extract outer shell voxels
    std::set<std::tuple<int,int,int>> extract_shell(
        const std::set<std::tuple<int,int,int>>& voxels)
    {
        std::set<std::tuple<int,int,int>> shell;
        for (const auto& v : voxels) {
            bool is_shell = false;
            for (int dx = -1; dx <= 1 && !is_shell; ++dx)
                for (int dy = -1; dy <= 1 && !is_shell; ++dy)
                    for (int dz = -1; dz <= 1 && !is_shell; ++dz)
                        if ((dx != 0 || dy != 0 || dz != 0) &&
                            !voxels.count({std::get<0>(v)+dx, std::get<1>(v)+dy, std::get<2>(v)+dz}))
                            is_shell = true;
            if (is_shell) shell.insert(v);
        }
        return shell;
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
        RCLCPP_INFO(this->get_logger(), "Processing point cloud with %zu points", current_count);
        last_point_count_ = current_count;

        // 1. Voxelize input at voxel_size_
        std::set<std::tuple<int,int,int>> voxels;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            voxels.insert(quantize(*iter_x, *iter_y, *iter_z, voxel_size_));
        }

        int N = upsample_N_;
        double res = voxel_size_;

        for (int n = 0; n < N; ++n) {
            res /= 2.0;
            voxels = upsample_voxels_by_2(voxels);
            // opening
            if (n <= 1) {
                voxels = selective_dilation(voxels);
            } else {
                voxels = full_erosion(voxels, 26);
                voxels = full_dilation(voxels);
            }
            if (n == N-1) {
                voxels = full_erosion(voxels, 10);
                voxels = selective_dilation(voxels);
            }
        }

        // Extract shell at final resolution
        std::set<std::tuple<int,int,int>> shell_voxels = extract_shell(voxels);

        // Output as point cloud
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> shell_points;
        for (const auto& idx : shell_voxels) {
            Eigen::Vector3f center = voxel_center(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx), res);
            shell_points.emplace_back(center.x(), center.y(), center.z());
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_shell(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_shell->points = shell_points;
        cloud_shell->width = static_cast<uint32_t>(shell_points.size());
        cloud_shell->height = 1;
        cloud_shell->is_dense = true;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_shell, output);
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