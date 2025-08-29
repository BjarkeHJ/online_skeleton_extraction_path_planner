#include "static_pointcloud_processor.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <cmath>

StaticPointcloudPostprocessNode::StaticPointcloudPostprocessNode()
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

std::tuple<int, int, int> StaticPointcloudPostprocessNode::quantize(float x, float y, float z, float res) {
    return std::make_tuple(
        static_cast<int>(std::floor(x / res)),
        static_cast<int>(std::floor(y / res)),
        static_cast<int>(std::floor(z / res)));
}

Eigen::Vector3f StaticPointcloudPostprocessNode::voxel_center(int ix, int iy, int iz, float res) {
    return Eigen::Vector3f(
        (ix + 0.5f) * res,
        (iy + 0.5f) * res,
        (iz + 0.5f) * res
    );
}

std::set<std::tuple<int,int,int>> StaticPointcloudPostprocessNode::upsample_voxels_by_2(
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

std::set<std::tuple<int,int,int>> StaticPointcloudPostprocessNode::full_erosion(
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

std::set<std::tuple<int,int,int>> StaticPointcloudPostprocessNode::full_dilation(
    const std::set<std::tuple<int,int,int>>& voxels)
{
    std::set<std::tuple<int,int,int>> dilated;
    for (const auto& v : voxels) {
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz)
                    if (!(dx == 0 && dy == 0 && dz == 0))
                        dilated.insert({std::get<0>(v)+dx, std::get<1>(v)+dy, std::get<2>(v)+dz});
        dilated.insert(v);
    }
    return dilated;
}

std::set<std::tuple<int,int,int>> StaticPointcloudPostprocessNode::selective_dilation(
    const std::set<std::tuple<int,int,int>>& eroded)
{
    std::set<std::tuple<int,int,int>> dilated = eroded;
    static const int dxs[6] = {1, -1, 0, 0, 0, 0};
    static const int dys[6] = {0, 0, 1, -1, 0, 0};
    static const int dzs[6] = {0, 0, 0, 0, 1, -1};

    for (const auto& v : eroded) {
        for (int i = 0; i < 6; ++i) {
            std::tuple<int,int,int> neighbor{
                std::get<0>(v)+dxs[i],
                std::get<1>(v)+dys[i],
                std::get<2>(v)+dzs[i]
            };
            if (!eroded.count(neighbor)) {
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

std::set<std::tuple<int,int,int>> StaticPointcloudPostprocessNode::extract_shell(
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

void StaticPointcloudPostprocessNode::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    size_t current_count = msg->width * msg->height;
    if (current_count <= last_point_count_) {
        if (!last_msg_.data.empty()) {
            pub_->publish(last_msg_);
        }
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Processing point cloud with %zu points", current_count);
    last_point_count_ = current_count;

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
        if (n <= 1) {
            voxels = selective_dilation(voxels);
        } else {
            voxels = full_erosion(voxels, 26);
            voxels = full_dilation(voxels);
        }
        if (n == N-1) {
            voxels = full_erosion(voxels, 26);
            voxels = selective_dilation(voxels);
        }
    }

    std::set<std::tuple<int,int,int>> shell_voxels = extract_shell(voxels);

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

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticPointcloudPostprocessNode>());
    rclcpp::shutdown();
    return 0;
}