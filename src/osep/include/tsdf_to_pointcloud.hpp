#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nvblox_msgs/msg/voxel_block_layer.hpp>
#include <unordered_map>
#include <tuple>
#include <string>
#include <boost/functional/hash.hpp>

struct ColoredPoint {
  float x, y, z;
  uint8_t r, g, b;
};

namespace std {
template <>
struct hash<std::tuple<int, int, int>> {
  std::size_t operator()(const std::tuple<int, int, int>& k) const {
    return std::get<0>(k) ^ (std::get<1>(k) << 8) ^ (std::get<2>(k) << 16);
  }
};
}

class TsdfToPointCloudNode : public rclcpp::Node
{
public:
  TsdfToPointCloudNode();

private:
  float cavity_fill_max_radius_;
  float voxel_size_;

  void callback(const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr msg);

  sensor_msgs::msg::PointCloud2 create_colored_pointcloud(
    const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr& msg,
    std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points,
    float voxel_res);

  void update_static_accumulation(
    const std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points);

  sensor_msgs::msg::PointCloud2 create_static_pointcloud(const std_msgs::msg::Header& header);

  void morphological_closing_xy(float voxel_res, int kernel_radius);

  rclcpp::Subscription<nvblox_msgs::msg::VoxelBlockLayer>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pub_;

  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> accumulated_points_;
};