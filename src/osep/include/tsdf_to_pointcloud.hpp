#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nvblox_msgs/msg/voxel_block_layer.hpp>
#include <unordered_map>
#include <tuple>
#include <string>

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
  void callback(const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr msg);

  std::unordered_map<std::tuple<int, int, int>, int> point_seen_count_;
  int min_observations_ = 3; // Or make this a parameter

  sensor_msgs::msg::PointCloud2 create_colored_pointcloud(
    const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr& msg,
    std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points,
    float voxel_res);

  void update_static_accumulation(
    const std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points);

  sensor_msgs::msg::PointCloud2 create_static_pointcloud(const std_msgs::msg::Header& header);

  rclcpp::Subscription<nvblox_msgs::msg::VoxelBlockLayer>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pub_;

  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> accumulated_points_;
};