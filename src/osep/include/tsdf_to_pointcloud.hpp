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
    // Simple hash combine
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

  rclcpp::Subscription<nvblox_msgs::msg::VoxelBlockLayer>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pub_;

  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> accumulated_points_;
};