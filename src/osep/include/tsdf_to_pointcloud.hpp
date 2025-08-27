#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nvblox_msgs/msg/voxel_block_layer.hpp>

class TsdfToPointCloudNode : public rclcpp::Node
{
public:
  TsdfToPointCloudNode();

private:
  void callback(const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr msg);

  rclcpp::Subscription<nvblox_msgs::msg::VoxelBlockLayer>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};