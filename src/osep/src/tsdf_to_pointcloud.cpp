#include "tsdf_to_pointcloud.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nvblox_msgs/msg/voxel_block.hpp>
#include <nvblox_msgs/msg/index3_d.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <cmath>
#include <cstring>

namespace {
inline std::tuple<int, int, int> quantize(float x, float y, float z, float res = 0.01f) {
  return std::make_tuple(
    static_cast<int>(std::round(x / res)),
    static_cast<int>(std::round(y / res)),
    static_cast<int>(std::round(z / res)));
}
}

TsdfToPointCloudNode::TsdfToPointCloudNode()
: Node("tsdf_to_pointcloud_node")
{
  this->declare_parameter<std::string>("output_topic", "osep/tsdf_pointcloud");
  this->declare_parameter<std::string>("static_output_topic", "osep/static_tsdf_pointcloud");
  this->declare_parameter<int>("min_observations", 5);
  std::string output_topic = this->get_parameter("output_topic").as_string();
  std::string static_output_topic = this->get_parameter("static_output_topic").as_string();
  min_observations_ = this->get_parameter("min_observations").as_int();

  sub_ = this->create_subscription<nvblox_msgs::msg::VoxelBlockLayer>(
    "/nvblox_node/tsdf_layer", 10,
    std::bind(&TsdfToPointCloudNode::callback, this, std::placeholders::_1));
  pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 10);
  static_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(static_output_topic, 1);
}

sensor_msgs::msg::PointCloud2 TsdfToPointCloudNode::create_colored_pointcloud(
    const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr& msg,
    std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points,
    float voxel_res)
{
  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header = msg->header;
  cloud_msg.height = 1;
  sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
  modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  size_t total_points = 0;
  for (const auto & block : msg->blocks) {
    total_points += block.centers.size();
  }
  modifier.resize(total_points);

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(cloud_msg, "rgb");

  for (const auto & block : msg->blocks) {
    for (size_t i = 0; i < block.centers.size(); ++i) {
      float x = block.centers[i].x;
      float y = block.centers[i].y;
      float z = block.centers[i].z;

      *iter_x = x;
      *iter_y = y;
      *iter_z = z;

      uint8_t r = 255, g = 255, b = 255;
      if (!block.colors.empty()) {
        const auto & color = block.colors[i];
        r = static_cast<uint8_t>(color.r * 255.0f);
        g = static_cast<uint8_t>(color.g * 255.0f);
        b = static_cast<uint8_t>(color.b * 255.0f);
      }
      uint32_t rgb = (r << 16) | (g << 8) | b;
      float rgb_float;
      std::memcpy(&rgb_float, &rgb, sizeof(float));
      *reinterpret_cast<float*>(&(*iter_rgb)) = rgb_float;

      // For static accumulation
      auto key = quantize(x, y, z, voxel_res);
      current_points[key] = ColoredPoint{x, y, z, r, g, b};

      ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
    }
  }
  cloud_msg.width = total_points;
  cloud_msg.is_dense = false;
  return cloud_msg;
}

void TsdfToPointCloudNode::update_static_accumulation(
    const std::unordered_map<std::tuple<int, int, int>, ColoredPoint>& current_points)
{
  for (const auto & kv : current_points) {
    auto key = kv.first;
    point_seen_count_[key]++;
    // Only add to static if seen enough times
    if (point_seen_count_[key] == min_observations_) {
      accumulated_points_.emplace(key, kv.second);
    }
  }
}

sensor_msgs::msg::PointCloud2 TsdfToPointCloudNode::create_static_pointcloud(const std_msgs::msg::Header& header)
{
  sensor_msgs::msg::PointCloud2 static_msg;
  static_msg.header = header;
  static_msg.height = 1;
  sensor_msgs::PointCloud2Modifier static_modifier(static_msg);
  static_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  static_modifier.resize(accumulated_points_.size());

  sensor_msgs::PointCloud2Iterator<float> s_iter_x(static_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> s_iter_y(static_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> s_iter_z(static_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> s_iter_rgb(static_msg, "rgb");

  for (const auto & kv : accumulated_points_) {
    const auto & pt = kv.second;
    *s_iter_x = pt.x;
    *s_iter_y = pt.y;
    *s_iter_z = pt.z;
    // Always set white color
    uint32_t rgb = (255 << 16) | (255 << 8) | 255;
    float rgb_float;
    std::memcpy(&rgb_float, &rgb, sizeof(float));
    *reinterpret_cast<float*>(&(*s_iter_rgb)) = rgb_float;
    ++s_iter_x; ++s_iter_y; ++s_iter_z; ++s_iter_rgb;
  }
  static_msg.width = accumulated_points_.size();
  static_msg.is_dense = false;
  return static_msg;
}

void TsdfToPointCloudNode::callback(const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr msg)
{
  float voxel_res = 0.01f; // 1cm grid for deduplication
  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> current_points;

  // 1. Create and publish colored pointcloud
  auto cloud_msg = create_colored_pointcloud(msg, current_points, voxel_res);
  pub_->publish(cloud_msg);

  // 2. Update static accumulation
  update_static_accumulation(current_points);

  // 3. Create and publish static (white) pointcloud
  auto static_msg = create_static_pointcloud(msg->header);
  static_pub_->publish(static_msg);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TsdfToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}