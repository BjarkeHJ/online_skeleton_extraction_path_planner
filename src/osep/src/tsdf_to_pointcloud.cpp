#include "tsdf_to_pointcloud.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nvblox_msgs/msg/voxel_block.hpp>
#include <nvblox_msgs/msg/index3_d.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <cmath>
#include <cstring>
#include <queue>
#include <set>

namespace {
inline std::tuple<int, int, int> quantize(float x, float y, float z, float res = 1.0f) {
  return std::make_tuple(
    static_cast<int>(std::floor(x / res)),
    static_cast<int>(std::floor(y / res)),
    static_cast<int>(std::floor(z / res)));
}
}

TsdfToPointCloudNode::TsdfToPointCloudNode()
: Node("tsdf_to_pointcloud_node")
{
  this->declare_parameter<std::string>("output_topic", "osep/tsdf/pointcloud");
  this->declare_parameter<std::string>("static_output_topic", "/osep/tsdf/static_pointcloud");
  this->declare_parameter<float>("cavity_fill_diameter", 5.0);
  this->declare_parameter<float>("voxel_size", 1.0);
  std::string output_topic = this->get_parameter("output_topic").as_string();
  std::string static_output_topic = this->get_parameter("static_output_topic").as_string();
  cavity_fill_diameter_ = this->get_parameter("cavity_fill_diameter").as_double();
  voxel_size_ = this->get_parameter("voxel_size").as_double();

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

      uint8_t r = 255, g = 255, b = 255; // Always white for original points
      // if (!block.colors.empty()) {
      //   const auto & color = block.colors[i];
      //   r = static_cast<uint8_t>(color.r * 255.0f);
      //   g = static_cast<uint8_t>(color.g * 255.0f);
      //   b = static_cast<uint8_t>(color.b * 255.0f);
      // }
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
    auto it = accumulated_points_.find(kv.first);
    if (it != accumulated_points_.end() && it->second.r == 0 && it->second.g == 0 && it->second.b == 0) {
      it->second = kv.second;
      int z = std::get<2>(kv.first);
      white_points_count_[z]++;
    } else if (accumulated_points_.emplace(kv.first, kv.second).second) {
      int z = std::get<2>(kv.first);
      white_points_count_[z]++;
    }
  }
}

void TsdfToPointCloudNode::morphological_closing_xy(float voxel_res, int kernel_radius)
{
  // 1. Group white points by z-index (quantized indices)
  std::unordered_map<int, std::set<std::pair<int, int>>> white_points;
  for (const auto& kv : accumulated_points_) {
    if (kv.second.r == 255 && kv.second.g == 255 && kv.second.b == 255) {
      int x = std::get<0>(kv.first); // quantized
      int y = std::get<1>(kv.first); // quantized
      int z = std::get<2>(kv.first); // quantized
      white_points[z].insert({x, y});
    }
  }

  for (const auto& [z, points] : white_points) {
    if (white_points_count_[z] == 0) continue; // Skip if no new white points
    white_points_count_[z] = 0;
    // Find bounds
    int min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
    for (const auto& p : points) {
      min_x = std::min(min_x, p.first);
      max_x = std::max(max_x, p.first);
      min_y = std::min(min_y, p.second);
      max_y = std::max(max_y, p.second);
    }
    min_x -= kernel_radius; max_x += kernel_radius;
    min_y -= kernel_radius; max_y += kernel_radius;
    int size_x = max_x - min_x + 1;
    int size_y = max_y - min_y + 1;

    // Build binary grid
    std::vector<std::vector<uint8_t>> grid(size_x, std::vector<uint8_t>(size_y, 0));
    for (const auto& p : points) {
      grid[p.first - min_x][p.second - min_y] = 1;
    }

    // Dilation
    std::vector<std::vector<uint8_t>> dilated = grid;
    for (int i = 0; i < size_x; ++i) {
      for (int j = 0; j < size_y; ++j) {
        if (grid[i][j]) {
          for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
              int ni = i + dx, nj = j + dy;
              if (ni >= 0 && ni < size_x && nj >= 0 && nj < size_y) {
                dilated[ni][nj] = 1;
              }
            }
          }
        }
      }
    }

    // Erosion
    std::vector<std::vector<uint8_t>> closed(size_x, std::vector<uint8_t>(size_y, 1));
    for (int i = 0; i < size_x; ++i) {
      for (int j = 0; j < size_y; ++j) {
        bool erode = false;
        for (int dx = -kernel_radius; dx <= kernel_radius && !erode; ++dx) {
          for (int dy = -kernel_radius; dy <= kernel_radius && !erode; ++dy) {
            int ni = i + dx, nj = j + dy;
            if (ni < 0 || ni >= size_x || nj < 0 || nj >= size_y || dilated[ni][nj] == 0) {
              erode = true;
            }
          }
        }
        closed[i][j] = erode ? 0 : 1;
      }
    }

    // Add new black points (those that are 1 in closed but not in original)
    size_t black_added = 0;
    for (int i = 0; i < size_x; ++i) {
      for (int j = 0; j < size_y; ++j) {
        if (closed[i][j] && !grid[i][j]) {
          int x = min_x + i; // quantized
          int y = min_y + j; // quantized
          auto key = std::make_tuple(x, y, z);
          if (accumulated_points_.count(key) == 0) {
            float fx = (x + 0.5f) * voxel_res;
            float fy = (y + 0.5f) * voxel_res;
            float fz = (z + 0.5f) * voxel_res;
            accumulated_points_[key] = ColoredPoint{fx, fy, fz, 0, 0, 0};
            ++black_added;
          }
        }
      }
    }
    if (black_added > 0) {
      RCLCPP_INFO(this->get_logger(), "Morphological closing (z=%d): added %zu black points", z, black_added);
    }
  }
}


sensor_msgs::msg::PointCloud2 TsdfToPointCloudNode::create_static_pointcloud(const std_msgs::msg::Header& header)
{
  sensor_msgs::msg::PointCloud2 static_msg;
  static_msg.header = header;
  static_msg.height = 1;
  sensor_msgs::PointCloud2Modifier static_modifier(static_msg);
  static_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb"); // Add "rgb"
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

    uint32_t rgb = (pt.r << 16) | (pt.g << 8) | pt.b;
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
  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> current_points;

  // 1. Create and publish colored pointcloud
  auto cloud_msg = create_colored_pointcloud(msg, current_points, voxel_size_);
  pub_->publish(cloud_msg);

  // 2. Update static accumulation
  update_static_accumulation(current_points);

  morphological_closing_xy(voxel_size_, std::floor(cavity_fill_diameter_ / (2 * voxel_size_)));

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