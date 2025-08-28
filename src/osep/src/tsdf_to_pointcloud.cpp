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
  this->declare_parameter<float>("cavity_fill_max_radius", 5.0);
  this->declare_parameter<float>("voxel_size", 1.0);
  std::string output_topic = this->get_parameter("output_topic").as_string();
  std::string static_output_topic = this->get_parameter("static_output_topic").as_string();
  cavity_fill_max_radius_ = this->get_parameter("cavity_fill_max_radius").as_double();
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
  // Only add new points, never remove
  for (const auto & kv : current_points) {
    accumulated_points_.emplace(kv.first, kv.second);
  }
}

void TsdfToPointCloudNode::fill_cavities_xy(float voxel_res, float max_radius)
{
  // 1. Build 2D occupancy grids for each z-slice and determine z-bounds
  std::unordered_map<int, std::set<std::pair<int, int>>> slice_occupied;
  int min_z = INT_MAX, max_z = INT_MIN;

  if (accumulated_points_.empty()) {
      RCLCPP_INFO(this->get_logger(), "No accumulated points to process.");
      return;
  }

  for (const auto& kv : accumulated_points_) {
    auto [qx, qy, qz] = kv.first;
    slice_occupied[qz].insert({qx, qy});
    min_z = std::min(min_z, qz);
    max_z = std::max(max_z, qz);
  }

  size_t points_added = 0;
  size_t total_cavities_found = 0;

  // Iterate over all relevant z-slices
  for (int qz = min_z; qz <= max_z; ++qz) {
    RCLCPP_INFO(this->get_logger(), "Processing z-slice %d.", qz);
    const auto& occupied_xy = slice_occupied.count(qz) ? slice_occupied.at(qz) : std::set<std::pair<int, int>>{};

    // 2. Perform multiple morphological dilation passes to close gaps
    std::set<std::pair<int, int>> dilated_occupied = occupied_xy;
    const int dilation_passes = 2; // Dilate 2 times to close wider gaps
    for (int i = 0; i < dilation_passes; ++i) {
      std::set<std::pair<int, int>> temp_dilated = dilated_occupied;
      for (const auto& p : dilated_occupied) {
        int x = p.first;
        int y = p.second;
        for (int dx = -1; dx <= 1; ++dx) {
          for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            temp_dilated.insert({x + dx, y + dy});
          }
        }
      }
      dilated_occupied = temp_dilated;
    }

    if (dilated_occupied.empty()) {
        continue;
    }

    // 3. Find bounds for this z-slice using the dilated grid
    int min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
    for (const auto& xy : dilated_occupied) {
      min_x = std::min(min_x, xy.first);
      max_x = std::max(max_x, xy.first);
      min_y = std::min(min_y, xy.second);
      max_y = std::max(max_y, xy.second);
    }
    min_x -= 1; max_x += 1; min_y -= 1; max_y += 1;

    // 4. Flood fill empty regions to find cavities
    std::set<std::pair<int, int>> visited;
    
    auto is_occupied = [&](int x, int y) {
      return dilated_occupied.count({x, y}) > 0;
    };

    std::vector<std::vector<std::pair<int, int>>> cavities;
    for (int x = min_x; x <= max_x; ++x) {
      for (int y = min_y; y <= max_y; ++y) {
        std::pair<int, int> p = {x, y};
        if (visited.count(p) || is_occupied(x, y)) {
          continue;
        }

        std::queue<std::pair<int, int>> q;
        std::vector<std::pair<int, int>> region;
        bool touches_object_boundary = false;
        
        q.push(p);
        visited.insert(p);

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();
          region.push_back({cx, cy});

          for (auto [dx, dy] : std::vector<std::pair<int, int>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
            int nx = cx + dx, ny = cy + dy;
            std::pair<int, int> np = {nx, ny};
            if (is_occupied(nx, ny)) {
              touches_object_boundary = true;
            }
            if (nx < min_x || nx > max_x || ny < min_y || ny > max_y || visited.count(np) || is_occupied(nx, ny)) {
              continue;
            }
            visited.insert(np);
            q.push(np);
          }
        }
        if (!touches_object_boundary) {
          cavities.push_back(region);
        }
      }
    }

    if (cavities.size() > 0) {
      RCLCPP_INFO(this->get_logger(), "Found %zu cavities in z-slice %d.", cavities.size(), qz);
    }
    total_cavities_found += cavities.size();

    // 5. Fill cavities in this z-slice if within radius
    for (const auto& region : cavities) {
      float cx = 0, cy = 0;
      for (const auto& p : region) { cx += p.first; cy += p.second; }
      cx /= region.size(); cy /= region.size();
      float sum_dist = 0;
      for (const auto& p : region) {
        float dist = std::hypot(p.first - cx, p.second - cy) * voxel_res;
        sum_dist += dist;
      }
      float avg_dist = sum_dist / region.size();
      RCLCPP_INFO(this->get_logger(), "Cavity average distance (z=%d): %.3f", qz, avg_dist);
      if (avg_dist > max_radius) continue;

      for (const auto& p : region) {
        auto key = std::make_tuple(p.first, p.second, qz);
        if (accumulated_points_.count(key) == 0) {
          float fx = p.first * voxel_res;
          float fy = p.second * voxel_res;
          float fz = qz * voxel_res;
          accumulated_points_[key] = ColoredPoint{fx, fy, fz, 0, 0, 0};
          ++points_added;
        }
      }
    }
  }

  if (total_cavities_found > 0) {
    RCLCPP_INFO(this->get_logger(), "Total cavities found: %zu", total_cavities_found);
  }
  if (points_added > 0) {
    RCLCPP_INFO(this->get_logger(), "Cavity filling: added %zu points", points_added);
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
  fill_cavities_xy(voxel_size_, cavity_fill_max_radius_);

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