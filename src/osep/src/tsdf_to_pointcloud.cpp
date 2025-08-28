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
  // Only add new points, never remove
  for (const auto & kv : current_points) {
    accumulated_points_.emplace(kv.first, kv.second);
  }
}

void TsdfToPointCloudNode::fill_cavities_xy(float voxel_res, float max_radius)
{
  // 1. Build 2D occupancy grid (x, y) -> set of z
  std::unordered_map<std::pair<int, int>, std::set<int>, boost::hash<std::pair<int, int>>> xy_occupied;
  for (const auto& kv : accumulated_points_) {
    auto [qx, qy, qz] = kv.first;
    xy_occupied[{qx, qy}].insert(qz);
  }

  // 2. Find bounds
  int min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
  for (const auto& kv : xy_occupied) {
    min_x = std::min(min_x, kv.first.first);
    max_x = std::max(max_x, kv.first.first);
    min_y = std::min(min_y, kv.first.second);
    max_y = std::max(max_y, kv.first.second);
  }

  // 3. Flood fill empty regions
  std::set<std::pair<int, int>> visited;
  auto is_occupied = [&](int x, int y) {
    return xy_occupied.count({x, y}) > 0;
  };

  std::vector<std::vector<std::pair<int, int>>> cavities;
  for (int x = min_x; x <= max_x; ++x) {
    for (int y = min_y; y <= max_y; ++y) {
      std::pair<int, int> p = {x, y};
      if (is_occupied(x, y) || visited.count(p)) continue;

      // Start BFS
      std::queue<std::pair<int, int>> q;
      std::vector<std::pair<int, int>> region;
      bool touches_border = false;
      q.push(p);
      visited.insert(p);

      while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        region.push_back({cx, cy});
        if (cx == min_x || cx == max_x || cy == min_y || cy == max_y)
          touches_border = true;

        for (auto [dx, dy] : std::vector<std::pair<int, int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
          int nx = cx + dx, ny = cy + dy;
          std::pair<int, int> np = {nx, ny};
          if (nx < min_x || nx > max_x || ny < min_y || ny > max_y) continue;
          if (is_occupied(nx, ny) || visited.count(np)) continue;
          visited.insert(np);
          q.push(np);
        }
      }
      if (!touches_border) cavities.push_back(region);
    }
  }

  // 4. Fill cavities if within radius
  size_t points_added = 0; // Track how many points we add
  for (const auto& region : cavities) {
    // Compute centroid and max distance
    float cx = 0, cy = 0;
    for (const auto& p : region) { cx += p.first; cy += p.second; }
    cx /= region.size(); cy /= region.size();
    float max_dist = 0;
    for (const auto& p : region) {
      float dist = std::hypot(p.first - cx, p.second - cy) * voxel_res;
      max_dist = std::max(max_dist, dist);
    }
    if (max_dist > max_radius) continue;

    // Fill: for each (x, y), add a point at the mean z of neighbors
    for (const auto& p : region) {
      std::vector<int> neighbor_z;
      for (auto [dx, dy] : std::vector<std::pair<int, int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
        auto it = xy_occupied.find({p.first + dx, p.second + dy});
        if (it != xy_occupied.end()) {
          neighbor_z.insert(neighbor_z.end(), it->second.begin(), it->second.end());
        }
      }
      if (neighbor_z.empty()) continue;
      float mean_z = std::accumulate(neighbor_z.begin(), neighbor_z.end(), 0.0f) / neighbor_z.size();
      auto key = std::make_tuple(p.first, p.second, static_cast<int>(std::round(mean_z)));
      if (accumulated_points_.count(key) == 0) {
        float fx = p.first * voxel_res;
        float fy = p.second * voxel_res;
        float fz = mean_z * voxel_res;
        accumulated_points_[key] = ColoredPoint{fx, fy, fz, 255, 255, 255};
        ++points_added;
      }
    }
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
  static_modifier.setPointCloud2FieldsByString(1, "xyz"); // Only XYZ
  static_modifier.resize(accumulated_points_.size());

  sensor_msgs::PointCloud2Iterator<float> s_iter_x(static_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> s_iter_y(static_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> s_iter_z(static_msg, "z");

  for (const auto & kv : accumulated_points_) {
    const auto & pt = kv.second;
    *s_iter_x = pt.x;
    *s_iter_y = pt.y;
    *s_iter_z = pt.z;
    ++s_iter_x; ++s_iter_y; ++s_iter_z;
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