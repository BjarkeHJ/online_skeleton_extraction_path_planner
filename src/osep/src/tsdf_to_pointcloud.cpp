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

void TsdfToPointCloudNode::morphological_closing_3d(float voxel_res, int kernel_radius)
{
    // 1. Collect all white points (quantized indices)
    std::set<std::tuple<int, int, int>> white_points;
    int min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN, min_z = INT_MAX, max_z = INT_MIN;
    for (const auto& kv : accumulated_points_) {
        if (kv.second.r == 255 && kv.second.g == 255 && kv.second.b == 255) {
            int x = std::get<0>(kv.first);
            int y = std::get<1>(kv.first);
            int z = std::get<2>(kv.first);
            white_points.insert({x, y, z});
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
    }
    if (white_points.empty()) return;

    min_x -= kernel_radius; max_x += kernel_radius;
    min_y -= kernel_radius; max_y += kernel_radius;
    min_z -= kernel_radius; max_z += kernel_radius;
    int size_x = max_x - min_x + 1;
    int size_y = max_y - min_y + 1;
    int size_z = max_z - min_z + 1;

    // 2. Build binary 3D grid
    std::vector<std::vector<std::vector<uint8_t>>> grid(size_x,
        std::vector<std::vector<uint8_t>>(size_y, std::vector<uint8_t>(size_z, 0)));
    for (const auto& p : white_points) {
        int x = std::get<0>(p) - min_x;
        int y = std::get<1>(p) - min_y;
        int z = std::get<2>(p) - min_z;
        grid[x][y][z] = 1;
    }

    // 3. Dilation
    std::vector<std::vector<std::vector<uint8_t>>> dilated = grid;
    for (int x = 0; x < size_x; ++x) {
        for (int y = 0; y < size_y; ++y) {
            for (int z = 0; z < size_z; ++z) {
                if (grid[x][y][z]) {
                    for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                        for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                            for (int dz = -kernel_radius; dz <= kernel_radius; ++dz) {
                                int nx = x + dx, ny = y + dy, nz = z + dz;
                                if (nx >= 0 && nx < size_x && ny >= 0 && ny < size_y && nz >= 0 && nz < size_z) {
                                    dilated[nx][ny][nz] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Erosion
    std::vector<std::vector<std::vector<uint8_t>>> closed(size_x,
        std::vector<std::vector<uint8_t>>(size_y, std::vector<uint8_t>(size_z, 1)));
    for (int x = 0; x < size_x; ++x) {
        for (int y = 0; y < size_y; ++y) {
            for (int z = 0; z < size_z; ++z) {
                bool erode = false;
                for (int dx = -kernel_radius; dx <= kernel_radius && !erode; ++dx) {
                    for (int dy = -kernel_radius; dy <= kernel_radius && !erode; ++dy) {
                        for (int dz = -kernel_radius; dz <= kernel_radius && !erode; ++dz) {
                            int nx = x + dx, ny = y + dy, nz = z + dz;
                            if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y || nz < 0 || nz >= size_z || dilated[nx][ny][nz] == 0) {
                                erode = true;
                            }
                        }
                    }
                }
                closed[x][y][z] = erode ? 0 : 1;
            }
        }
    }

    // 5. Add new black points (those that are 1 in closed but not in original)
    size_t black_added = 0;
    for (int x = 0; x < size_x; ++x) {
        for (int y = 0; y < size_y; ++y) {
            for (int z = 0; z < size_z; ++z) {
                if (closed[x][y][z] && !grid[x][y][z]) {
                    int gx = min_x + x;
                    int gy = min_y + y;
                    int gz = min_z + z;
                    auto key = std::make_tuple(gx, gy, gz);
                    if (accumulated_points_.count(key) == 0) {
                        float fx = (gx + 0.5f) * voxel_res;
                        float fy = (gy + 0.5f) * voxel_res;
                        float fz = (gz + 0.5f) * voxel_res;
                        accumulated_points_[key] = ColoredPoint{fx, fy, fz, 0, 0, 0};
                        ++black_added;
                    }
                }
            }
        }
    }
    if (black_added > 0) {
        RCLCPP_INFO(this->get_logger(), "3D Morphological closing: added %zu black points", black_added);
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
  static size_t last_white_point_count_ = 0;
  static sensor_msgs::msg::PointCloud2 last_static_msg;
  static int callback_counter = 0;

  bool has_points = false;
  for (const auto& block : msg->blocks) {
      if (!block.centers.empty()) {
          has_points = true;
          break;
      }
  }

  if (!has_points) {
      // Publish at least every 2nd callback if we have a static cloud
      if (last_static_msg.data.size() > 0) {
          callback_counter++;
          if (callback_counter >= 2) {
              static_pub_->publish(last_static_msg);
              callback_counter = 0;
          }
      }
      return;
  }

  // If we get here, the message has points!
  std::unordered_map<std::tuple<int, int, int>, ColoredPoint> current_points;

  // 1. Create and publish colored pointcloud
  auto cloud_msg = create_colored_pointcloud(msg, current_points, voxel_size_);
  pub_->publish(cloud_msg);

  // 2. Update static accumulation
  update_static_accumulation(current_points);

  // 3. Only run morph and publish new static cloud if white points increased
  size_t current_white_points = 0;
  for (const auto& kv : accumulated_points_) {
      if (kv.second.r == 255 && kv.second.g == 255 && kv.second.b == 255) {
          ++current_white_points;
      }
  }

  if (current_white_points > last_white_point_count_) {
      morphological_closing_3d(voxel_size_, std::floor(cavity_fill_diameter_ / (2 * voxel_size_)));
      last_white_point_count_ = current_white_points;
      last_static_msg = create_static_pointcloud(msg->header);
  }

  // Always publish static cloud if we have points
  if (last_static_msg.data.size() > 0) {
      static_pub_->publish(last_static_msg);
  }
  callback_counter = 0; // reset counter since we published due to new data
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TsdfToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}