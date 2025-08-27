#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nvblox_msgs/msg/voxel_block_layer.hpp>
#include <nvblox_msgs/msg/voxel_block.hpp>
#include <nvblox_msgs/msg/index3_d.hpp>
#include <std_msgs/msg/color_rgba.hpp>

class TsdfToPointCloudNode : public rclcpp::Node
{
public:
  TsdfToPointCloudNode()
  : Node("tsdf_to_pointcloud_node")
  {
    sub_ = this->create_subscription<nvblox_msgs::msg::VoxelBlockLayer>(
      "/nvblox_node/tsdf_layer", 10,
      std::bind(&TsdfToPointCloudNode::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("osep/tsdf_pointcloud", 10);
  }

private:
  void callback(const nvblox_msgs::msg::VoxelBlockLayer::SharedPtr msg)
  {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header = msg->header;
    cloud_msg.height = 1;

    // Define fields: x, y, z, rgb
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
        *iter_x = block.centers[i].x;
        *iter_y = block.centers[i].y;
        *iter_z = block.centers[i].z;

        uint8_t r = 255, g = 255, b = 255;
        if (!block.colors.empty()) {
          const auto & color = block.colors[i];
          r = static_cast<uint8_t>(color.r * 255.0f);
          g = static_cast<uint8_t>(color.g * 255.0f);
          b = static_cast<uint8_t>(color.b * 255.0f);
        }
        // Pack RGB into float (PCL style)
        uint32_t rgb = (r << 16) | (g << 8) | b;
        float rgb_float;
        std::memcpy(&rgb_float, &rgb, sizeof(float));
        *reinterpret_cast<float*>(&(*iter_rgb)) = rgb_float;

        ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
      }
    }
    cloud_msg.width = total_points;
    cloud_msg.is_dense = false;

    pub_->publish(cloud_msg);
  }

  rclcpp::Subscription<nvblox_msgs::msg::VoxelBlockLayer>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TsdfToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}