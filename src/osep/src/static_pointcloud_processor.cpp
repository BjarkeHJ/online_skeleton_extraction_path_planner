#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

// Only the hash specialization goes in std
namespace std {
template <>
struct hash<std::tuple<int, int, int>> {
  std::size_t operator()(const std::tuple<int, int, int>& k) const {
    return std::get<0>(k) ^ (std::get<1>(k) << 8) ^ (std::get<2>(k) << 16);
  }
};
}

// Quantize a point to voxel indices
inline std::tuple<int, int, int> quantize(float x, float y, float z, float res) {
    return std::make_tuple(
        static_cast<int>(std::round(x / res)),
        static_cast<int>(std::round(y / res)),
        static_cast<int>(std::round(z / res)));
}

class StaticPointcloudPostprocessNode : public rclcpp::Node
{
public:
    StaticPointcloudPostprocessNode()
    : Node("static_pointcloud_postprocess_node")
    {
        this->declare_parameter<std::string>("static_input_topic", "osep/tsdf/static_pointcloud");
        this->declare_parameter<std::string>("output_topic", "osep/tsdf/upsampled_static_pointcloud");
        this->declare_parameter<double>("voxel_size", 1.0); // 1 m default

        static_input_topic_ = this->get_parameter("static_input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        voxel_size_ = this->get_parameter("voxel_size").as_double();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            static_input_topic_, 1,
            std::bind(&StaticPointcloudPostprocessNode::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 1);
    }

private:
    std::string static_input_topic_;
    std::string output_topic_;
    double voxel_size_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    size_t last_point_count_ = 0;
    sensor_msgs::msg::PointCloud2 last_msg_;

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (msg->width * msg->height == last_point_count_) {
            // No change, publish previous result
            pub_->publish(last_msg_);
            return;
        }

        // 1. Remove sparse points (group by original voxel_size)
        std::unordered_map<std::tuple<int, int, int>, std::vector<std::array<float, 3>>> coarse_voxels;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            auto key = quantize(*iter_x, *iter_y, *iter_z, voxel_size_*10);
            coarse_voxels[key].push_back({*iter_x, *iter_y, *iter_z});
        }
        std::vector<std::array<float, 3>> filtered_points;
        for (const auto& kv : coarse_voxels) {
            if (kv.second.size() >= 30) {
                filtered_points.insert(filtered_points.end(), kv.second.begin(), kv.second.end());
            }
        }

        // 2. Upsample to fine grid (0.01 * voxel_size)
        double fine_res = voxel_size_ * 0.01;
        std::unordered_map<std::tuple<int, int, int>, std::array<float, 3>> fine_points;
        for (const auto& pt : filtered_points) {
            auto key = quantize(pt[0], pt[1], pt[2], fine_res);
            fine_points[key] = pt;
        }

        // 3. (Optional) Smooth the surface (simple mean filter)
        std::unordered_map<std::tuple<int, int, int>, std::array<float, 3>> smoothed_points;
        std::array<std::tuple<int, int, int>, 6> neighbors = {{
            {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}
        }};
        for (const auto& kv : fine_points) {
            std::array<float, 3> sum = kv.second;
            int count = 1;
            for (const auto& n : neighbors) {
                auto neighbor_key = std::make_tuple(
                    std::get<0>(kv.first) + std::get<0>(n),
                    std::get<1>(kv.first) + std::get<1>(n),
                    std::get<2>(kv.first) + std::get<2>(n)
                );
                auto it = fine_points.find(neighbor_key);
                if (it != fine_points.end()) {
                    sum[0] += it->second[0];
                    sum[1] += it->second[1];
                    sum[2] += it->second[2];
                    ++count;
                }
            }
            smoothed_points[kv.first] = {sum[0]/count, sum[1]/count, sum[2]/count};
        }

        // 4. Keep only surface points (at least one neighbor missing)
        std::vector<std::array<float, 3>> surface_points;
        for (const auto& kv : smoothed_points) {
            bool is_surface = false;
            for (const auto& n : neighbors) {
                auto neighbor_key = std::make_tuple(
                    std::get<0>(kv.first) + std::get<0>(n),
                    std::get<1>(kv.first) + std::get<1>(n),
                    std::get<2>(kv.first) + std::get<2>(n)
                );
                if (smoothed_points.find(neighbor_key) == smoothed_points.end()) {
                    is_surface = true;
                    break;
                }
            }
            if (is_surface) {
                surface_points.push_back(kv.second);
            }
        }

        // 5. Publish surface pointcloud
        sensor_msgs::msg::PointCloud2 out_msg;
        out_msg.header = msg->header;
        out_msg.height = 1;
        out_msg.width = surface_points.size();
        out_msg.is_dense = false;
        sensor_msgs::PointCloud2Modifier modifier(out_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(surface_points.size());

        sensor_msgs::PointCloud2Iterator<float> ox(out_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> oy(out_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> oz(out_msg, "z");
        for (const auto& pt : surface_points) {
            *ox = pt[0]; *oy = pt[1]; *oz = pt[2];
            ++ox; ++oy; ++oz;
        }

        pub_->publish(out_msg);
        last_point_count_ = msg->width * msg->height;
        last_msg_ = out_msg;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticPointcloudPostprocessNode>());
    rclcpp::shutdown();
    return 0;
}