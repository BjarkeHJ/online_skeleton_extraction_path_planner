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

// Helper: 2D convex hull (Graham scan, for XY projection)
std::vector<std::array<float, 3>> convex_hull_xy(const std::vector<std::array<float, 3>>& points) {
    if (points.size() < 3) return points;
    std::vector<std::array<float, 3>> pts = points;
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });
    std::vector<std::array<float, 3>> hull;
    // Lower hull
    for (const auto& pt : pts) {
        while (hull.size() >= 2) {
            auto& q = hull[hull.size()-2];
            auto& r = hull[hull.size()-1];
            if ((r[0]-q[0])*(pt[1]-q[1]) - (r[1]-q[1])*(pt[0]-q[0]) <= 0)
                hull.pop_back();
            else break;
        }
        hull.push_back(pt);
    }
    // Upper hull
    size_t t = hull.size() + 1;
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        const auto& pt = *it;
        while (hull.size() >= t) {
            auto& q = hull[hull.size()-2];
            auto& r = hull[hull.size()-1];
            if ((r[0]-q[0])*(pt[1]-q[1]) - (r[1]-q[1])*(pt[0]-q[0]) <= 0)
                hull.pop_back();
            else break;
        }
        hull.push_back(pt);
    }
    hull.pop_back();
    return hull;
}

// Linear interpolation between two points
std::vector<std::array<float, 3>> interpolate_between(
    const std::array<float, 3>& a, const std::array<float, 3>& b, int num_points)
{
    std::vector<std::array<float, 3>> result;
    for (int i = 1; i < num_points; ++i) {
        float t = float(i) / num_points;
        result.push_back({
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2])
        });
    }
    return result;
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

        // 1. Group points by voxel
        std::unordered_map<std::tuple<int, int, int>, std::vector<std::array<float, 3>>> voxel_groups;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            auto key = quantize(*iter_x, *iter_y, *iter_z, voxel_size_*10);
            voxel_groups[key].push_back({*iter_x, *iter_y, *iter_z});
        }

        // 2. Remove groups with less than 30 points
        std::vector<std::array<float, 3>> filtered_points;
        for (const auto& kv : voxel_groups) {
            if (kv.second.size() >= 30) {
                filtered_points.insert(filtered_points.end(), kv.second.begin(), kv.second.end());
            }
        }

        // 3. Curve upsampling along surface boundary
        std::vector<std::array<float, 3>> upsampled_points;
        for (const auto& kv : voxel_groups) {
            const auto& group = kv.second;
            if (group.size() < 3) continue;
            auto hull = convex_hull_xy(group);
            // Add hull points
            upsampled_points.insert(upsampled_points.end(), hull.begin(), hull.end());
            // Interpolate between hull points
            for (size_t i = 0; i < hull.size(); ++i) {
                const auto& a = hull[i];
                const auto& b = hull[(i+1)%hull.size()];
                auto interp = interpolate_between(a, b, 3); // 2 extra points between each hull edge
                upsampled_points.insert(upsampled_points.end(), interp.begin(), interp.end());
            }
        }

        // 4. Publish upsampled pointcloud
        sensor_msgs::msg::PointCloud2 out_msg;
        out_msg.header = msg->header;
        out_msg.height = 1;
        out_msg.width = upsampled_points.size();
        out_msg.is_dense = false;
        sensor_msgs::PointCloud2Modifier modifier(out_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(upsampled_points.size());

        sensor_msgs::PointCloud2Iterator<float> ox(out_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> oy(out_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> oz(out_msg, "z");
        for (const auto& pt : upsampled_points) {
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