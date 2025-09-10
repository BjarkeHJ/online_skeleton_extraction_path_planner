#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <set>
#include <tuple>
#include <cmath>
#include <string>

class StaticTSDFPostprocessNode : public rclcpp::Node {
public:
    StaticTSDFPostprocessNode()
    : Node("static_tsdf_postprocess_node"),
      last_point_count_(0)
    {
        this->declare_parameter<std::string>("static_input_topic", "osep/tsdf/static_pointcloud");
        this->declare_parameter<std::string>("output_topic", "osep/tsdf/skeleton");
        this->declare_parameter<double>("voxel_size", 1.0);

        static_input_topic_ = this->get_parameter("static_input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        voxel_size_ = this->get_parameter("voxel_size").as_double();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            static_input_topic_, 1,
            std::bind(&StaticTSDFPostprocessNode::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 1);
    }

private:
    // Quantize point to voxel index
    std::tuple<int,int,int> quantize(float x, float y, float z, float res) {
        return std::make_tuple(
            static_cast<int>(std::floor(x / res)),
            static_cast<int>(std::floor(y / res)),
            static_cast<int>(std::floor(z / res)));
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        size_t current_count = msg->width * msg->height;
        if (current_count <= last_point_count_) {
            if (!last_msg_.data.empty()) {
                pub_->publish(last_msg_);
            }
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Processing point cloud with %zu points", current_count);
        last_point_count_ = current_count;

        // Convert to PCL (no intensity)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Input cloud is empty.");
            return;
        }

        // 1. Build supervoxel population map (with double voxel size)
        double super_voxel_size = 4.0 * voxel_size_;
        std::map<std::tuple<int,int,int>, int> supervoxel_counts;
        for (const auto& pt : cloud->points) {
            auto v = quantize(pt.x, pt.y, pt.z, super_voxel_size);
            supervoxel_counts[v]++;
        }

        // 2. Compute weighted centroid
        Eigen::Vector3f centroid(0, 0, 0);
        float total_weight = 0.0f;
        for (const auto& pt : cloud->points) {
            auto v = quantize(pt.x, pt.y, pt.z, super_voxel_size);
            float weight = 1.0f / static_cast<float>(supervoxel_counts[v]);
            centroid += weight * Eigen::Vector3f(pt.x, pt.y, pt.z);
            total_weight += weight;
        }
        if (total_weight > 0.0f) {
            centroid /= total_weight;
        } else {
            centroid.setZero();
        }

        // 3. Compute distances from centroid and store with index
        std::vector<std::pair<size_t, float>> idx_dist;
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            Eigen::Vector3f vec = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) - centroid;
            idx_dist.emplace_back(i, vec.norm());
        }

        // 4. Sort by distance descending
        std::sort(idx_dist.begin(), idx_dist.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });

        // 5. Select up to 20 points with unique directions and min distance from centroid
        std::vector<size_t> selected_indices;
        std::vector<Eigen::Vector3f> selected_dirs;
        const float dot_threshold = 0.8f; // adjust for angular separation
        const float min_dist = 10.0f * voxel_size_; // minimum distance from centroid

        for (const auto& [idx, dist] : idx_dist) {
            if (dist < min_dist) continue; // skip points too close to centroid
            Eigen::Vector3f dir = Eigen::Vector3f(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z) - centroid;
            if (dir.norm() == 0) continue;
            dir.normalize();
            bool too_close = false;
            for (const auto& sel_dir : selected_dirs) {
                if (dir.dot(sel_dir) > dot_threshold) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                selected_indices.push_back(idx);
                selected_dirs.push_back(dir);
                if (selected_indices.size() >= 20) break;
            }
        }

        // 6. Create a point cloud of the selected points and add the centroid
        pcl::PointCloud<pcl::PointXYZ>::Ptr extrem_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t idx : selected_indices) {
            extrem_cloud->points.push_back(cloud->points[idx]);
        }
        // Add the centroid as a point
        pcl::PointXYZ centroid_pt;
        centroid_pt.x = centroid.x();
        centroid_pt.y = centroid.y();
        centroid_pt.z = centroid.z();
        extrem_cloud->points.push_back(centroid_pt);

        extrem_cloud->width = extrem_cloud->points.size();
        extrem_cloud->height = 1;
        extrem_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*extrem_cloud, output);
        output.header = msg->header;
        pub_->publish(output);

        last_msg_ = output;
        last_point_count_ = msg->width * msg->height;
    }

    std::string static_input_topic_;
    std::string output_topic_;
    double voxel_size_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    sensor_msgs::msg::PointCloud2 last_msg_;
    size_t last_point_count_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticTSDFPostprocessNode>());
    rclcpp::shutdown();
    return 0;
}