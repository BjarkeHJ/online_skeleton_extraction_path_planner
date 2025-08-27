#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <Eigen/Core>

class StaticPointcloudPostprocessNode : public rclcpp::Node
{
public:
    StaticPointcloudPostprocessNode()
    : Node("static_pointcloud_postprocess_node")
    {
        this->declare_parameter<std::string>("static_input_topic", "osep/tsdf/static_pointcloud");
        this->declare_parameter<std::string>("output_topic", "osep/tsdf/upsampled_static_pointcloud");
        this->declare_parameter<double>("voxel_size", 1.0);  // 1.0 m by default

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
        size_t current_count = msg->width * msg->height;
        if (current_count <= last_point_count_) {
            if (!last_msg_.data.empty())
                pub_->publish(last_msg_);
            return;
        }
        last_point_count_ = current_count;

        // Convert ROS msg to PCL cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Create octree at voxel_size_ resolution
        pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ> octree(voxel_size_);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        // Extract voxel centers
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxelCenters;
        octree.getOccupiedVoxelCenters(voxelCenters);

        int N = 3;
        double sub_step = voxel_size_ / N;
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> upsampled_points;
        for (const auto& center : voxelCenters) {
            double start_x = center.x - voxel_size_ / 2.0 + sub_step / 2.0;
            double start_y = center.y - voxel_size_ / 2.0 + sub_step / 2.0;
            double start_z = center.z - voxel_size_ / 2.0 + sub_step / 2.0;
            for (int ix = 0; ix < N; ++ix) {
                for (int iy = 0; iy < N; ++iy) {
                    for (int iz = 0; iz < N; ++iz) {
                        pcl::PointXYZ pt;
                        pt.x = start_x + ix * sub_step;
                        pt.y = start_y + iy * sub_step;
                        pt.z = start_z + iz * sub_step;
                        upsampled_points.push_back(pt);
                    }
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_upsampled->points = upsampled_points;
        cloud_upsampled->width = static_cast<uint32_t>(cloud_upsampled->points.size());
        cloud_upsampled->height = 1;
        cloud_upsampled->is_dense = true;

        // Convert back to ROS msg
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_upsampled, output);
        output.header = msg->header;

        pub_->publish(output);
        last_msg_ = output;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticPointcloudPostprocessNode>());
    rclcpp::shutdown();
    return 0;
}