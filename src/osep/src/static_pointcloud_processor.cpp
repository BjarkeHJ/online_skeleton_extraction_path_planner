#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <queue>
#include <set>
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
        this->declare_parameter<int>("upsample_N", 3);       // Default upsampling factor

        static_input_topic_ = this->get_parameter("static_input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        voxel_size_ = this->get_parameter("voxel_size").as_double();
        upsample_N_ = this->get_parameter("upsample_N").as_int();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            static_input_topic_, 1,
            std::bind(&StaticPointcloudPostprocessNode::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 1);
    }

private:
    std::string static_input_topic_;
    std::string output_topic_;
    double voxel_size_;
    int upsample_N_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    size_t last_point_count_ = 0;
    sensor_msgs::msg::PointCloud2 last_msg_;

    // Filtering function: remove points not in a group of at least min_group_size per voxel
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
    filter_sparse_groups(const sensor_msgs::msg::PointCloud2 &msg, double voxel_size, int min_group_size)
    {
        std::map<std::tuple<int, int, int>, std::vector<std::array<float, 3>>> voxel_groups;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
        for (size_t i = 0; i < msg.width * msg.height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            int ix = static_cast<int>(std::floor(*iter_x / voxel_size));
            int iy = static_cast<int>(std::floor(*iter_y / voxel_size));
            int iz = static_cast<int>(std::floor(*iter_z / voxel_size));
            voxel_groups[{ix, iy, iz}].push_back(std::array<float, 3>{*iter_x, *iter_y, *iter_z});
        }

        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> filtered_points;
        for (const auto& kv : voxel_groups) {
            if (kv.second.size() >= min_group_size) {
                for (const auto& arr : kv.second) {
                    filtered_points.emplace_back(arr[0], arr[1], arr[2]);
                }
            }
        }
        return filtered_points;
    }

    // Only keep voxels on the true outer shell (not internal cavities)
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
    keep_outer_surface_voxels(const std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>& voxelCenters, double voxel_size)
    {
        // Build set of occupied voxel indices
        std::set<std::tuple<int, int, int>> occupied_voxels;
        int min_x = INT_MAX, max_x = INT_MIN;
        int min_y = INT_MAX, max_y = INT_MIN;
        int min_z = INT_MAX, max_z = INT_MIN;
        for (const auto& c : voxelCenters) {
            int ix = static_cast<int>(std::round(c.x / voxel_size));
            int iy = static_cast<int>(std::round(c.y / voxel_size));
            int iz = static_cast<int>(std::round(c.z / voxel_size));
            occupied_voxels.insert({ix, iy, iz});
            min_x = std::min(min_x, ix); max_x = std::max(max_x, ix);
            min_y = std::min(min_y, iy); max_y = std::max(max_y, iy);
            min_z = std::min(min_z, iz); max_z = std::max(max_z, iz);
        }

        // Flood fill to find outside air
        std::set<std::tuple<int, int, int>> outside_air;
        std::queue<std::tuple<int, int, int>> q;
        for (int ix = min_x; ix <= max_x; ++ix)
        for (int iy = min_y; iy <= max_y; ++iy)
        for (int iz = min_z; iz <= max_z; ++iz) {
            bool on_boundary = (ix == min_x || ix == max_x ||
                                iy == min_y || iy == max_y ||
                                iz == min_z || iz == max_z);
            auto idx = std::make_tuple(ix, iy, iz);
            if (on_boundary && occupied_voxels.count(idx) == 0) {
                q.push(idx);
                outside_air.insert(idx);
            }
        }
        const int dx[6] = {1, -1, 0, 0, 0, 0};
        const int dy[6] = {0, 0, 1, -1, 0, 0};
        const int dz[6] = {0, 0, 0, 0, 1, -1};
        while (!q.empty()) {
            auto idx = q.front(); q.pop();
            int ix = std::get<0>(idx), iy = std::get<1>(idx), iz = std::get<2>(idx);
            for (int n = 0; n < 6; ++n) {
                auto nidx = std::make_tuple(ix + dx[n], iy + dy[n], iz + dz[n]);
                if (ix + dx[n] < min_x || ix + dx[n] > max_x ||
                    iy + dy[n] < min_y || iy + dy[n] > max_y ||
                    iz + dz[n] < min_z || iz + dz[n] > max_z)
                    continue;
                if (occupied_voxels.count(nidx) == 0 && outside_air.count(nidx) == 0) {
                    q.push(nidx);
                    outside_air.insert(nidx);
                }
            }
        }

        // Keep only voxels that have a neighbor in outside_air
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> outer_surface_voxels;
        for (const auto& c : voxelCenters) {
            int ix = static_cast<int>(std::round(c.x / voxel_size));
            int iy = static_cast<int>(std::round(c.y / voxel_size));
            int iz = static_cast<int>(std::round(c.z / voxel_size));
            for (int n = 0; n < 6; ++n) {
                auto nidx = std::make_tuple(ix + dx[n], iy + dy[n], iz + dz[n]);
                if (outside_air.count(nidx)) {
                    outer_surface_voxels.push_back(c);
                    break;
                }
            }
        }
        return outer_surface_voxels;
    }

    // Upsampling function
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
    upsample_voxels(const std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>& voxelCenters, double voxel_size, int N)
    {
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> upsampled_points;
        double sub_step = voxel_size / N;
        for (const auto& center : voxelCenters) {
            double start_x = center.x - voxel_size / 2.0 + sub_step / 2.0;
            double start_y = center.y - voxel_size / 2.0 + sub_step / 2.0;
            double start_z = center.z - voxel_size / 2.0 + sub_step / 2.0;
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
        return upsampled_points;
    }

    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
    smooth_points(const std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>& input_points, double neighbor_radius)
    {
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> smoothed_points;
        for (const auto& pt : input_points) {
            Eigen::Vector3f sum(pt.x, pt.y, pt.z);
            int count = 1;
            for (const auto& neighbor : input_points) {
                if (&pt == &neighbor) continue;
                double dx = pt.x - neighbor.x;
                double dy = pt.y - neighbor.y;
                double dz = pt.z - neighbor.z;
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < neighbor_radius * 1.01) {
                    sum += Eigen::Vector3f(neighbor.x, neighbor.y, neighbor.z);
                    count++;
                }
            }
            pcl::PointXYZ smooth_pt;
            smooth_pt.x = sum.x() / count;
            smooth_pt.y = sum.y() / count;
            smooth_pt.z = sum.z() / count;
            smoothed_points.push_back(smooth_pt);
        }
        return smoothed_points;
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        size_t current_count = msg->width * msg->height;
        if (current_count <= last_point_count_) {
            if (!last_msg_.data.empty())
                pub_->publish(last_msg_);
            return;
        }
        last_point_count_ = current_count;
        auto filtered_points = filter_sparse_groups(*msg, voxel_size_*10, 30);

        // Convert ROS msg to PCL cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->points = filtered_points;
        cloud->width = static_cast<uint32_t>(filtered_points.size());
        cloud->height = 1;
        cloud->is_dense = true;

        // Create octree at voxel_size_ resolution
        pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ> octree(voxel_size_);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        // Extract voxel centers
        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxelCenters;
        octree.getOccupiedVoxelCenters(voxelCenters);

        // Upsample using the function
        auto upsampled_points = upsample_voxels(voxelCenters, voxel_size_, upsample_N_);
        // Only keep the outer surface points
        auto outer_surface_upsampled = keep_outer_surface_voxels(upsampled_points, voxel_size_/upsample_N_);// Work from here!!

        double neighbor_radius = voxel_size_ / upsample_N_;
        auto smoothed_points = smooth_points(outer_surface_upsampled, neighbor_radius);


        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_upsampled->points = smoothed_points;
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