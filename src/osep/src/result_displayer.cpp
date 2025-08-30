#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_set>
#include <tuple>
#include <cmath>
#include <functional>
#include <chrono>
#include <thread>

namespace std {
template <>
struct hash<std::tuple<int, int, int>> {
    std::size_t operator()(const std::tuple<int, int, int>& k) const {
        return std::get<0>(k) ^ (std::get<1>(k) << 8) ^ (std::get<2>(k) << 16);
    }
};
}

struct PoseData {
    double time;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

class ResultDisplayer : public rclcpp::Node {
public:
    ResultDisplayer()
    : Node("result_displayer_cpp")
    {
        std::string results_dir = "/workspaces/isaac_ros-dev/src/osep/results/";

        this->declare_parameter<std::string>("csv_file", "wind_0_gt.csv");
        this->declare_parameter<std::string>("pcd_file", "wind_0_voxels_0.1.pcd");
        this->declare_parameter<std::string>("frame_id", "odom");
        this->declare_parameter<int>("stride", 1);
        this->declare_parameter<double>("voxel_size", 0.1);
        this->declare_parameter<double>("pyramid_length", 20.0);
        this->declare_parameter<double>("pyramid_width", 15.0);
        this->declare_parameter<double>("pyramid_height", 15.0);
        this->declare_parameter<double>("detection_distance", 20.0);
        this->declare_parameter<double>("playback_speed", 1.0);

        std::string csv_file = results_dir + this->get_parameter("csv_file").as_string();
        std::string pcd_file = results_dir + this->get_parameter("pcd_file").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        int stride = this->get_parameter("stride").as_int();
        double voxel_size = this->get_parameter("voxel_size").as_double();
        double pyramid_length = this->get_parameter("pyramid_length").as_double();
        double pyramid_width = this->get_parameter("pyramid_width").as_double();
        double pyramid_height = this->get_parameter("pyramid_height").as_double();
        double detection_distance = this->get_parameter("detection_distance").as_double();
        double playback_speed = this->get_parameter("playback_speed").as_double();

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("processed_cloud", 1);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("pyramid_marker", 1);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("replayed_pose", 1);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("replayed_path_sofar", 1);

        // Load path from CSV
        std::vector<PoseData> path = load_path_from_csv(csv_file, stride);
        if (path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Path is empty, aborting.");
            return;
        }

        // Load point cloud using PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_xyz) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file %s", pcd_file.c_str());
            return;
        }

        // Convert to PointXYZRGB (default red)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->points.resize(cloud_xyz->points.size());
        cloud->width = cloud_xyz->width;
        cloud->height = cloud_xyz->height;
        cloud->is_dense = cloud_xyz->is_dense;
        for (size_t i = 0; i < cloud_xyz->points.size(); ++i) {
            cloud->points[i].x = cloud_xyz->points[i].x;
            cloud->points[i].y = cloud_xyz->points[i].y;
            cloud->points[i].z = cloud_xyz->points[i].z;
            cloud->points[i].r = 255;
            cloud->points[i].g = 0;
            cloud->points[i].b = 0;
        }
        std::vector<bool> point_seen(cloud->points.size(), false);

        nav_msgs::msg::Path path_so_far;
        path_so_far.header.frame_id = frame_id_;

        // Build a set of occupied voxels for fast lookup (for occlusion)
        std::unordered_set<std::tuple<int,int,int>> points_voxels;
        for (const auto& pt : cloud->points) {
            points_voxels.emplace(
                static_cast<int>(std::floor(pt.x / voxel_size)),
                static_cast<int>(std::floor(pt.y / voxel_size)),
                static_cast<int>(std::floor(pt.z / voxel_size))
            );
        }

        // Real-time playback loop
        auto wall_start = std::chrono::steady_clock::now();
        double csv_start = path.front().time;
        for (size_t i = 0; i < path.size(); ++i) {
            const auto& pose = path[i];
            Eigen::Vector3d drone_pos = pose.position;
            double drone_yaw = get_yaw_from_quaternion(pose.orientation);

            // Set all points to red by default, green if seen
            for (size_t j = 0; j < cloud->points.size(); ++j) {
                if (point_seen[j]) {
                    cloud->points[j].r = 0;
                    cloud->points[j].g = 255;
                    cloud->points[j].b = 0;
                } else {
                    cloud->points[j].r = 255;
                    cloud->points[j].g = 0;
                    cloud->points[j].b = 0;
                }
            }

            // Mark points as green if visible (and accumulate)
            for (size_t j = 0; j < cloud->points.size(); ++j) {
                if (point_seen[j]) continue;
                Eigen::Vector3d pt(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z);
                double dist = (pt - drone_pos).norm();
                if (dist > detection_distance)
                    continue;
                if (!is_in_pyramid(pt, drone_pos, drone_yaw, pyramid_length, pyramid_width, pyramid_height))
                    continue;
                if (raycast_occluded(pt, drone_pos, points_voxels, voxel_size))
                    continue;
                point_seen[j] = true;
                cloud->points[j].r = 0;
                cloud->points[j].g = 255;
                cloud->points[j].b = 0;
            }

            auto stamp = this->now();
            publish_pointcloud(cloud, stamp);
            publish_pyramid_marker(pose.position, pose.orientation, stamp, pyramid_length, pyramid_width, pyramid_height);
            publish_pose(pose, stamp);
            publish_path(path_so_far, pose, stamp);

            // Wait until the right wall time for the next pose
            if (i + 1 < path.size()) {
                double csv_elapsed = path[i + 1].time - csv_start;
                double wall_elapsed = csv_elapsed / playback_speed;
                auto target_wall = wall_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(wall_elapsed));
                auto now = std::chrono::steady_clock::now();
                if (now < target_wall) {
                    std::this_thread::sleep_until(target_wall);
                } else if ((now - target_wall) > std::chrono::milliseconds(10)) {
                    RCLCPP_WARN(this->get_logger(),
                        "Processing is behind schedule by %.3f seconds at step %zu",
                        std::chrono::duration<double>(now - target_wall).count(), i);
                }
            }
        }

        size_t green_points = std::count(point_seen.begin(), point_seen.end(), true);
        size_t total_points = point_seen.size();
        double coverage = (total_points > 0) ? (100.0 * green_points / total_points) : 0.0;
        RCLCPP_INFO(this->get_logger(), "Coverage: %.2f%% (%zu / %zu)", coverage, green_points, total_points);
        RCLCPP_INFO(this->get_logger(), "Processing complete.");
    }

private:
    // Helper: Load CSV path
    std::vector<PoseData> load_path_from_csv(const std::string& csv_file, int stride) {
        std::vector<PoseData> path;
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Could not open CSV file: %s", csv_file.c_str());
            return path;
        }
        std::string line;
        std::getline(file, line); // skip header
        int idx = 0;
        while (std::getline(file, line)) {
            if (stride > 1 && (idx % stride) != 0) {
                ++idx;
                continue;
            }
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> tokens;
            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }
            if (tokens.size() < 8) continue;
            PoseData pd;
            pd.time = std::stod(tokens[0]);
            pd.position = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
            pd.orientation = Eigen::Quaterniond(std::stod(tokens[7]), std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]));
            path.push_back(pd);
            ++idx;
        }
        return path;
    }

    // Helper: Get yaw from quaternion
    double get_yaw_from_quaternion(const Eigen::Quaterniond& q) {
        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // Helper: Pyramid check
    bool is_in_pyramid(const Eigen::Vector3d& pt, const Eigen::Vector3d& drone_pos, double drone_yaw,
                       double pyramid_length, double pyramid_width, double pyramid_height)
    {
        Eigen::Vector3d rel = pt - drone_pos;
        double c = std::cos(-drone_yaw), s = std::sin(-drone_yaw);
        double x = c * rel.x() - s * rel.y();
        double y = s * rel.x() + c * rel.y();
        double z = rel.z();
        if (x < 0 || x > pyramid_length) return false;
        double half_w = (pyramid_width / 2.0) * (x / pyramid_length);
        double half_h = (pyramid_height / 2.0) * (x / pyramid_length);
        if (std::abs(y) > half_w || std::abs(z) > half_h) return false;
        return true;
    }

    bool raycast_occluded(const Eigen::Vector3d& pt, const Eigen::Vector3d& drone_pos,
                        const std::unordered_set<std::tuple<int,int,int>>& points_set, double voxel_size)
    {
        Eigen::Vector3d dir = pt - drone_pos;
        double dist = dir.norm();
        if (dist < 1e-6) return false;
        dir /= dist;

        // Start and end voxel
        auto voxel_from = std::make_tuple(
            static_cast<int>(std::floor(drone_pos.x() / voxel_size)),
            static_cast<int>(std::floor(drone_pos.y() / voxel_size)),
            static_cast<int>(std::floor(drone_pos.z() / voxel_size))
        );
        auto voxel_to = std::make_tuple(
            static_cast<int>(std::floor(pt.x() / voxel_size)),
            static_cast<int>(std::floor(pt.y() / voxel_size)),
            static_cast<int>(std::floor(pt.z() / voxel_size))
        );

        // Current voxel
        int x = std::get<0>(voxel_from);
        int y = std::get<1>(voxel_from);
        int z = std::get<2>(voxel_from);

        int x_end = std::get<0>(voxel_to);
        int y_end = std::get<1>(voxel_to);
        int z_end = std::get<2>(voxel_to);

        // Step direction
        int stepX = (dir.x() > 0) ? 1 : -1;
        int stepY = (dir.y() > 0) ? 1 : -1;
        int stepZ = (dir.z() > 0) ? 1 : -1;

        // Compute tMax (distance to first voxel boundary) and tDelta (step size per voxel)
        auto voxel_boundary = [&](double coord, double d, int v) {
            return (d > 0) ? ( (v + 1) * voxel_size - coord ) / d
                        : ( coord - v * voxel_size ) / -d;
        };

        double tMaxX = voxel_boundary(drone_pos.x(), dir.x(), x);
        double tMaxY = voxel_boundary(drone_pos.y(), dir.y(), y);
        double tMaxZ = voxel_boundary(drone_pos.z(), dir.z(), z);

        double tDeltaX = std::abs(voxel_size / dir.x());
        double tDeltaY = std::abs(voxel_size / dir.y());
        double tDeltaZ = std::abs(voxel_size / dir.z());

        // Traverse voxels until we reach the target voxel
        while (!(x == x_end && y == y_end && z == z_end)) {
            auto voxel = std::make_tuple(x, y, z);
            if (points_set.count(voxel) && voxel != voxel_to) {
                return true; // occluded
            }

            // Step to next voxel
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += stepX;
                    tMaxX += tDeltaX;
                } else {
                    z += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += stepY;
                    tMaxY += tDeltaY;
                } else {
                    z += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
        }

        return false; // no occlusion
    }

    // Helper: Publish point cloud
    void publish_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const rclcpp::Time& stamp) {
        sensor_msgs::msg::PointCloud2 pc_msg;
        pcl::toROSMsg(*cloud, pc_msg);
        pc_msg.header.frame_id = frame_id_;
        pc_msg.header.stamp = stamp;
        pc_pub_->publish(pc_msg);
    }

    // Helper: Publish pyramid marker
    void publish_pyramid_marker(const Eigen::Vector3d& pos, const Eigen::Quaterniond& q, const rclcpp::Time& stamp,
                               double L, double W, double H)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = stamp;
        marker.ns = "pyramid";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        std::vector<Eigen::Vector3d> corners = {
            {0, 0, 0}, // tip (drone)
            {L, -W/2, -H/2},
            {L, -W/2, H/2},
            {L, W/2, H/2},
            {L, W/2, -H/2}
        };
        Eigen::Matrix3d R = q.toRotationMatrix();
        for (auto& c : corners) c = R * c + pos;

        std::vector<std::pair<int, int>> lines = {
            {0,1},{0,2},{0,3},{0,4}, // tip to base
            {1,2},{2,3},{3,4},{4,1}  // base edges
        };
        for (const auto& l : lines) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = corners[l.first](0); p1.y = corners[l.first](1); p1.z = corners[l.first](2);
            p2.x = corners[l.second](0); p2.y = corners[l.second](1); p2.z = corners[l.second](2);
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }
        marker_pub_->publish(marker);
    }

    // Helper: Publish pose
    void publish_pose(const PoseData& pose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.frame_id = frame_id_;
        pose_msg.header.stamp = stamp;
        pose_msg.pose.position.x = pose.position.x();
        pose_msg.pose.position.y = pose.position.y();
        pose_msg.pose.position.z = pose.position.z();
        pose_msg.pose.orientation.x = pose.orientation.x();
        pose_msg.pose.orientation.y = pose.orientation.y();
        pose_msg.pose.orientation.z = pose.orientation.z();
        pose_msg.pose.orientation.w = pose.orientation.w();
        pose_pub_->publish(pose_msg);
    }

    // Helper: Publish path so far
    void publish_path(nav_msgs::msg::Path& path_so_far, const PoseData& pose, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.frame_id = frame_id_;
        pose_msg.header.stamp = stamp;
        pose_msg.pose.position.x = pose.position.x();
        pose_msg.pose.position.y = pose.position.y();
        pose_msg.pose.position.z = pose.position.z();
        pose_msg.pose.orientation.x = pose.orientation.x();
        pose_msg.pose.orientation.y = pose.orientation.y();
        pose_msg.pose.orientation.z = pose.orientation.z();
        pose_msg.pose.orientation.w = pose.orientation.w();
        path_so_far.header.stamp = stamp;
        path_so_far.poses.push_back(pose_msg);
        path_pub_->publish(path_so_far);
    }

    std::string frame_id_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ResultDisplayer>();
    rclcpp::shutdown();
    return 0;
}