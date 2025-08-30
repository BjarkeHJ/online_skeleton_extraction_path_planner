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
#include <cmath>
#include <functional>
#include <chrono>
#include <thread>
#include <limits>
#include <algorithm>

struct PoseData {
    double time;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

// ------------------------
// Fast voxel key (pack into uint64_t)
// 21 bits per axis, bias to support negatives
// supports coords in approx [-1,048,575, +1,048,575]
// ------------------------
static inline uint64_t pack_voxel(int x, int y, int z) {
    constexpr uint32_t MASK21 = 0x1FFFFF; // 21 bits
    constexpr int32_t BIAS = 1 << 20;     // 1,048,576
    uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(x + BIAS) & MASK21);
    uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(y + BIAS) & MASK21);
    uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(z + BIAS) & MASK21);
    return (ux << 42) | (uy << 21) | uz;
}

using VoxelSet = std::unordered_set<uint64_t>;

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
        voxel_size_ = this->get_parameter("voxel_size").as_double();
        pyramid_length_ = this->get_parameter("pyramid_length").as_double();
        pyramid_width_ = this->get_parameter("pyramid_width").as_double();
        pyramid_height_ = this->get_parameter("pyramid_height").as_double();
        detection_distance_ = this->get_parameter("detection_distance").as_double();
        playback_speed_ = this->get_parameter("playback_speed").as_double();

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
        cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_->points.resize(cloud_xyz->points.size());
        cloud_->width = cloud_xyz->width;
        cloud_->height = cloud_xyz->height;
        cloud_->is_dense = cloud_xyz->is_dense;
        for (size_t i = 0; i < cloud_xyz->points.size(); ++i) {
            cloud_->points[i].x = cloud_xyz->points[i].x;
            cloud_->points[i].y = cloud_xyz->points[i].y;
            cloud_->points[i].z = cloud_xyz->points[i].z;
            cloud_->points[i].r = 255;
            cloud_->points[i].g = 0;
            cloud_->points[i].b = 0;
        }
        point_seen_.assign(cloud_->points.size(), false);

        nav_msgs::msg::Path path_so_far;
        path_so_far.header.frame_id = frame_id_;

        // Build a set of occupied voxels for fast lookup (for occlusion)
        build_voxel_set(cloud_, voxel_size_, points_voxels_);

        // Real-time playback loop (optimized tests: dist^2, frustum then raycast)
        auto wall_start = std::chrono::steady_clock::now();
        double csv_start = path.front().time;

        // Precompute squared detection distance
        double detection_distance2 = detection_distance_ * detection_distance_;
        const double inv_voxel = 1.0 / voxel_size_;

        for (size_t i = 0; i < path.size(); ++i) {
            const auto& pose = path[i];
            const Eigen::Vector3d drone_pos = pose.position;
            double drone_yaw = get_yaw_from_quaternion(pose.orientation);
            const double c = std::cos(-drone_yaw);
            const double s = std::sin(-drone_yaw);

            // Only check unseen points; update color only when a point becomes seen
            for (size_t j = 0; j < cloud_->points.size(); ++j) {
                if (point_seen_[j]) continue;

                const auto &p = cloud_->points[j];
                // squared distance test
                const double dx = p.x - drone_pos.x();
                const double dy = p.y - drone_pos.y();
                const double dz = p.z - drone_pos.z();
                const double d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > detection_distance2) continue;

                // frustum / pyramid test (fast inline)
                Eigen::Vector3d rel(dx, dy, dz);
                double x = c * rel.x() - s * rel.y();
                double y = s * rel.x() + c * rel.y();
                double z = rel.z();
                if (x < 0.0 || x > pyramid_length_) continue;
                double half_w = (pyramid_width_ / 2.0) * (x / pyramid_length_);
                double half_h = (pyramid_height_ / 2.0) * (x / pyramid_length_);
                if (std::abs(y) > half_w || std::abs(z) > half_h) continue;

                // raycast occlusion test (voxel DDA). note: we pass the packed voxel set
                if (raycast_occluded(Eigen::Vector3d(p.x, p.y, p.z), drone_pos, points_voxels_, voxel_size_)) continue;

                // Mark seen and recolor once
                point_seen_[j] = true;
                cloud_->points[j].r = 0;
                cloud_->points[j].g = 255;
                cloud_->points[j].b = 0;
            }

            auto stamp = this->now();
            publish_pointcloud(cloud_, stamp);
            publish_pyramid_marker(pose.position, pose.orientation, stamp, pyramid_length_, pyramid_width_, pyramid_height_);
            publish_pose(pose, stamp);
            publish_path(path_so_far, pose, stamp);

            // Wait until the right wall time for the next pose
            if (i + 1 < path.size()) {
                double csv_elapsed = path[i + 1].time - csv_start;
                double wall_elapsed = csv_elapsed / playback_speed_;
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

        size_t green_points = std::count(point_seen_.begin(), point_seen_.end(), true);
        size_t total_points = point_seen_.size();
        double coverage = (total_points > 0) ? (100.0 * green_points / total_points) : 0.0;
        RCLCPP_INFO(this->get_logger(), "Coverage: %.2f%% (%zu / %zu)", coverage, green_points, total_points);
        RCLCPP_INFO(this->get_logger(), "Processing complete.");
    }

private:
    // --------------------
    // CSV loader
    // --------------------
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
            if (tokens.size() < 8) { ++idx; continue; }
            PoseData pd;
            pd.time = std::stod(tokens[0]);
            pd.position = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
            pd.orientation = Eigen::Quaterniond(std::stod(tokens[7]), std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]));
            path.push_back(pd);
            ++idx;
        }
        return path;
    }

    // --------------------
    // Quaternion -> yaw
    // --------------------
    double get_yaw_from_quaternion(const Eigen::Quaterniond& q) {
        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // --------------------
    // Build voxel set from PCL cloud (fast)
    // --------------------
    static inline int ffloor_to_int(double v) {
        return static_cast<int>(std::floor(v));
    }

    void build_voxel_set(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, double voxel_size, VoxelSet& out_set) {
        out_set.clear();
        out_set.reserve(std::max<size_t>(256, cloud->points.size() * 2));
        const double inv_voxel = 1.0 / voxel_size;
        for (const auto& p : cloud->points) {
            int vx = ffloor_to_int(p.x * inv_voxel);
            int vy = ffloor_to_int(p.y * inv_voxel);
            int vz = ffloor_to_int(p.z * inv_voxel);
            out_set.emplace(pack_voxel(vx, vy, vz));
        }
    }

    // --------------------
    // 3D DDA raycast using packed voxel keys
    // Returns true if the point is occluded (an occupied voxel closer than the target)
    // --------------------
    bool raycast_occluded(const Eigen::Vector3d& pt, const Eigen::Vector3d& drone_pos,
                            const VoxelSet& points_set, double voxel_size) const
        {
            Eigen::Vector3d dir = pt - drone_pos;
            const double dist = dir.norm();
            if (dist < 1e-9) return false;
            dir /= dist;

            const double inv_voxel = 1.0 / voxel_size;
            int x = ffloor_to_int(drone_pos.x() * inv_voxel);
            int y = ffloor_to_int(drone_pos.y() * inv_voxel);
            int z = ffloor_to_int(drone_pos.z() * inv_voxel);
            const int x_end = ffloor_to_int(pt.x() * inv_voxel);
            const int y_end = ffloor_to_int(pt.y() * inv_voxel);
            const int z_end = ffloor_to_int(pt.z() * inv_voxel);
            const uint64_t target_key = pack_voxel(x_end, y_end, z_end);

            const int stepX = (dir.x() > 0.0) ? 1 : -1;
            const int stepY = (dir.y() > 0.0) ? 1 : -1;
            const int stepZ = (dir.z() > 0.0) ? 1 : -1;

            auto first_boundary = [&](double pos, double d, int v)->double {
                if (std::abs(d) < 1e-12) return std::numeric_limits<double>::infinity();
                double next_plane = (d > 0.0) ? ((v + 1) * voxel_size) : (v * voxel_size);
                return (next_plane - pos) / d;
            };
            auto delta_t = [&](double d)->double {
                return (std::abs(d) < 1e-12) ? std::numeric_limits<double>::infinity() : voxel_size / std::abs(d);
            };

            double tMaxX = first_boundary(drone_pos.x(), dir.x(), x);
            double tMaxY = first_boundary(drone_pos.y(), dir.y(), y);
            double tMaxZ = first_boundary(drone_pos.z(), dir.z(), z);

            double tDeltaX = delta_t(dir.x());
            double tDeltaY = delta_t(dir.y());
            double tDeltaZ = delta_t(dir.z());

            // Guard iterations to avoid pathological infinite loops
            const int max_iters = std::max(16, 3 + static_cast<int>(std::ceil(dist * inv_voxel) * 3));
            int iters = 0;

            while (!(x == x_end && y == y_end && z == z_end) && iters++ < max_iters) {
                uint64_t key = pack_voxel(x, y, z);
                if (key != target_key && points_set.find(key) != points_set.end()) {
                    return true; // occluded
                }

                // Advance along smallest tMax
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

            // final check inside target voxel is not necessary (we ignore target voxel)
            return false;
        }

    // --------------------
    // Publishing helpers
    // --------------------
    void publish_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const rclcpp::Time& stamp) {
        sensor_msgs::msg::PointCloud2 pc_msg;
        pcl::toROSMsg(*cloud, pc_msg);
        pc_msg.header.frame_id = frame_id_;
        pc_msg.header.stamp = stamp;
        pc_pub_->publish(pc_msg);
    }

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
        marker.scale.x = 0.1;  // line thickness
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        // Pyramid corners in local frame
        std::vector<Eigen::Vector3d> corners = {
            {0, 0, 0},           // tip (drone position)
            {L, -W/2, -H/2},     // base corners
            {L, -W/2,  H/2},
            {L,  W/2,  H/2},
            {L,  W/2, -H/2}
        };

        // Transform corners into world frame
        Eigen::Matrix3d R = q.toRotationMatrix();
        for (auto& c : corners) {
            c = R * c + pos;
        }

        // Define edges of pyramid
        std::vector<std::pair<int, int>> lines = {
            {0,1},{0,2},{0,3},{0,4}, // tip to base
            {1,2},{2,3},{3,4},{4,1}  // base edges
        };

        // Convert to ROS points
        for (const auto& l : lines) {
            geometry_msgs::msg::Point p1, p2;

            p1.x = corners[l.first].x();
            p1.y = corners[l.first].y();
            p1.z = corners[l.first].z();

            p2.x = corners[l.second].x();
            p2.y = corners[l.second].y();
            p2.z = corners[l.second].z();

            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }

        marker_pub_->publish(marker);
    }


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

    // --------------------
    // Members
    // --------------------
    std::string frame_id_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
    std::vector<char> point_seen_; // char uses less memory than bool vector specialization
    VoxelSet points_voxels_;

    double voxel_size_;
    double pyramid_length_;
    double pyramid_width_;
    double pyramid_height_;
    double detection_distance_;
    double playback_speed_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ResultDisplayer>();
    rclcpp::shutdown();
    return 0;
}
