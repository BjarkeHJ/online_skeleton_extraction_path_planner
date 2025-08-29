#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <set>
#include <tuple>
#include <Eigen/Core>

class StaticPointcloudPostprocessNode : public rclcpp::Node
{
public:
    StaticPointcloudPostprocessNode();

private:
    std::string static_input_topic_;
    std::string output_topic_;
    double voxel_size_;
    int upsample_N_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    size_t last_point_count_ = 0;
    sensor_msgs::msg::PointCloud2 last_msg_;

    static std::tuple<int, int, int> quantize(float x, float y, float z, float res);
    static Eigen::Vector3f voxel_center(int ix, int iy, int iz, float res);

    std::set<std::tuple<int,int,int>> upsample_voxels_by_2(const std::set<std::tuple<int,int,int>>& voxels);
    std::set<std::tuple<int,int,int>> full_erosion(const std::set<std::tuple<int,int,int>>& voxels, int erosion_threshold);
    std::set<std::tuple<int,int,int>> full_dilation(const std::set<std::tuple<int,int,int>>& voxels);
    std::set<std::tuple<int,int,int>> selective_dilation(const std::set<std::tuple<int,int,int>>& eroded);
    std::set<std::tuple<int,int,int>> extract_shell(const std::set<std::tuple<int,int,int>>& voxels);

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
};