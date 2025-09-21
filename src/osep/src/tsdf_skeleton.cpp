#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <set>
#include <tuple>
#include <cmath>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <random>

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

        // Convert to OpenCV matrix
        cv::Mat data(cloud->points.size(), 3, CV_32F);
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            data.at<float>(i, 0) = cloud->points[i].x;
            data.at<float>(i, 1) = cloud->points[i].y;
            data.at<float>(i, 2) = cloud->points[i].z;
        }

        // GMM clustering using OpenCV EM with elbow detection
        int max_clusters = 20;
        double threshold = 0.01; // 1% relative improvement cutoff
        std::vector<double> bics;
        std::vector<cv::Ptr<cv::ml::EM>> models;
        int elbow_idx = -1;
        int best_k = 1;
        double best_bic = std::numeric_limits<double>::max();
        std::vector<int> best_labels;
        cv::Ptr<cv::ml::EM> best_em;

        for (int k = 1; k <= max_clusters; ++k) {
            auto em = cv::ml::EM::create();
            em->setClustersNumber(k);
            em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
            em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.1));
            cv::Mat labels;
            em->trainEM(data, cv::noArray(), labels, cv::noArray());

            // Correct log-likelihood calculation
            double log_likelihood = 0.0;
            cv::Mat probs;
            for (int i = 0; i < data.rows; ++i) {
                cv::Vec2d result = em->predict2(data.row(i), probs);
                log_likelihood += result[0];
            }

            int n_params = k * (3 + 3 * (3 + 1) / 2); // mean + cov for each cluster
            double bic = -2 * log_likelihood + n_params * std::log(data.rows);

            bics.push_back(bic);
            models.push_back(em);

            RCLCPP_INFO(this->get_logger(), "k=%d, BIC=%.2f, log_likelihood=%.2f", k, bic, log_likelihood);

            if (k > 1) {
                double improvement = -(bics.back() - bics[bics.size() - 2]) / std::abs(bics[bics.size() - 2]);
                if (improvement < threshold) {
                    elbow_idx = k - 1;
                    RCLCPP_INFO(this->get_logger(), "Elbow found at k=%d (improvement=%.4f < %.2f)", elbow_idx, improvement, threshold);
                    break;
                }
            }
        }

        if (elbow_idx == -1) {
            // No elbow found, pick k with minimum BIC
            elbow_idx = std::min_element(bics.begin(), bics.end()) - bics.begin() + 1;
        }
        best_k = elbow_idx;
        best_em = models[best_k - 1];

        // Get labels for best_k
        cv::Mat labels;
        best_em->trainEM(data, cv::noArray(), labels, cv::noArray());
        best_labels.resize(labels.rows);
        for (int i = 0; i < labels.rows; ++i) {
            best_labels[i] = labels.at<int>(i, 0);
        }

        RCLCPP_INFO(this->get_logger(), "Optimal number of clusters: %d", best_k);

        // Assign colors to clusters
        std::vector<cv::Vec3b> colors(best_k);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int i = 0; i < best_k; ++i) {
            colors[i] = cv::Vec3b(dist(rng), dist(rng), dist(rng));
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        colored_cloud->header = cloud->header;
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            pcl::PointXYZRGB pt;
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            int label = best_labels[i];
            pt.r = colors[label][0];
            pt.g = colors[label][1];
            pt.b = colors[label][2];
            colored_cloud->points.push_back(pt);
        }
        colored_cloud->width = colored_cloud->points.size();
        colored_cloud->height = 1;
        colored_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*colored_cloud, output);
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