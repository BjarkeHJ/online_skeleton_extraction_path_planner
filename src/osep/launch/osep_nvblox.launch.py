from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep',
            executable='tsdf_to_pointcloud_node',
            name='tsdf_to_pointcloud_node',
            parameters=[{'output_topic': 'osep/tsdf_pointcloud'}],  # Set your topic here
            output='screen'
        )
    ])