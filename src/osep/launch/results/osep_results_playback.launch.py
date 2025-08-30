from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config_name = 'results.rviz'  # <-- Set your config file name here
    rviz_config_path = os.path.join(
        get_package_share_directory('osep'),
        'config/visualization',
        rviz_config_name
    )

    return LaunchDescription([
        Node(
            package='osep',
            executable='result_displayer_node',
            name='result_displayer',
            output='screen',
            parameters=[
                {'csv_file': 'wind_0_gt.csv'},
                {'pcd_file': 'wind_0_outer_shell_voxels_0.1.pcd'},
                {'frame_id': 'odom'},
                {'stride': 3},
                {'playback_speed': 10.0},
                {'voxel_size': 0.1},
                {'pyramid_length': 20.0},
                {'pyramid_width': 20.0},
                {'pyramid_height': 15.0},
                {'detection_distance': 25.0}
            ]
        ),
        Node(
            package='osep',
            executable='static_tf_publisher',
            name='static_tf_publisher',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        )
    ])