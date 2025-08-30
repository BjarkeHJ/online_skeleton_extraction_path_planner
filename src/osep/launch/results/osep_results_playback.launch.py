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
        # Node(
        #     package='osep',
        #     executable='result_processor',
        #     name='result_processor',
        #     parameters=[
        #         {'topic_name': '/osep/results_pcd'},
        #         {'filename': 'wind_0_voxels_0.1.pcd'},
        #         {'voxel_size': 0.1},
        #         {'pyramid_length': 20.0},
        #         {'pyramid_width': 15.0},
        #         {'pyramid_height': 15.0},
        #         {'frame_id': 'odom'},
        #         {'camera_frame': 'base_link'},
        #         {'detection_distance': 20.0}
        #     ]
        # ),
        # Node(
        #     package='osep',
        #     executable='ground_truth_republisher',
        #     name='ground_truth_republisher',
        #     parameters=[
        #         {'csv_file': 'wind_0_gt.csv'},
        #         {'topic': '/osep/replayed_path'},
        #         {'frame_id': 'odom'},
        #         {'playback_rate': 1.0},
        #         {'stride': 3}
        #     ]
        # ),
        Node(
            package='osep',
            executable='result_displayer_node',
            name='result_displayer',
            output='screen',
            parameters=[
                {'csv_file': 'wind_0_gt.csv'},
                {'pcd_file': 'wind_0_voxels_0.1.pcd'},
                {'frame_id': 'odom'},
                {'stride': 3},
                {'voxel_size': 0.1},
                {'pyramid_length': 20.0},
                {'pyramid_width': 15.0},
                {'pyramid_height': 15.0},
                {'detection_distance': 20.0}
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