from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep',
            executable='ground_truth_path_saver',
            name='ground_truth_path_saver',
            output='screen',
            parameters=[
                {'topic': '/osep/ground_truth'},
                {'result_file': 'wind_0_gt.csv'}
            ]
        )
    ])