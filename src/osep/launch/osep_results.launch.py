from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
        package='osep',
        executable='cloud_publisher',
        name='cloud_publisher',
        parameters=[
            {'topic_name': '/osep/static_shell'},
            {'filename': 'shell_static_cloud_60.pcd'}
        ]
    )
    ])