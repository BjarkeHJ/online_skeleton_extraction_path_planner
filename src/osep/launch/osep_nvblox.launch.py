from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep',
            executable='tsdf_to_pointcloud_node',
            name='tsdf_to_pointcloud_node',
            parameters=[{
                'output_topic': 'osep/tsdf_pointcloud',
                'static_output_topic': 'osep/static_tsdf_pointcloud'
            }],
            output='screen'
        ),
        Node(
            package='osep',
            executable='static_pointcloud_postprocessor_node',
            name='static_pointcloud_postprocessor_node',
            parameters=[{
                'input_topic': 'osep/static_tsdf_pointcloud',          
                'output_topic': 'osep/upsampled_static_pointcloud'
            }]
                    
        )
    ])