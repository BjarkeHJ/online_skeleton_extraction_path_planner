from launch import LaunchDescription
from launch_ros.actions import Node

FRAME_ID = "base_link"
SAFETY_DISTANCE = 10.0
INTERPOLATION_DISTANCE = 3.0
INSPECTION_SPEED = 2.5
VOXEL_SIZE = 0.5

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep',
            executable='tsdf_to_pointcloud_node',
            name='tsdf_to_pointcloud_node',
            parameters=[{
                'output_topic': 'osep/tsdf/pointcloud',
                'static_output_topic': 'osep/tsdf/static_pointcloud',
                'cavity_fill_diameter': 5.0,
                'voxel_size': VOXEL_SIZE
            }],
            output='screen'
        ),
        Node(
            package='osep',
            executable='static_pointcloud_postprocessor_node',
            name='static_pointcloud_postprocessor_node',
            parameters=[{
                'input_topic': 'osep/tsdf/static_pointcloud',
                'output_topic': 'osep/tsdf/upsampled_static_pointcloud',
                'voxel_size': VOXEL_SIZE,
                'upsample_N': 2
            }],
        ),
    ])