from launch import LaunchDescription
from launch_ros.actions import Node

FRAME_ID = "base_link"
SAFETY_DISTANCE = 10.0
INTERPOLATION_DISTANCE = 3.0
INSPECTION_SPEED = 2.5
VOXEL_SIZE = 1.0

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep',
            executable='static_pointcloud_postprocessor_node',
            name='static_pointcloud_postprocessor_node',
            parameters=[{
                'input_topic': 'osep/tsdf/static_pointcloud',
                'output_topic': 'osep/tsdf/shell_static_pointcloud',
                'voxel_size': VOXEL_SIZE,
                'upsample_N': 2,
            }],
        ),
        Node(
            package='osep',
            executable='cloud_saver',
            name='cloud_saver',
            parameters=[{
                'topic_name': '/osep/tsdf/shell_static_pointcloud',
                'filename': 'shell_static_cloud_60.pcd'
            }]
        )
    ])