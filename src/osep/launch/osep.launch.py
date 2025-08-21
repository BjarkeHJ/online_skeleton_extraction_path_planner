# This file is copied from the root launch directory for packaging
from launch import LaunchDescription
from launch_ros.actions import Node

FRAME_ID = "base_link"
SAFETY_DISTANCE = 10.0

TOPIC_NAMES = {
    "VEL_CMD": '/osep/vel_cmd',
    "PATH": '/planner/smoothed_path',
    "COSTMAP": '/local_costmap/costmap',
    "VIEWPOINTS": '/osep/viewpoints'
}

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osep_simulation_environment',
            executable='px4_msg_converter_node',
            name='px4_msg_converter_node',
            parameters=[{
                'osep_vel_cmd': TOPIC_NAMES["VEL_CMD"]
            }]
        ),
        Node(
            package='osep_simulation_environment',
            executable='px4_vel_controller',
            name='px4_vel_controller',
            parameters=[{
                'path_topic': TOPIC_NAMES["PATH"],
                'osep_vel_cmd': TOPIC_NAMES["VEL_CMD"],
                'interpolation_distance': 3.0,
                'max_speed': 15.0,
                'inspection_speed': 2.0,
                'max_yaw_to_velocity_angle_deg': 120.0,
                'frequency': 100
            }]
        ),
        Node(
            package="osep_2d_map",
            executable="costmap_2d_node",
            name="costmap_2d_node",
            output="screen",
            parameters=[
                {"resolution": 1.0},
                {"free_center_radius": 5.0},
                {"local_map_size": 400.0},
                {"global_map_size": 1600.0},
                {"frame_id": FRAME_ID},
                {"safety_distance": SAFETY_DISTANCE},
            ]
        ),
        Node(
            package="osep_2d_map",
            executable="path_interpolator_node",
            name="planner",
            output="screen",
            parameters=[
                {"frame_id": FRAME_ID},
                {"interpolation_distance": 2.0},
                {"costmap_topic": TOPIC_NAMES["COSTMAP"]},
                {"waypoints_topic": TOPIC_NAMES["VIEWPOINTS"]},
                {"path_planner_prefix": "/planner"},
                {"ground_truth_update_interval": 8000},
                {"safety_distance": SAFETY_DISTANCE},
            ]
        ),
        Node(
            package="osep_skeleton_decomp",
            executable="osep_rosa",
            name="RosaNode",
            output="screen",
            parameters=[

            ]
        ),
        Node(
            package="osep_skeleton_decomp",
            executable="osep_gskel",
            name="GSkelNode",
            output="screen",
            parameters=[

            ]
        ),
        Node(
            package="osep_planning",
            executable="osep_planner",
            name="PlannerNode",
            output="screen",
            parameters=[

            ]
        ),
        Node(
            package="osep_planning",
            executable="osep_viewpoint_manager",
            name="ViewpointNode",
            output="screen",
            parameters=[

            ]
        )
    ])