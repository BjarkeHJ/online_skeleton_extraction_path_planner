# This file is copied from the root launch directory for packaging
from launch import LaunchDescription
from launch_ros.actions import Node

FRAME_ID = "base_link"
SAFETY_DISTANCE = 10.0
INTERPOLATION_DISTANCE = 3.0
INSPECTION_SPEED = 2.5
CLEARING_DISTANCE = 1.0


TOPIC_NAMES = {
    "VEL_CMD": '/osep/vel_cmd',
    "PATH": '/osep/path',
    "COSTMAP": '/osep/local_costmap/costmap',
    "VIEWPOINTS": '/osep/viewpoints',
    "VIEWPOINTS_ADJUSTED": '/osep/viewpoints_adjusted',
    "GROUND_TRUTH": '/osep/ground_truth'
}

def generate_launch_description():
    return LaunchDescription([
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
                {"costmap_topic": TOPIC_NAMES["COSTMAP"]},
            ]
        ),
        Node(
            package="osep_2d_map",
            executable="path_interpolator_node",
            name="planner",
            output="screen",
            parameters=[
                {"frame_id": FRAME_ID},
                {"interpolation_distance": INTERPOLATION_DISTANCE},
                {"costmap_topic": TOPIC_NAMES["COSTMAP"]},
                {"viewpoints_topic": TOPIC_NAMES["VIEWPOINTS"]},
                {"path_topic": TOPIC_NAMES["PATH"]},
                {"ground_truth_topic": TOPIC_NAMES["GROUND_TRUTH"]},
                {"viewpoints_adjusted_topic": TOPIC_NAMES["VIEWPOINTS_ADJUSTED"]},
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