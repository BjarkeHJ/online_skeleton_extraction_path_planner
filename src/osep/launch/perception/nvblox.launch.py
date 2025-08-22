# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from launch import Action, LaunchDescription
from launch_ros.descriptions import ComposableNode
import isaac_ros_launch_utils as lu

from nvblox_ros_python_utils.nvblox_launch_utils import NvbloxMode, NvbloxCamera
from nvblox_ros_python_utils.nvblox_constants import NVBLOX_CONTAINER_NAME

def get_isaac_sim_remappings(num_cameras: int, lidar: bool) -> List[Tuple[str, str]]:
    remappings = []
    camera_names = ['isaac/stereo_camera/left'][:num_cameras]
    for i, name in enumerate(camera_names):
        remappings.append((f'camera_{i}/depth/image', f'{name}/depth'))
        remappings.append((f'camera_{i}/depth/camera_info', f'{name}/camera_info'))
        remappings.append((f'camera_{i}/color/image', f'{name}/image'))
        remappings.append((f'camera_{i}/color/camera_info', f'{name}/camera_info'))
    if lidar:
        remappings.append(('pointcloud', '/isaac/lidar/raw/pointcloud'))
    return remappings

def add_nvblox(args: lu.ArgumentContainer) -> List[Action]:

    mode = NvbloxMode[args.mode]
    camera = NvbloxCamera[args.camera]
    num_cameras = int(args.num_cameras)
    use_lidar = lu.is_true(args.lidar)
    esdf_mode = args.esdf_mode

    if esdf_mode == '3d':
        base_config = lu.get_path('osep', 'config/nvblox/nvblox_3d.yaml')
    else:
        base_config = lu.get_path('osep', 'config/nvblox/nvblox_2d.yaml')

    remappings = get_isaac_sim_remappings(num_cameras, use_lidar)

    parameters = []
    parameters.append(base_config)
    parameters.append({'num_cameras': num_cameras})
    parameters.append({'use_lidar': use_lidar})

    # Add the nvblox node.
    nvblox_node = ComposableNode(
        name='nvblox_node',
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        remappings=remappings,
        parameters=parameters,
    )

    actions = []
    if args.run_standalone:
        actions.append(lu.component_container(args.container_name))
    actions.append(lu.load_composable_nodes(args.container_name, [nvblox_node]))
    actions.append(
        lu.log_info(
            ["Starting nvblox with the '",
             str(camera), "' camera in '",
             str(mode), "' mode."]))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('mode')
    args.add_arg('camera')
    args.add_arg('num_cameras', 1)
    args.add_arg('lidar', 'False')
    args.add_arg('container_name', NVBLOX_CONTAINER_NAME)
    args.add_arg('run_standalone', 'False')
    args.add_arg('esdf_mode', '3d')

    args.add_opaque_function(add_nvblox)
    return LaunchDescription(args.get_launch_actions())
