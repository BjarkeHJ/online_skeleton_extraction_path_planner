#!/bin/bash

# Update package lists to ensure we have the latest information
echo "Updating package lists..."
sudo apt-get update

# List of package names (used for colcon)
PACKAGE_NAMES="
    isaac_ros_nvblox
    px4_msgs
    osep_nvblox_utils
    osep_core
    osep_simulation_environment
    osep
"

# Derive rosdep package paths from package names
INSTALL_PACKAGES=$(echo ${PACKAGE_NAMES} | sed "s~[^ ]*~${ISAAC_ROS_WS}/src/&~g")

# Update rosdep and install dependencies from the workspace
echo "Updating rosdep and installing dependencies from ${ISAAC_ROS_WS}/src..."
rosdep update
rosdep install -i -r --from-paths ${INSTALL_PACKAGES} --rosdistro humble -y

# Navigate to the isaac_ros-dev workspace
echo "Navigating to ${ISAAC_ROS_WS}..."
cd ${ISAAC_ROS_WS}

# Build the workspace using colcon
echo "Building the workspace with colcon..."
colcon build --symlink-install --base-paths ${INSTALL_PACKAGES}

echo "Script completed successfully!"