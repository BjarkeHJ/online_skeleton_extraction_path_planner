#!/bin/bash

echo "Fetching necessary submodules"

SUBMODULES=(
    "src/px4_msgs"
    "src/osep_simulation_environment"
    "src/osep_core"
)

for submodule in "${SUBMODULES[@]}"; do
    echo "Initializing and updating submodule: $submodule"
    git submodule update --init --recursive "$submodule"
done

echo "Selected submodules (and their nested submodules) initialized."

# Update package lists to ensure we have the latest information
echo "Updating package lists..."
sudo apt-get update

# List of package names for colcon
PACKAGE_NAMES="
    px4_msgs
    osep_simulation_environment
    osep_core
    osep
"

# Derive rosdep package paths from colcon package names
INSTALL_PACKAGES=$(echo ${PACKAGE_NAMES} | sed "s~[^ ]*~${OSEP_ROS_WS}/src/&~g")

# Update rosdep and install dependencies from the workspace, excluding the ignored packages
echo "Updating rosdep and installing dependencies from ${OSEP_ROS_WS}/src..."
rosdep update
rosdep install -i -r --from-paths ${INSTALL_PACKAGES} --rosdistro humble -y

# Navigate to the OSEP workspace
echo "Navigating to ${OSEP_ROS_WS}..."
cd ${OSEP_ROS_WS}

# Build the workspace using colcon, ignoring the specified packages
echo "Building the workspace with colcon..."
colcon build --symlink-install --base-paths ${INSTALL_PACKAGES}

echo "Script completed successfully!"