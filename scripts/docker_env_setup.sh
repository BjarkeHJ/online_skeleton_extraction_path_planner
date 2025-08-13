#!/bin/bash

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt Version 1.17.8


# Step 1: Set up NVIDIA container toolkit repository and install it
echo "Setting up NVIDIA container toolkit..."

# Add GPG key for the NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \

# Add NVIDIA container toolkit repository to the sources list
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update apt-get
sudo apt-get update

# Install the NVIDIA container toolkit
echo "Installing NVIDIA container toolkit..."
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the NVIDIA runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
sudo systemctl restart docker
sudo systemctl daemon-reload && sudo systemctl restart docker

# Step 2: Install Git Large File Storage (LFS)
echo "Installing Git LFS..."
sudo apt-get install git-lfs
git lfs install --skip-repo

# Step 3: Navigate to the Isaac ROS workspace
echo "Navigating to Isaac ROS workspace..."
cd ${ISAAC_ROS_WS}/src

# Step 4: Install necessary tools
echo "Installing additional tools..."
sudo apt-get install -y curl jq tar

# Step 5: Define variables for NGC resource
NGC_ORG="nvidia"
NGC_TEAM="isaac"
PACKAGE_NAME="isaac_ros_nvblox"
NGC_RESOURCE="isaac_ros_nvblox_assets"
NGC_FILENAME="quickstart.tar.gz"
MAJOR_VERSION=3
MINOR_VERSION=2

# Step 6: Fetch the latest version of the package
echo "Fetching the latest version of $NGC_RESOURCE..."

VERSION_REQ_URL="https://catalog.ngc.nvidia.com/api/resources/versions?orgName=$NGC_ORG&teamName=$NGC_TEAM&name=$NGC_RESOURCE&isPublic=true&pageNumber=0&pageSize=100&sortOrder=CREATED_DATE_DESC"
AVAILABLE_VERSIONS=$(curl -s -H "Accept: application/json" "$VERSION_REQ_URL")

# Extract the latest compatible version
LATEST_VERSION_ID=$(echo $AVAILABLE_VERSIONS | jq -r "
    .recipeVersions[]
    | .versionId as \$v
    | \$v | select(test(\"^\\\\d+\\\\.\\\\d+\\\\.\\\\d+$\"))
    | split(\".\") | {major: .[0]|tonumber, minor: .[1]|tonumber, patch: .[2]|tonumber}
    | select(.major == $MAJOR_VERSION and .minor <= $MINOR_VERSION)
    | \$v
    " | sort -V | tail -n 1)

# Step 7: Download and extract the assets
if [ -z "$LATEST_VERSION_ID" ]; then
    echo "No corresponding version found for Isaac ROS $MAJOR_VERSION.$MINOR_VERSION"
    echo "Found versions:"
    echo $AVAILABLE_VERSIONS | jq -r '.recipeVersions[].versionId'
else
    echo "Downloading Isaac ROS assets..."
    mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets

    FILE_REQ_URL="https://api.ngc.nvidia.com/v2/resources/$NGC_ORG/$NGC_TEAM/$NGC_RESOURCE/versions/$LATEST_VERSION_ID/files/$NGC_FILENAME"
    curl -LO --request GET "${FILE_REQ_URL}"

    # Extract the downloaded tar file
    tar -xf ${NGC_FILENAME} -C ${ISAAC_ROS_WS}/isaac_ros_assets
    rm ${NGC_FILENAME}
fi

# Step 8: Set up ROS
export ROS_DOMAIN_ID=21

echo "Setup complete!"
