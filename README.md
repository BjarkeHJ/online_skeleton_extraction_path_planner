# Online Skeleton Extraction Path Planner (OSEP)
This guide provides instructions for setting up the OSEP environment.


</details>

<details>
<summary> <b>Workspace Setup</b> </summary>

This guide is based on a slightly modified version from [Isaac ROS NVBlox Setup](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/index.html#set-up-package-name).

1. **Create a workspace directory**:

```
mkdir -p  ~/workspaces/
```

2. **Clone the OSEP repository**:
```
cd  ~/workspaces && \
git clone https://github.com/BjarkeHJ/online_skeleton_extraction_path_planner.git isaac_ros-dev
```

3. **Set the workspace environment variable**:

```
echo "export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=21" >> ~/.bashrc
source ~/.bashrc
```

4. **Setup Simulation Environment**:
```
echo -e '\npegasus_launch() {\n    cd "${ISAAC_ROS_WS}" && ./src/osep_simulation_environment/launch_pegasus.sh\n}\n' >> ~/.bashrc

source ~/.bashrc
```

5. **Setup Docker Environment**:
```
cd ${ISAAC_ROS_WS} && \
./scripts/docker_env_setup.sh
```
</details>




<details>
<summary><b>Using the Workspace</b></summary>

1. **Launching Simulation Environment**

To launch the simulation environment, run the following commands:

```
pegasus_launch
```


2. **Launching Docker**

To launch the Docker container, run the following commands:

```
cd $ISAAC_ROS_WS/src/isaac_ros_common && \
./scripts/run_dev.sh
```
Inside the docker conainter, you need to build the work space

```
cd ${ISAAC_ROS_WS}
./scripts/build_docker_workspace.sh
```

3. **Running OSEP**

Inside the docker container run:
```
source install/setup.bash
ros2 launch osep osep.launch.py
```

In another docker terminal:
```
source install/setup.bash
ros2 launch osep 2d_nvblox.launch.py
```
</details>

