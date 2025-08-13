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

cd ~/workspaces/isaac_ros-dev/ && \
git submodule update --init --recursive
```

3. **Set the workspace environment variable**:

```
echo "export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev/" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=<your_domain_id>" >> ~/.bashrc
source ~/.bashrc
```

4. **Setup Simulation Environment**:
```
echo 'alias pegasus_launch="cd ${ISAAC_ROS_WS} && ./src/osep_simulation_environment/launch_pegasus.sh"' >> ~/.bashrc

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
</details>

