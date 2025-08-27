cmake_minimum_required(VERSION 3.10)
project(osep)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


add_executable(nvblox_voxel_map src/nvblox_voxel_map.cpp)
ament_target_dependencies(nvblox_voxel_map
  rclcpp
  sensor_msgs
  nvblox_msgs
)

install(TARGETS nvblox_voxel_map
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
