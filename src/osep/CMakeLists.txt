cmake_minimum_required(VERSION 3.10)
project(osep)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

add_executable(tsdf_to_pointcloud_node src/tsdf_to_pointcloud.cpp)
target_include_directories(tsdf_to_pointcloud_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tsdf_to_pointcloud_node
  rclcpp
  sensor_msgs
  nvblox_msgs
)

install(TARGETS tsdf_to_pointcloud_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()