cmake_minimum_required(VERSION 3.10)
project(osep)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
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

add_executable(static_pointcloud_postprocessor_node src/static_pointcloud_processor.cpp)
target_include_directories(static_pointcloud_postprocessor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(static_pointcloud_postprocessor_node ${PCL_LIBRARIES})
ament_target_dependencies(static_pointcloud_postprocessor_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)
install(TARGETS static_pointcloud_postprocessor_node
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(result_displayer_node src/result_displayer.cpp)
target_include_directories(result_displayer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(result_displayer_node ${PCL_LIBRARIES})
ament_target_dependencies(result_displayer_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

install(TARGETS result_displayer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})


# Install Python node
install(
  PROGRAMS src/tsdf_to_skeleton.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tsdf_to_skeleton_node
)

install(
  PROGRAMS src/static_pointcloud_saver.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cloud_saver
)

install(
  PROGRAMS src/ground_truth_path_saver.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ground_truth_path_saver
)

install(
  PROGRAMS src/static_tf_pub.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME static_tf_publisher
)

install(
  PROGRAMS src/test_flight.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_flight
)

ament_package()