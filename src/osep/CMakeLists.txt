cmake_minimum_required(VERSION 3.10)
project(osep)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
