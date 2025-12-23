cmake_minimum_required(VERSION 3.8)
project(webots_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install world and proto files
install(DIRECTORY
  worlds
  protos
  DESTINATION share/${PROJECT_NAME}
)

# Install controllers
install(DIRECTORY
  controllers
  DESTINATION share/${PROJECT_NAME}
)

# Install resource marker
install(FILES
  resource/webots_sim
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
