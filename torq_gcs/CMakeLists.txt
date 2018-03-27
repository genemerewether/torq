cmake_minimum_required(VERSION 2.8.3)
project(torq_gcs)

find_package(catkin REQUIRED COMPONENTS rospy rviz)

catkin_package(CATKIN_DEPENDS message_runtime)
catkin_python_setup()

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
