cmake_minimum_required(VERSION 2.8.3)
project(px4_msgs)

find_package(catkin REQUIRED COMPONENTS roscpp message_generation std_msgs)
# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

add_message_files(
  FILES
  PolyTraj.msg
)

#uncomment if you have defined services
#add_service_files(
#  FILES
  # TODO: List your msg files here
#)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS message_runtime std_msgs
    INCLUDE_DIRS
    LIBRARIES
)

include_directories(include)
