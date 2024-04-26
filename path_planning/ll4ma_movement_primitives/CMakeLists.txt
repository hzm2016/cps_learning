cmake_minimum_required(VERSION 2.8.3)
project(ll4ma_movement_primitives)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  actionlib
  actionlib_msgs
  geometry_msgs
  trajectory_msgs
  ll4ma_trajectory_util
  ll4ma_trajectory_msgs
)

catkin_python_setup()

add_message_files(DIRECTORY msg)
add_service_files(DIRECTORY srv)
add_action_files(DIRECTORY action)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
  trajectory_msgs
  ll4ma_trajectory_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  rospy
  std_msgs
  geometry_msgs
  trajectory_msgs
  ll4ma_trajectory_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
