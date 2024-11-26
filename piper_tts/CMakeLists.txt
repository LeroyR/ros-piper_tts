cmake_minimum_required(VERSION 2.8.12)
project(piper_tts)

find_package(catkin REQUIRED COMPONENTS
  rospy    
  dynamic_reconfigure
  )

catkin_python_setup()

catkin_package()

install(PROGRAMS
 scripts/direct_piper_tts_pal
 scripts/forward_piper_tts_pal
 DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

