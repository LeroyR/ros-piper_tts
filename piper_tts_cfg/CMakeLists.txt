cmake_minimum_required(VERSION 3.5)
project(piper_tts_cfg)

find_package(catkin REQUIRED 
  COMPONENTS
  dynamic_reconfigure
  )

## Gencfg
generate_dynamic_reconfigure_options(
  cfg/TTS.cfg
)

catkin_package()

