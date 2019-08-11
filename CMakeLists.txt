cmake_minimum_required(VERSION 2.8.12)
project(gym-line-follower)
# Set source directory
set(SOURCE_DIR_GEN "gym_line_follower/trackutils")
set(SOURCE_DIR_IR "gym_line_follower/irsensor")
set(SOURCE_DIR ${SOURCE_DIR_IR}
               ${SOURCE_DIR_GEN})
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR_GEN})
include_directories(${SOURCE_DIR_IR})
set(SOURCES_GEN
    "${SOURCE_DIR_GEN}/collision.cpp"
    "${SOURCE_DIR_GEN}/primitives.cpp")

set(SOURCES_IR "${SOURCE_DIR_IR}/irsensor.cpp")

# Generate Python module
add_subdirectory(lib/pybind11)
pybind11_add_module(trackutils ${SOURCES_GEN} "${SOURCE_DIR_GEN}/bindings.cpp")
pybind11_add_module(irsensor ${SOURCES_IR} "${SOURCE_DIR_IR}/bindings.cpp")

