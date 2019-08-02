cmake_minimum_required(VERSION 2.8.12)
project(gym-line-follower)
# Set source directory
set(SOURCE_DIR "gym_line_follower/genetic")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES "${SOURCE_DIR}/collision.cpp"
            "${SOURCE_DIR}/primitives.cpp")

# Generate Python module
add_subdirectory(lib/pybind11)
pybind11_add_module(trackcpp ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

