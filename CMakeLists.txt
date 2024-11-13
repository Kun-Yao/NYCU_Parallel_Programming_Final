# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(PP_Final)

# Set C++ standard to C++17 or newer
set(CMAKE_CXX_STANDARD 17)

# Add the include directory for stb_image.h
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files to a variable
set(SOURCES
    src/main.cpp
)

# Add executable
add_executable(my_program ${SOURCES})
