# CMake minimum version required
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name
project(ADN_ALIGN LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set debugging flags (only if you are in Debug mode)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# Define the source and include directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Add the include directory to the include paths
include_directories(${INCLUDE_DIR})

# Main source file at the root
set(MAIN_FILE "${CMAKE_SOURCE_DIR}/main.cpp")

# Find all source files (both .cpp and .cu)
file(GLOB_RECURSE SOURCES_CPP "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE SOURCES_CU "${SRC_DIR}/*.cu")

# Create the executable
add_executable(adn.o
    ${MAIN_FILE}
    ${SOURCES_CPP}
    ${SOURCES_CU} # Add the specific kernel file here
)

# Set CUDA properties, you can set architecture or other cuda_flags here
# Targeting Ada Lovelace (compute capability 89)
set_target_properties(adn.o PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80"
)
