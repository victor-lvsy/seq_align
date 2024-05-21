# CMake minimum version required

cmake_minimum_required(VERSION 3.10)



# Project name

project(ADN_ALIGN)



# Add the executable

add_executable(ADN_ALIGN.o main.cpp nw.cpp)



# Include this directory for header files

target_include_directories(ADN_ALIGN.o PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})