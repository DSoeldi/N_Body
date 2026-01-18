cmake_minimum_required(VERSION 4.0)
project(N_Body_cpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(N_Body_cpp main.cpp)

target_include_directories(N_Body_cpp PRIVATE ${CMAKE_SOURCE_DIR}/lib)
## include directories for external libraries
#include_directories(${CMAKE_SOURCE_DIR}/lib)
#
## Link external library
#target_link_libraries(N_Body_cpp PRIVATE ${CMAKE_SOURCE_DIR}/lib/eigen-5.0.0)