cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(MyProj VERSION 0.1.0 LANGUAGES CXX C)

find_package(CUDAToolkit REQUIRED)

enable_testing()
add_subdirectory(as_library)
add_subdirectory(as_header)
