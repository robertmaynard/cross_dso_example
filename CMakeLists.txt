

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)


project(MyProj VERSION 0.1.0 LANGUAGES CXX C)

find_package(CUDAToolkit REQUIRED)

# make sure everything has the CUDA toolkit include paths
link_libraries(CUDA::toolkit)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(a_objs OBJECT a.cpp)
add_library(b_objs OBJECT b.cpp)
add_library(c_objs OBJECT c.cpp)
add_library(main_objs OBJECT main.cpp)

target_compile_features(a_objs PRIVATE cxx_std_17)
target_compile_features(b_objs PRIVATE cxx_std_17)
target_compile_features(c_objs PRIVATE cxx_std_17)
target_compile_features(main_objs PRIVATE cxx_std_17)

function(add_build_setup name CUDART_MODE cudart_mode CUBLAS_MODE cublas_mode)
  add_library(a_${name} SHARED $<TARGET_OBJECTS:a_objs>)
  add_library(b_${name} SHARED $<TARGET_OBJECTS:b_objs>)
  add_library(c_${name} SHARED $<TARGET_OBJECTS:c_objs>)
  add_executable(${name} $<TARGET_OBJECTS:main_objs>)

  target_link_libraries(${name} PRIVATE c_${name} b_${name} a_${name})

  string(TOLOWER ${cublas_mode} cublas_mode)
  string(TOLOWER ${cudart_mode} cudart_mode)
  message(STATUS "a_${name} ${cublas_mode} ${cudart_mode}")
  if(cublas_mode STREQUAL static)
    target_link_libraries(a_${name} PRIVATE CUDA::cublas_static)
    target_link_libraries(b_${name} PRIVATE CUDA::cublas_static)
    target_link_libraries(c_${name} PRIVATE CUDA::cublas_static)
  else()
    target_link_libraries(a_${name} PRIVATE CUDA::cublas)
    target_link_libraries(b_${name} PRIVATE CUDA::cublas)
    target_link_libraries(c_${name} PRIVATE CUDA::cublas)
  endif()

  if(cudart_mode STREQUAL static)
    target_link_libraries(a_${name} PRIVATE CUDA::cudart_static)
    target_link_libraries(b_${name} PRIVATE CUDA::cudart_static)
    target_link_libraries(c_${name} PRIVATE CUDA::cudart_static)
    target_link_libraries(${name} PRIVATE CUDA::cudart_static)
  else()
    target_link_libraries(a_${name} PRIVATE CUDA::cudart)
    target_link_libraries(b_${name} PRIVATE CUDA::cudart)
    target_link_libraries(c_${name} PRIVATE CUDA::cudart)
    target_link_libraries(${name} PRIVATE CUDA::cudart)
  endif()

endfunction()

function(add_test_setups name)
  add_test(NAME ${name}_separate_handles COMMAND ${name})

  add_test(NAME ${name}_init_cuda_first COMMAND ${name} 1)

  add_test(NAME ${name}_shared_handles COMMAND ${name} 2)

  add_test(NAME ${name}_shared_handles_with_init COMMAND ${name} 3)

endfunction()


add_build_setup(STATIC_STATIC CUDART_MODE STATIC CUBLAS_MODE STATIC)
add_build_setup(STATIC_SHARED CUDART_MODE STATIC CUBLAS_MODE SHARED)
add_build_setup(SHARED_STATIC CUDART_MODE SHARED CUBLAS_MODE STATIC)
add_build_setup(SHARED_SHARED CUDART_MODE SHARED CUBLAS_MODE SHARED)

enable_testing()
add_test_setups(STATIC_STATIC)
add_test_setups(STATIC_SHARED)
add_test_setups(SHARED_STATIC)
add_test_setups(SHARED_SHARED)
