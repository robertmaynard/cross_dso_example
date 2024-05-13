

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)


project(MyProj VERSION 0.1.0 LANGUAGES CXX C)

find_package(CUDAToolkit REQUIRED)


function(add_build_setup name CUDART_MODE cudart_mode CUBLAS_MODE cublas_mode)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_library(${name}_compute SHARED compute.cpp)
  add_library(${name}_construct SHARED construct.cpp)
  add_executable(${name} main.cpp)

  target_compile_features(${name}_compute PUBLIC cxx_std_17)
  target_compile_features(${name}_construct PUBLIC cxx_std_17)

  # Order needs to be construct first, or via `PUBLIC` on `compute`
  target_link_libraries(${name}_compute PUBLIC ${name}_construct)

  target_link_libraries(${name} PRIVATE ${name}_compute)

  string(TOLOWER ${cublas_mode} cublas_mode)
  string(TOLOWER ${cudart_mode} cudart_mode)
  if(cublas_mode STREQUAL static)
    target_link_libraries(${name}_compute PRIVATE CUDA::cublas_static)
    target_link_libraries(${name}_construct PRIVATE CUDA::cublas_static)
  else()
    target_link_libraries(${name}_compute PRIVATE CUDA::cublas)
    target_link_libraries(${name}_construct PRIVATE CUDA::cublas)
  endif()

  if(cudart_mode STREQUAL static)
    target_link_libraries(${name}_compute PRIVATE CUDA::cudart_static)
    target_link_libraries(${name}_construct PRIVATE CUDA::cudart_static)
    target_link_libraries(${name} PRIVATE CUDA::cudart_static)
  else()
    target_link_libraries(${name}_compute PRIVATE CUDA::cudart)
    target_link_libraries(${name}_construct PRIVATE CUDA::cudart)
    target_link_libraries(${name} PRIVATE CUDA::cudart)
  endif()

endfunction()

function(add_test_setups name)
  add_test(NAME ${name}_separate_handles COMMAND ${name})
  add_test(NAME ${name}_shared_handles COMMAND ${name} 2)
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
