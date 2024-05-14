
function(add_build_setup name CUDART_MODE cudart_mode CUBLAS_MODE cublas_mode)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_library(${name}_compute_Dot SHARED computeDot.cpp)
  add_library(${name}_compute_Gemm SHARED computeGemm.cpp)
  add_executable(${name} main.cpp)

  target_compile_features(${name}_compute_Dot PUBLIC cxx_std_17)
  target_compile_features(${name}_compute_Gemm PUBLIC cxx_std_17)

  target_link_libraries(${name} PRIVATE ${name}_compute_Gemm ${name}_compute_Dot)

  string(TOLOWER ${cublas_mode} cublas_mode)
  string(TOLOWER ${cudart_mode} cudart_mode)
  if(cublas_mode STREQUAL static)
    target_link_libraries(${name}_compute_Dot PRIVATE CUDA::cublas_static)
    target_link_libraries(${name}_compute_Gemm PRIVATE CUDA::cublas_static)
  else()
    target_link_libraries(${name}_compute_Dot PRIVATE CUDA::cublas)
    target_link_libraries(${name}_compute_Gemm PRIVATE CUDA::cublas)
  endif()

  if(cudart_mode STREQUAL static)
    target_link_libraries(${name}_compute_Dot PRIVATE CUDA::cudart_static)
    target_link_libraries(${name}_compute_Gemm PRIVATE CUDA::cudart_static)
    target_link_libraries(${name} PRIVATE CUDA::cudart_static)
  else()
    target_link_libraries(${name}_compute_Dot PRIVATE CUDA::cudart)
    target_link_libraries(${name}_compute_Gemm PRIVATE CUDA::cudart)
    target_link_libraries(${name} PRIVATE CUDA::cudart)
  endif()

endfunction()

function(add_test_setups name)
  add_test(NAME ${name} COMMAND ${name})
endfunction()


add_build_setup(HEADER_STATIC_STATIC CUDART_MODE STATIC CUBLAS_MODE STATIC)
add_build_setup(HEADER_STATIC_SHARED CUDART_MODE STATIC CUBLAS_MODE SHARED)
add_build_setup(HEADER_SHARED_STATIC CUDART_MODE SHARED CUBLAS_MODE STATIC)
add_build_setup(HEADER_SHARED_SHARED CUDART_MODE SHARED CUBLAS_MODE SHARED)

enable_testing()
add_test_setups(HEADER_STATIC_STATIC)
add_test_setups(HEADER_STATIC_SHARED)
add_test_setups(HEADER_SHARED_STATIC)
add_test_setups(HEADER_SHARED_SHARED)
