
#include <iostream>

cublas_resource::cublas_resource() {
  m_stream_view = cuda_stream_per_thread;
  auto stat = cublasCreate(&m_cublas_res);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
  }
  cublasSetStream(m_cublas_res, m_stream_view);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS stream association failed\n");
  }
}
cublas_resource::~cublas_resource() { cublasDestroy(m_cublas_res); }

cublasHandle_t &cublas_resource::get_resource() {
  return m_cublas_res;
}

cudaStream_t cublas_resource::get_stream() { return m_stream_view.value(); }

cublas_resource make_handle() {
  cublas_resource h;
  return h;
}
