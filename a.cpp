#include "helper.h"

void compute(cublasHandle_t &h, std::size_t size, double *A, double *B,
             double &result);

void cublas_user_a(handle_cacher &h, double *A, double *B, std::size_t size) {
  std::cout << "calling cublas_user_a" << std::endl;
  double result = 0.0;

  /* step 3: compute */
  auto &r = h.get_resource();
  compute(r, size, A, B, result);
  CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
}
