#include "helper.h"

void compute(cublasHandle_t &h, std::size_t size, double *A, double *B,
             double &result) {
  constexpr int incx = 1;
  constexpr int incy = 1;

  std::cout << "b-pre cublasDdot" << std::endl;
  CUBLAS_CHECK(cublasDdot(h, size, A, incx, B, incy, &result));
  std::cout << "b-post cublasDdot" << std::endl;
}

void cublas_user(cublas_resource &h, double *A, double *B, std::size_t size) {
  std::cout << "calling cublas_user_b" << std::endl;
  double result = 0.0;

  /* step 3: compute */
  auto &r = h.get_resource();
  compute(r, size, A, B, result);
  CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
}
