#include "helper.h"
#include "helper_impl.h"

void cublas_user_gemm(cublas_resource &h, double *A, double *B, double *C,
                      std::size_t size) {
  std::cout << "calling cublas_user_gemm" << std::endl;
  double result = 0.0;

  /* step 3: compute */
  auto &r = h.get_resource();
  constexpr int incx = 1;
  constexpr int incy = 1;
  double alpha = 1.0f;
  double beta = 0.0f;

  std::cout << "b-pre cublasDgemm" << std::endl;
  CUBLAS_CHECK(cublasDgemm(r, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                             &alpha, A, size, B, size, &beta, C, size));

  std::cout << "b-post cublasDgemm" << std::endl;
  CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
}
