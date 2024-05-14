#include "helper.h"
#include "helper_impl.h"

void cublas_user_dot(cublas_resource &h, double *A, double *B, std::size_t size) {
  std::cout << "calling cublas_user_dot" << std::endl;
  double result = 0.0;

  /* step 3: compute */
  auto &r = h.get_resource();
  constexpr int incx = 1;
  constexpr int incy = 1;

  std::cout << "b-pre cublasDdot" << std::endl;
  CUBLAS_CHECK(cublasDdot(r, size, A, incx, B, incy, &result));
  std::cout << "b-post cublasDdot" << std::endl;
  CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
}
