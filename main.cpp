
#include "helper.h"
#include <vector>

handle_cacher make_handle(char);

void cublas_user_a(handle_cacher &h, double* A, double* B, std::size_t size);
void cublas_user_b(handle_cacher &h, double* A, double* B, std::size_t size);
void cublas_user_c(handle_cacher &h, double* A, double* B, std::size_t size);

int main(int argc, char** argv) {
  char mode = '0';
  if(argc > 1 ) {
    mode = argv[1][0];
  }

  using data_type = double;
  data_type* d_A = nullptr;
  data_type* d_B = nullptr;
  const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
  const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice));

  auto h = make_handle(mode);
  cublas_user_a(h, d_A, d_B, A.size());
  cublas_user_b(h, d_A, d_B, A.size());
  cublas_user_c(h, d_A, d_B, A.size());

  return 0;
}
