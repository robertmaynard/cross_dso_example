/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "helper.h"
#include <vector>

cublas_resource make_handle();

void cublas_user_dot(cublas_resource &h, double* A, double* B, std::size_t size);
void cublas_user_gemm(cublas_resource &h, double* A, double* B, double* C, std::size_t size);

int main(int argc, char** argv) {
  using data_type = double;
  data_type* d_A = nullptr;
  data_type* d_B = nullptr;
  data_type* d_C = nullptr;
  const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
  const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
  const std::vector<data_type> C = {1.0, 1.0, 1.0, 1.0};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C),
                        sizeof(data_type) * C.size()));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(d_C, C.data(), sizeof(data_type) * C.size(),
                             cudaMemcpyHostToDevice));

  auto h = make_handle();
  cublas_user_dot(h, d_A, d_B, A.size());
  cublas_user_gemm(h, d_A, d_B, d_C, A.size());

  return 0;
}
