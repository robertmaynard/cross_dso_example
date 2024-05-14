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
#include "helper_impl.h"

__attribute__ ((visibility ("default")))
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
