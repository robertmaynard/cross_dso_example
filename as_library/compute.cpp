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
