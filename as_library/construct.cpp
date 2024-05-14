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
  if (m_init) {
    cudaFree(0);
  }
  if (!m_shared) {
    cublasDestroy(m_cublas_res);

    m_stream_view = cuda_stream_per_thread;
    auto stat = cublasCreate(&m_cublas_res);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf("CUBLAS re-initialization failed\n");
    }

    cublasSetStream(m_cublas_res, m_stream_view);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf("CUBLAS stream re-association failed\n");
    }
  }
  return m_cublas_res;
}

cudaStream_t cublas_resource::get_stream() { return m_stream_view.value(); }

cublas_resource make_handle(char mode) {
  cublas_resource h;

  if (mode == '1' || mode == '3') {
    h.m_init = true;
  }
  if (mode == '2' || mode == '3') {
    h.m_shared = true;
  }
  std::cout << "mode: " << mode << std::endl;
  std::cout << "handle shared: " << h.m_shared << std::endl;
  std::cout << "handle init: " << h.m_init << std::endl;
  return h;
}
