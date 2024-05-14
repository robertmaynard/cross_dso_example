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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    auto status = static_cast<cudaError_t>(error);                             \
    if (status != cudaSuccess) {                                               \
      std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"        \
                << __LINE__ << std::endl;                                      \
      std::exit(status);                                                       \
    }                                                                          \
  }

#define CUBLAS_CHECK(error)                                                    \
  {                                                                            \
    auto status = static_cast<cublasStatus_t>(error);                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cout << cublasGetStatusString(status) << " " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
      std::exit(status);                                                       \
    }                                                                          \
  }


class cuda_stream_view {
public:
  constexpr cuda_stream_view() = default;
  ~cuda_stream_view() = default;
  constexpr cuda_stream_view(cuda_stream_view const &) =
      default; ///< @default_copy_constructor
  constexpr cuda_stream_view(cuda_stream_view &&) =
      default; ///< @default_move_constructor
  constexpr cuda_stream_view &operator=(cuda_stream_view const &) =
      default; ///< @default_copy_assignment{cuda_stream_view}
  constexpr cuda_stream_view &operator=(cuda_stream_view &&) =
      default; ///< @default_move_assignment{cuda_stream_view}

  // Disable construction from literal 0
  constexpr cuda_stream_view(int) = delete; //< Prevent cast from 0
  constexpr cuda_stream_view(std::nullptr_t) =
      delete; //< Prevent cast from nullptr

  constexpr cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

  [[nodiscard]] constexpr cudaStream_t value() const noexcept {
    return stream_;
  }
  constexpr operator cudaStream_t() const noexcept { return value(); }

  void synchronize() const { cudaStreamSynchronize(stream_); }

private:
  cudaStream_t stream_{};
};

static const cuda_stream_view cuda_stream_per_thread{cudaStreamPerThread};

struct cublas_resource {
  cublas_resource();
  ~cublas_resource();

  cublasHandle_t &get_resource();
  cudaStream_t get_stream();

private:
  cuda_stream_view m_stream_view;
  cublasHandle_t m_cublas_res;
};
