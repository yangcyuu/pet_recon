#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "Test.h"
#include "include/basic/CudaPtr.hpp"
void d_print_max_value(
    float const *d_data, std::size_t size) {
  if (size == 0)
    return;
  float max_value = thrust::reduce(thrust::cuda::par.on(openpni::basic::cuda_ptr::default_stream()),
                                   thrust::device_pointer_cast(d_data), thrust::device_pointer_cast(d_data + size),
                                   0.0f, thrust::maximum<float>());
  printf("max value: %f\n", max_value);
}
void d_print_none_zero_average_value(
    float const *d_data, std::size_t size) {
  if (size == 0)
    return;
  float sum_value = thrust::reduce(thrust::cuda::par.on(openpni::basic::cuda_ptr::default_stream()),
                                   thrust::device_pointer_cast(d_data), thrust::device_pointer_cast(d_data + size),
                                   0.0f, thrust::plus<float>());
  std::size_t count = thrust::transform_reduce(
      thrust::cuda::par.on(openpni::basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(d_data),
      thrust::device_pointer_cast(d_data + size), [] __device__(float x) -> std::size_t { return x != 0.0f ? 1 : 0; },
      0ULL, thrust::plus<std::size_t>());
  float min_value = sum_value / static_cast<float>(count);
  printf("non-zero average value: %f\n", min_value);
}