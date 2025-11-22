#include <algorithm>
#include <numeric>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_scan.h>

#include "include/basic/CudaPtr.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/tools/Parallel.cuh"
#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::example {
void h_parallel_add(
    // out = in1 + in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] + in2[idx]; });
}
void h_parallel_add(
    // out+=in
    float const *in, float *out, std::size_t size) {
  h_parallel_add(in, out, out, size);
}
void d_parallel_add(
    // out = in1 + in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] + in2[idx]; });
}
void d_parallel_add(
    // out+=in
    float const *in, float *out, std::size_t size) {
  d_parallel_add(in, out, out, size);
}
void d_parallel_add(
    // out = in + value
    float const *in, float value, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] + value; });
}

} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parallel_multiply_add(
    // out = alpha * in1 + in2
    float alpha, float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = alpha * in1[idx] + in2[idx]; });
}
void d_parallel_multiply_add(
    //  out+=alpha*in
    float alpha, float const *in, float *out, std::size_t size) {
  d_parallel_multiply_add(alpha, in, out, out, size);
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
template <typename T>
void _h_parallel_sub(
    // out = in1 - in2
    T const *in1, T const *in2, T *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] - in2[idx]; });
}
template <typename T>
void _h_parallel_sub(
    // out-=in
    T const *in, T *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] -= in[idx]; });
}
template <typename T>
void _d_parallel_sub(
    // out = in1 - in2
    T const *in1, T const *in2, T *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] - in2[idx]; });
}
template <typename T>
void _d_parallel_sub(
    // out = in - value
    T const *in, T value, T *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] - value; });
}
template <typename T>
void _d_parallel_sub(
    // out-=in
    T const *in, T *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] -= in[idx]; });
}

void h_parallel_sub(
    // out = in1 - in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  _h_parallel_sub(in1, in2, out, size);
}
void h_parallel_sub(
    // out-=in
    float const *in, float *out, std::size_t size) {}
void d_parallel_sub(
    // out = in1 - in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  _d_parallel_sub(in1, in2, out, size);
}
void d_parallel_sub(
    // out = in - value
    float const *in, float value, float *out, std::size_t size) {
  _d_parallel_sub(in, value, out, size);
}
void d_parallel_sub(
    // out-=in
    float const *in, float *out, std::size_t size) {
  _d_parallel_sub(in, out, size);
}

void h_parallel_sub(
    std::size_t const *in1, std::size_t const *in2, std::size_t *out, std::size_t size) {
  _h_parallel_sub(in1, in2, out, size);
}
void h_parallel_sub(
    std::size_t const *in, std::size_t *out, std::size_t size) {
  _h_parallel_sub(in, out, size);
}
void d_parallel_sub(
    std::size_t const *in1, std::size_t const *in2, std::size_t *out, std::size_t size) {
  _d_parallel_sub(in1, in2, out, size);
}
void d_parallel_sub(
    std::size_t const *in, std::size_t value, std::size_t *out, std::size_t size) {
  _d_parallel_sub(in, value, out, size);
}
void d_parallel_sub(
    std::size_t const *in, std::size_t *out, std::size_t size) {
  _d_parallel_sub(in, out, size);
}

} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_mul(
    // out=in1*in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] * in2[idx]; });
}
void h_parallel_mul(
    // out=in*value
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] *= in[idx]; });
}
void d_parallel_mul(
    // out=in1*in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] * in2[idx]; });
}
void d_parallel_mul(
    // out=in*value
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] *= in[idx]; });
}
void d_parallel_mul(
    // out=in*value
    float const *in, float value, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] * value; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parallel_div(
    // out = in1 / in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] / in2[idx]; });
}
void d_parallel_div(
    // out/=in
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] /= in[idx]; });
}
void d_parallel_div(
    // out/=in
    float const *in, float value, float *out, std::size_t size) {
  d_parallel_mul(in, 1.0f / value, out, size);
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_copy(
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in[idx]; });
}
void d_parallel_copy(
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx]; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_fill(
    float *out, float value, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = value; });
}
void d_parallel_fill(
    float *out, float value, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = value; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
float h_sum_reduce(
    float const *in, std::size_t size) {
  if (size == 0)
    return 0.0f;
  float sum = 0.0f;
  std::for_each(in, in + size, [&sum](float v) { sum += v; });
  return sum;
}
float d_sum_reduce(
    float const *in, std::size_t size) {
  if (size == 0)
    return 0.0f;
  return thrust::reduce(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(in),
                        thrust::device_pointer_cast(in + size), 0.0f, thrust::plus<float>());
}
float h_average_reduce(
    float const *in, std::size_t size) {
  if (size == 0)
    return 0.0f;
  return h_sum_reduce(in, size) / static_cast<float>(size);
}
float d_average_reduce(
    float const *in, std::size_t size) {
  if (size == 0)
    return 0.0f;
  return d_sum_reduce(in, size) / static_cast<float>(size);
}

} // namespace openpni::experimental::example

namespace openpni::experimental::example {
std::size_t h_count_equals(
    uint16_t const *in, uint16_t value, std::size_t size) {
  if (size == 0)
    return 0;
  return static_cast<std::size_t>(std::count(in, in + size, value));
}
std::size_t d_count_equals(
    uint16_t const *in, uint16_t value, std::size_t size) {
  if (size == 0)
    return 0;
  return static_cast<std::size_t>(thrust::count(thrust::cuda::par.on(basic::cuda_ptr::default_stream()),
                                                thrust::device_pointer_cast(in), thrust::device_pointer_cast(in + size),
                                                value));
}
} // namespace openpni::experimental::example

namespace openpni::experimental::example {
void h_inclusive_sum(
    uint32_t const *in, uint32_t *out, std::size_t size) {
  if (size == 0)
    return;
  std::inclusive_scan(in, in + size, out);
}
void d_inclusive_sum(
    uint32_t const *in, uint32_t *out, std::size_t size) { // out长度需为size+1
  if (size == 0)
    return;
  thrust::inclusive_scan(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(in),
                         thrust::device_pointer_cast(in + size), thrust::device_pointer_cast(out));
}
void h_inclusive_sum_equals(
    uint16_t const *in, uint16_t value, uint32_t *out, std::size_t size) {
  if (size == 0)
    return;
  std::transform_inclusive_scan(in, in + size, out, std::plus<uint32_t>(),
                                [=](uint16_t b) { return b == value ? 1 : 0; });
}
void d_inclusive_sum_equals(
    uint16_t const *in, uint16_t value, uint32_t *out, std::size_t size) {
  if (size == 0)
    return;
  thrust::transform_inclusive_scan(
      thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
      thrust::counting_iterator<std::size_t>(size), thrust::device_pointer_cast(out),
      [=] __host__ __device__(std::size_t idnex) -> uint32_t {
        return in[idnex] == value ? static_cast<uint32_t>(1u) : static_cast<uint32_t>(0u);
      },
      thrust::plus<uint32_t>());
}
void h_inclusive_sum_any_of(
    uint16_t const *in, uint16_t const *values, uint16_t valueCount, uint32_t *out, std::size_t size) {
  if (size == 0)
    return;
  std::transform_inclusive_scan(in, in + size, out, std::plus<uint32_t>(), [=](uint16_t b) {
    return std::any_of(values, values + valueCount, [=](uint16_t v) { return b == v; }) ? 1 : 0;
  });
}
void d_inclusive_sum_any_of(
    uint16_t const *in, uint16_t const *values, uint16_t valueCount, uint32_t *out, std::size_t size) {
  if (size == 0)
    return;

  thrust::transform_inclusive_scan(
      thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
      thrust::counting_iterator<std::size_t>(size), thrust::device_pointer_cast(out),
      [=] __host__ __device__(std::size_t index) -> uint32_t {
        uint16_t b = in[index];
        for (uint16_t i = 0; i < valueCount; i++)
          if (b == values[i])
            return 1u;
        return 0u;
      },
      thrust::plus<uint32_t>());
}
} // namespace openpni::experimental::example
