#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "include/basic/CudaPtr.hpp"
#include "include/experimental/tools/Parallel.cuh"
#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::example {
void h_parralel_add(
    // out = in1 + in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] + in2[idx]; });
}
void h_parralel_add(
    // out+=in
    float const *in, float *out, std::size_t size) {
  h_parralel_add(in, out, out, size);
}
void d_parralel_add(
    // out = in1 + in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] + in2[idx]; });
}
void d_parralel_add(
    // out+=in
    float const *in, float *out, std::size_t size) {
  d_parralel_add(in, out, out, size);
}
void d_parralel_add(
    // out = in + value
    float const *in, float value, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] + value; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_sub(
    // out = in1 - in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] - in2[idx]; });
}
void h_parralel_sub(
    // out-=in
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] -= in[idx]; });
}
void d_parralel_sub(
    // out = in1 - in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] - in2[idx]; });
}
void d_parralel_sub(
    // out = in - value
    float const *in, float value, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] - value; });
}
void d_parralel_sub(
    // out-=in
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] -= in[idx]; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_mul(
    // out=in1*in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in1[idx] * in2[idx]; });
}
void h_parralel_mul(
    // out=in*value
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] *= in[idx]; });
}
void d_parralel_mul(
    // out=in1*in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] * in2[idx]; });
}
void d_parralel_mul(
    // out=in*value
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] *= in[idx]; });
}
void d_parralel_mul(
    // out=in*value
    float const *in, float value, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx] * value; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parralel_div(
    // out = in1 / in2
    float const *in1, float const *in2, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in1[idx] / in2[idx]; });
}
void d_parralel_div(
    // out/=in
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] /= in[idx]; });
}
void d_parralel_div(
    // out/=in
    float const *in, float value, float *out, std::size_t size) {
  d_parralel_mul(in, 1.0f / value, out, size);
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_copy(
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = in[idx]; });
}
void d_parralel_copy(
    float const *in, float *out, std::size_t size) {
  tools::parallel_for_each_CUDA(size, [=] __device__(std::size_t idx) { out[idx] = in[idx]; });
}
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_fill(
    float *out, float value, std::size_t size) {
  tools::parallel_for_each(size, [=](std::size_t idx) { out[idx] = value; });
}
void d_parralel_fill(
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
} // namespace openpni::experimental::example
