#pragma once
#include <cuda_runtime.h>
namespace openpni::experimental::example {
void h_parralel_add(float const *in1, float const *in2, float *out, std::size_t size);
void h_parralel_add(float const *in, float *out, std::size_t size);
void d_parralel_add(float const *in1, float const *in2, float *out, std::size_t size);
void d_parralel_add(float const *in, float *out, std::size_t size);
void d_parralel_add(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_sub(float const *in1, float const *in2, float *out, std::size_t size);
void h_parralel_sub(float const *in, float *out, std::size_t size);
void d_parralel_sub(float const *in1, float const *in2, float *out, std::size_t size);
void d_parralel_sub(float const *in, float value, float *out, std::size_t size);
void d_parralel_sub(float const *in, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_mul(float const *in1, float const *in2, float *out, std::size_t size);
void h_parralel_mul(float const *in, float *out, std::size_t size);
void d_parralel_mul(float const *in1, float const *in2, float *out, std::size_t size);
void d_parralel_mul(float const *in, float *out, std::size_t size);
void d_parralel_mul(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parralel_div(float const *in1, float const *in2, float *out, std::size_t size);
void d_parralel_div(float const *in, float *out, std::size_t size);
void d_parralel_div(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_copy(float const *in, float *out, std::size_t size);
void d_parralel_copy(float const *in, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parralel_fill(float *out, float value, std::size_t size);
void d_parralel_fill(float *out, float value, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
float h_sum_reduce(float const *in, std::size_t size);
float d_sum_reduce(float const *in, std::size_t size);
} // namespace openpni::experimental::example
