#pragma once
namespace openpni::experimental::example {
void h_parallel_add(float const *in1, float const *in2, float *out, std::size_t size);
void h_parallel_add(float const *in, float *out, std::size_t size);
void d_parallel_add(float const *in1, float const *in2, float *out, std::size_t size);
void d_parallel_add(float const *in, float *out, std::size_t size);
void d_parallel_add(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parallel_multiply_add(float alpha, float const *in1, float const *in2, float *out, std::size_t size);
void d_parallel_multiply_add(float alpha, float const *in, float *out, std::size_t size);
} // namespace openpni::experimental::example

namespace openpni::experimental::example {
void h_parallel_sub(float const *in1, float const *in2, float *out, std::size_t size);
void h_parallel_sub(float const *in, float *out, std::size_t size);
void d_parallel_sub(float const *in1, float const *in2, float *out, std::size_t size);
void d_parallel_sub(float const *in, float value, float *out, std::size_t size);
void d_parallel_sub(float const *in, float *out, std::size_t size);

void h_parallel_sub(std::size_t const *in1, std::size_t const *in2, std::size_t *out, std::size_t size);
void h_parallel_sub(std::size_t const *in, std::size_t *out, std::size_t size);
void d_parallel_sub(std::size_t const *in1, std::size_t const *in2, std::size_t *out, std::size_t size);
void d_parallel_sub(std::size_t const *in, std::size_t value, std::size_t *out, std::size_t size);
void d_parallel_sub(std::size_t const *in, std::size_t *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_mul(float const *in1, float const *in2, float *out, std::size_t size);
void h_parallel_mul(float const *in, float *out, std::size_t size);
void d_parallel_mul(float const *in1, float const *in2, float *out, std::size_t size);
void d_parallel_mul(float const *in, float *out, std::size_t size);
void d_parallel_mul(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void d_parallel_div(float const *in1, float const *in2, float *out, std::size_t size);
void d_parallel_div(float const *in, float *out, std::size_t size);
void d_parallel_div(float const *in, float value, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_copy(float const *in, float *out, std::size_t size);
void d_parallel_copy(float const *in, float *out, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_parallel_fill(float *out, float value, std::size_t size);
void d_parallel_fill(float *out, float value, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
float h_sum_reduce(float const *in, std::size_t size);
float d_sum_reduce(float const *in, std::size_t size);
float h_average_reduce(float const *in, std::size_t size);
float d_average_reduce(float const *in, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
std::size_t h_count_equals(uint16_t const *in, uint16_t value, std::size_t size);
std::size_t d_count_equals(uint16_t const *in, uint16_t value, std::size_t size);
} // namespace openpni::experimental::example
namespace openpni::experimental::example {
void h_inclusive_sum(uint32_t const *in, uint32_t *out, std::size_t size);
void d_inclusive_sum(uint32_t const *in, uint32_t *out, std::size_t size);
void h_inclusive_sum_equals(uint16_t const *in, uint16_t value, uint32_t *out, std::size_t size);
void d_inclusive_sum_equals(uint16_t const *in, uint16_t value, uint32_t *out, std::size_t size);
void h_inclusive_sum_any_of(uint16_t const *in, uint16_t const *values, uint16_t valueCount, uint32_t *out,
                            std::size_t size);
void d_inclusive_sum_any_of(uint16_t const *in, uint16_t const *values, uint16_t valueCount, uint32_t *out,
                            std::size_t size);
} // namespace openpni::experimental::example