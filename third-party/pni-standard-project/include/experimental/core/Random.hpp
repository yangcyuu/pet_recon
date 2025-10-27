#pragma once
#include <cstdint>
namespace openpni::experimental::core {
struct Xorshift128 {
  __PNI_CUDA_MACRO__ Xorshift128(
      uint32_t seed)
      : x(seed)
      , y(0xf)
      , z(0xf00)
      , w(0xf0000) {}
  uint32_t operator()() {
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = w ^ (w >> 19) ^ t ^ (t >> 8);
  }

private:
  uint32_t x, y, z, w;
};
__PNI_CUDA_MACRO__
inline float instant_random_float(
    std::size_t seed) {
  return (seed * 2654435761U) % 10000U / 10000.0f;
}
} // namespace openpni::experimental::core
