#pragma once
#include <cmath>
#include <cuda_runtime_api.h>
#include <type_traits>
namespace openpni {
template <typename T>
concept FloatingPoint_c = std::is_floating_point_v<T>;
template <typename T>
concept Integral_c = std::is_integral_v<T>;
template <typename T>
concept Arithmetic_c = std::is_arithmetic_v<T>;
} // namespace openpni
namespace openpni::experimental::core {
template <typename T>
__PNI_CUDA_MACRO__ inline constexpr T min(
    T a, T b) {
  return a < b ? a : b;
}
template <typename T>
__PNI_CUDA_MACRO__ inline constexpr T max(
    T a, T b) {
  return a > b ? a : b;
}
template <typename T>
struct FMath {};
template <>
struct FMath<int> {
  __PNI_CUDA_MACRO__ static constexpr int abs(
      int x) {
    return x < 0 ? -x : x;
  }
  __PNI_CUDA_MACRO__ static constexpr int min(
      int a, int b) {
    return a < b ? a : b;
  }
  __PNI_CUDA_MACRO__ static constexpr int max(
      int a, int b) {
    return a > b ? a : b;
  }
};
template <>
struct FMath<float> {
  __PNI_CUDA_MACRO__ static constexpr float abs(
      float x) {
    return fabsf(x);
  }
  __PNI_CUDA_MACRO__ static constexpr float min(
      float a, float b) {
    return fminf(a, b);
  }
  __PNI_CUDA_MACRO__ static constexpr float max(
      float a, float b) {
    return fmaxf(a, b);
  }
  __PNI_CUDA_MACRO__ static constexpr float fsqrt(
      float x) {
    return sqrt(x);
  }
  __PNI_CUDA_MACRO__ static constexpr float rfsqrt(
      float x) {
#ifdef __CUDA_ARCH__
    return rsqrtf(x);
#else
    return 1.f / fsqrt(x);
#endif
  }
  __PNI_CUDA_MACRO__ static constexpr float fpow(
      float base, float exponent) {
    return powf(base, exponent);
  }
  __PNI_CUDA_MACRO__ static constexpr float fexp(
      float x) {
    return expf(x);
  }
  __PNI_CUDA_MACRO__ static constexpr float calStandardGaussian(
      float x, float mean, float gaussianStandardDeviation) {
    return (x - mean) / gaussianStandardDeviation;
  };
  __PNI_CUDA_MACRO__ static constexpr float gauss_integral(
      float x1, float x2, float mean,
      float gaussianStandardDeviation) // for any guassian with mean and deviation
  {
    constexpr float sqrt2 = 1.41421356237f;
    x1 = calStandardGaussian(x1, mean, gaussianStandardDeviation);
    x2 = calStandardGaussian(x2, mean, gaussianStandardDeviation);
    return (erff(x2 / sqrt2) - erff(x1 / sqrt2)) * 0.5f;
  }
  __PNI_CUDA_MACRO__ static constexpr float flog(
      float x) {
    return log(x);
  }
  __PNI_CUDA_MACRO__ static constexpr bool isNaN(
      float x) {
    return x != x;
  }
  __PNI_CUDA_MACRO__ static constexpr bool isInf(
      float x) {
#ifdef __CUDA_ARCH__
    return isinf(x);
#else
    return std::isinf(x);
#endif
  }
  __PNI_CUDA_MACRO__ static constexpr bool isBad(
      float x) {
    return isNaN(x) || isInf(x);
  }
  __PNI_CUDA_MACRO__ static constexpr float ffloor(
      float x) {
    return floor(x);
  }
  __PNI_CUDA_MACRO__ static constexpr float fceil(
      float x) {
    return ceil(x);
  }
};
template <>
struct FMath<double> {
  __PNI_CUDA_MACRO__ static constexpr double abs(
      double x) {
    return fabs(x);
  }
  __PNI_CUDA_MACRO__ static constexpr double min(
      double a, double b) {
    return fmin(a, b);
  }
  __PNI_CUDA_MACRO__ static constexpr double max(
      double a, double b) {
    return fmax(a, b);
  }
  __PNI_CUDA_MACRO__ static constexpr double fsqrt(
      double x) {
    return sqrt(x);
  }
  __PNI_CUDA_MACRO__ static constexpr double rfsqrt(
      double x) {
#ifdef __CUDA_ARCH__
    return rsqrt(x);
#else
    return 1.0 / fsqrt(x);
#endif
  }
  __PNI_CUDA_MACRO__ static constexpr double fpow(
      double base, double exponent) {
    return pow(base, exponent);
  }
  __PNI_CUDA_MACRO__ static constexpr double fexp(
      double x) {
    return expf(x);
  }
  __PNI_CUDA_MACRO__ static constexpr double calStandardGaussian(
      double x, double mean, double gaussianStandardDeviation) {
    return (x - mean) / gaussianStandardDeviation;
  };
  __PNI_CUDA_MACRO__ static constexpr double gauss_integral(
      double x1, double x2, double mean,
      double gaussianStandardDeviation) // for any guassian with mean and deviation
  {
    constexpr double sqrt2 = 1.4142135623730951;
    x1 = calStandardGaussian(x1, mean, gaussianStandardDeviation);
    x2 = calStandardGaussian(x2, mean, gaussianStandardDeviation);
    return (erff(x2 / sqrt2) - erff(x1 / sqrt2)) * 0.5f;
  }
  __PNI_CUDA_MACRO__ static constexpr double flog(
      double x) {
    return log(x);
  }
  __PNI_CUDA_MACRO__ static constexpr bool isNaN(
      double x) {
    return x != x;
  }
  __PNI_CUDA_MACRO__ static constexpr bool isInf(
      double x) {
#ifdef __CUDA_ARCH__
    return isinf(x);
#else
    return std::isinf(x);
#endif
  }
  __PNI_CUDA_MACRO__ static constexpr bool isBad(
      double x) {
    return isNaN(x) || isInf(x);
  }
  __PNI_CUDA_MACRO__ static constexpr double ffloor(
      double x) {
    return floor(x);
  }
  __PNI_CUDA_MACRO__ static constexpr double fceil(
      double x) {
    return ceil(x);
  }
};

template <Integral_c T>
__PNI_CUDA_MACRO__ inline T dev_ceil(
    T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T sigmoid(
    T x) {
  return T(1) / (T(1) + FMath<T>::fexp(-x));
}

template <typename T, typename U = T>
__PNI_CUDA_MACRO__ inline T value_or(
    const T *__ptr, U __default) {
  return __ptr ? *__ptr : __default;
}
} // namespace openpni::experimental::core