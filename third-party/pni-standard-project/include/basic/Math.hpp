#pragma once
#include <cmath>
#include <span>
#include <type_traits>
#include <vector>

#include "../experimental/core/BasicMath.hpp"
// #include "Image.hpp"
// #include "Vector.hpp"
// openpni
namespace openpni::basic {
template <typename T>
__PNI_CUDA_MACRO__ inline T min(
    T a, T b) {
  return a < b ? a : b;
}
template <typename T>
__PNI_CUDA_MACRO__ inline T max(
    T a, T b) {
  return a > b ? a : b;
}
template <typename T>
__PNI_CUDA_MACRO__ inline T abs(
    T x) {
  return x < 0 ? -x : x;
}

template <typename T>
struct FMath {};
template <>
struct FMath<int> {
  __PNI_CUDA_MACRO__ static int abs(
      int x) {
    return x < 0 ? -x : x;
  }
  __PNI_CUDA_MACRO__ static int min(
      int a, int b) {
    return a < b ? a : b;
  }
  __PNI_CUDA_MACRO__ static int max(
      int a, int b) {
    return a > b ? a : b;
  }
};
template <>
struct FMath<float> {
  __PNI_CUDA_MACRO__ static float abs(
      float x) {
    return fabsf(x);
  }
  __PNI_CUDA_MACRO__ static float min(
      float a, float b) {
    return fminf(a, b);
  }
  __PNI_CUDA_MACRO__ static float max(
      float a, float b) {
    return fmaxf(a, b);
  }
  __PNI_CUDA_MACRO__ static float fsqrt(
      float x) {
    return sqrt(x);
  }
  __PNI_CUDA_MACRO__ static float fpow(
      float base, float exponent) {
    return powf(base, exponent);
  }
  __PNI_CUDA_MACRO__ static float fexp(
      float x) {
    return expf(x);
  }
  __PNI_CUDA_MACRO__ static float calStandardGaussian(
      float x, float mean, float gaussianStandardDeviation) {
    return (x - mean) / gaussianStandardDeviation;
  };
  __PNI_CUDA_MACRO__ static float gauss_integral(
      float x1, float x2, float mean,
      float gaussianStandardDeviation) // for any guassian with mean and deviation
  {
    constexpr float sqrt2 = 1.41421356237f;
    x1 = calStandardGaussian(x1, mean, gaussianStandardDeviation);
    x2 = calStandardGaussian(x2, mean, gaussianStandardDeviation);
    return (erff(x2 / sqrt2) - erff(x1 / sqrt2)) * 0.5f;
  }
  __PNI_CUDA_MACRO__ static float flog(
      float x) {
    return log(x);
  }
  __PNI_CUDA_MACRO__ static bool isNaN(
      float x) {
    return std::isnan(x);
  }
  __PNI_CUDA_MACRO__ static bool isInf(
      float x) {
    return std::isinf(x);
  }
};
template <>
struct FMath<double> {
  __PNI_CUDA_MACRO__ static double abs(
      double x) {
    return fabs(x);
  }
  __PNI_CUDA_MACRO__ static double min(
      double a, double b) {
    return fmin(a, b);
  }
  __PNI_CUDA_MACRO__ static double max(
      double a, double b) {
    return fmax(a, b);
  }
  __PNI_CUDA_MACRO__ static double fsqrt(
      double x) {
    return sqrt(x);
  }
  __PNI_CUDA_MACRO__ static double fpow(
      double base, double exponent) {
    return pow(base, exponent);
  }
  __PNI_CUDA_MACRO__ static double fexp(
      double x) {
    return expf(x);
  }
  __PNI_CUDA_MACRO__ static double calStandardGaussian(
      double x, double mean, double gaussianStandardDeviation) {
    return (x - mean) / gaussianStandardDeviation;
  };
  __PNI_CUDA_MACRO__ static double gauss_integral(
      double x1, double x2, double mean,
      double gaussianStandardDeviation) // for any guassian with mean and deviation
  {
    constexpr double sqrt2 = 1.4142135623730951;
    x1 = calStandardGaussian(x1, mean, gaussianStandardDeviation);
    x2 = calStandardGaussian(x2, mean, gaussianStandardDeviation);
    return (erff(x2 / sqrt2) - erff(x1 / sqrt2)) * 0.5f;
  }
  __PNI_CUDA_MACRO__ static double flog(
      double x) {
    return log(x);
  }
  __PNI_CUDA_MACRO__ static bool isNaN(
      double x) {
    return std::isnan(x);
  }
  __PNI_CUDA_MACRO__ static bool isInf(
      double x) {
    return std::isinf(x);
  }
};
} // namespace
  // openpni::basic

namespace openpni::basic {

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T smallFloat(
    T a) {
  if (FMath<T>::abs(a) < T(1e-8))
    return (a >= 0 ? 1. : -1.) * T(1e-8);
  return a;
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T sigmoid(
    T x) {
  if (x > 15)
    return T(1);
  else if (x < -15)
    return T(0);
  return T(1) / (T(1) + FMath<T>::fexp(-x));
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T pos_sigmoid(
    T x) {
  const auto value = 2 * sigmoid(x) - T(1);
  return value > T(0) ? value : T(0);
}

__PNI_CUDA_MACRO__ inline int sgn(
    bool b) {
  return b ? 1 : -1;
}

template <typename T>
__PNI_CUDA_MACRO__ inline int negative(
    T t) {
  return t < 0 ? -1 : 0;
}
template <typename T, typename U = T>
__PNI_CUDA_MACRO__ inline T value_or(
    const T *__ptr, U __default) {
  return __ptr ? *__ptr : __default;
}

struct _LinearFitting {
  template <FloatingPoint_c CalculationPresicion>
  struct _WithBias {
    using C = CalculationPresicion;
    int n = 0;
    C sumX = C(0);  // sum(x)
    C sumY = C(0);  // sum(y)
    C sumXY = C(0); // sum(x*y)
    C sumXX = C(0); // sum(x*x)

    __PNI_CUDA_MACRO__ void add(
        C __x, C __y) {
      sumX += __x;
      sumY += __y;
      sumXY += __x * __y;
      sumXX += __x * __x;
      n++;
    }
    __PNI_CUDA_MACRO__ C slope() { return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX); }
    __PNI_CUDA_MACRO__ C intercept() { return (sumXX * sumY - sumX * sumXY) / (n * sumXX - sumX * sumX); }
    __PNI_CUDA_MACRO__ C predict(
        C __x) {
      return slope() * __x + intercept();
    }
  };
  template <FloatingPoint_c CalculationPresicion>
  auto withBias() const {
    return _WithBias<CalculationPresicion>{};
  }
  template <FloatingPoint_c CalculationPresicion>
  struct _WithoutBias {
    using C = CalculationPresicion;

    C sumXY = C(0); // sum(x*y)
    C sumXX = C(0); // sum(x*x)

    __PNI_CUDA_MACRO__ void add(
        C __x, C __y) {
      sumXY += __x * __y;
      sumXX += __x * __x;
    }
    __PNI_CUDA_MACRO__ C slope() { return fabs(sumXY / sumXX); }
    __PNI_CUDA_MACRO__ C intercept() { return 0; }
    __PNI_CUDA_MACRO__ C predict(
        C __x) {
      return slope() * __x;
    }
  };
  template <FloatingPoint_c CalculationPresicion>
  auto withoutBias() const {
    return _WithoutBias<CalculationPresicion>{};
  }
  template <FloatingPoint_c CalculationPresicion>
  struct _Result {
    CalculationPresicion slope;
    CalculationPresicion intercept;
  };
  template <FloatingPoint_c CalculationPresicion>
  __PNI_CUDA_MACRO__ _Result<CalculationPresicion> instant(
      std::span<const CalculationPresicion> x, std::span<const CalculationPresicion> y) const {
    _WithBias<CalculationPresicion> fitter;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i)
      fitter.add(x[i], y[i]);
    return {fitter.slope(), fitter.intercept()};
  }
};
inline constexpr _LinearFitting LinearFitting{};

template <Integral_c T>
__PNI_CUDA_MACRO__ inline unsigned isPowerOf(
    T n) {
  if (n <= 0)
    return 0;
  unsigned exponent = 0;
  while ((T(1) << exponent) < n) {
    ++exponent;
  }
  return exponent;
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T HWHM2Sigma(
    T __hwhm) {
  constexpr T sqrt2ln2 = 1.1774100225; // std::sqrt(2 * std::numbers::ln2_v<T>);
  return __hwhm / sqrt2ln2;
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T linearInterpolation(
    T x, T x1, T y1, T x2, T y2) {
  if (x1 == x2)
    return y1;
  return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

template <Integral_c T>
__PNI_CUDA_MACRO__ inline T dev_ceil(
    T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T, typename Compare>
__PNI_CUDA_MACRO__ inline void bubble_sort(
    T *__begin, T *__end, Compare comp) {
  if (__begin == __end)
    return;
  if (__begin + 1 == __end)
    return;
  for (auto i = __begin; i != __end; ++i) {
    for (auto j = __begin; j != __end - 1; ++j) {
      if (comp(*(j + 1), *j)) {
        auto temp = *j;
        *j = *(j + 1);
        *(j + 1) = temp;
      }
    }
  }
}
template <typename T>
__PNI_CUDA_MACRO__ inline void average_normalize(
    T *__begin, T *__end) {
  if (__begin == __end)
    return;
  if (__begin + 1 == __end)
    return;
  T sum = 0;
  for (auto i = __begin; i != __end; ++i)
    sum += *i;
  if (sum == 0)
    return;
  for (auto i = __begin; i != __end; ++i)
    *i /= sum;
}
template <typename T>
__PNI_CUDA_MACRO__ inline void maximun_normalize(
    T *__begin, T *__end) {
  if (__begin == __end)
    return;
  if (__begin + 1 == __end)
    return;
  T maxv = *__begin;
  for (auto i = __begin; i != __end; ++i)
    if (*i > maxv)
      maxv = *i;
  if (maxv == 0)
    return;
  for (auto i = __begin; i != __end; ++i)
    *i /= maxv;
}
template <typename T>
__PNI_CUDA_MACRO__ inline void standard_normalize(
    T *__begin, T *__end) {
  if (__begin == __end)
    return;
  if (__begin + 1 == __end)
    return;
  T sum = 0;
  T sum2 = 0;
  int n = 0;
  for (auto i = __begin; i != __end; ++i) {
    sum += *i;
    sum2 += (*i) * (*i);
    n++;
  }
  T mean = sum / n;
  T variance = sum2 / n - mean * mean;
  if (variance <= 0)
    return;
  T stddev = FMath<T>::fsqrt(variance);
  for (auto i = __begin; i != __end; ++i)
    *i = (*i - mean) / stddev;
}

// template <typename ValueType>
// __PNI_CUDA_MACRO__ inline ValueType kNN(
//     Image3DInputSpan<ValueType> __input, basic::Vec3<std::size_t> __index, basic::Vec3<std::size_t> __kernelSize,
//     std::size_t __KNNnumbers) {
//   auto kernelSizeHalf = __kernelSize / 2;
//   __kernelSize = kernelSizeHalf * 2 + basic::Vec3<std::size_t>(1, 1, 1); // ensure odd
//   auto kernelSpan = Span3::create(__kernelSize.x, __kernelSize.y, __kernelSize.z);
//   ValueType temp[kernelSpan.total_size()];
//   for (int dz = -kernelSizeHalf.z; dz <= kernelSizeHalf.z; ++dz)
//     for (int dy = -kernelSizeHalf.y; dy <= kernelSizeHalf.y; ++dy)
//       for (int dx = -kernelSizeHalf.x; dx <= kernelSizeHalf.x; ++dx) {
//         auto posInInput = __index + basic::Vec3<std::size_t>(dx, dy, dz);
//         auto posInKernel = kernelSpan(dx + kernelSizeHalf.x, dy + kernelSizeHalf.y, dz + kernelSizeHalf.z);
//         if (__input.geometry.in(posInInput))
//           temp[posInKernel] = __input.at(posInInput);
//         else
//           temp[posInKernel] = ValueType(0);
//       }
//   standard_normalize(temp, temp + kernelSpan.total_size());
//   bubble_sort(temp, temp + kernelSpan.total_size(), [](const auto &a, const auto &b) { return a < b; });
//   __KNNnumbers = basic::FMath<std::size_t>::min(__KNNnumbers, kernelSpan.total_size());
// }
} // namespace
  // openpni::basic
