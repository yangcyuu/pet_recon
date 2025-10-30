#pragma once
#include "../core/BasicMath.hpp"
namespace openpni::experimental::algorithms {
template <FloatingPoint_c T>
  requires(std::is_same_v<T, std::remove_cv_t<T>>)
void set_max_value_to_1(
    std::span<T> data) {
  if (data.empty())
    return;
  T max_value = *std::max_element(data.begin(), data.end());
  if (max_value <= 0)
    return;
  for (auto &v : data)
    v /= max_value;
}

template <Arithmetic_c T>
struct AverageHelper {
  void add(
      T value) {
    sum += value;
    count++;
  }
  std::optional<T> get() const {
    if (count > 0)
      return sum / static_cast<T>(count);
    else
      return std::nullopt;
  }
  static T apply(
      auto &&range, T _or) { // _or: when no valid value, return this value
    AverageHelper<T> avg;
    for (auto v : range)
      avg.add(v);
    return avg.get().value_or(_or);
  }

private:
  std::size_t count = 0;
  T sum = 0;
};

enum LinearFittingType { LinearFitting_NoBias, LinearFitting_WithBias };
template <typename T, LinearFittingType fittingType>
struct LinearFittingHelper {};
template <typename T>
struct LinearFittingHelper<T, LinearFitting_NoBias> {
  __PNI_CUDA_MACRO__ void add(
      T x, T y) {
    sumX2 += x * x;
    sumXY += x * y;
    count++;
  }
  __PNI_CUDA_MACRO__ T slope() const {
    if (count == 0 || sumX2 == 0)
      return 0;
    return sumXY / sumX2;
  }
  __PNI_CUDA_MACRO__ T intercept() const { return 0; }
  __PNI_CUDA_MACRO__ T predict(
      T x) const {
    return slope() * x + intercept();
  }
  __PNI_CUDA_MACRO__ std::size_t getCount() const { return count; }

private:
  std::size_t count = 0;
  T sumX2 = 0;
  T sumXY = 0;
};
template <typename T>
struct LinearFittingHelper<T, LinearFitting_WithBias> {
  __PNI_CUDA_MACRO__ void add(
      T x, T y) {
    sumX += x;
    sumY += y;
    sumX2 += x * x;
    sumXY += x * y;
    count++;
  }
  __PNI_CUDA_MACRO__ T slope() const {
    if (count == 0 || sumX2 * count == sumX * sumX)
      return 0;
    return (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX);
  }
  __PNI_CUDA_MACRO__ T intercept() const {
    if (count == 0 || sumX2 * count == sumX * sumX)
      return 0;
    return (sumX2 * sumY - sumX * sumXY) / (sumX2 * count - sumX * sumX);
  }
  __PNI_CUDA_MACRO__ T predict(
      T x) const {
    if (count == 0 || sumX2 * count == sumX * sumX)
      return 0;
    return slope() * x + intercept();
  }
  __PNI_CUDA_MACRO__ std::size_t getCount() const { return count; }

private:
  std::size_t count = 0;
  T sumX = 0;
  T sumY = 0;
  T sumX2 = 0;
  T sumXY = 0;
};

template <typename T>
__PNI_CUDA_MACRO__ inline auto cosine(
    const core::Vector<T, 3> &__a, const core::Vector<T, 3> &__b) {
  if constexpr (std::is_same_v<T, float>)
    return __a.dot(__b) * core::FMath<float>::rfsqrt(__a.l22() * __b.l22());
  else
    return __a.dot(__b) * core::FMath<double>::rfsqrt(__a.l22() * __b.l22());
}
template <typename T>
__PNI_CUDA_MACRO__ inline auto sine(
    const core::Vector<T, 3> &__a, const core::Vector<T, 3> &__b) {
  if constexpr (std::is_same_v<T, float>)
    return core::FMath<float>::fsqrt(1 - cosine(__a, __b) * cosine(__a, __b));
  else
    return core::FMath<double>::fsqrt(1 - cosine(__a, __b) * cosine(__a, __b));
}

template <typename T>
__PNI_CUDA_MACRO__ inline auto calculateProjectionArea(
    T planeArea, const core::Vector<T, 3> &planeNormal,
    const core::Vector<T, 3> &projectionDirection) { // 给定一个平面面积 以及平面法向量 求其在任意向量方向的投影面积

  // 计算两个向量的正弦值
  T abs_sinTheta = core::FMath<T>::abs(cosine(planeNormal, projectionDirection));

  // 投影面积 = 原面积 * |sin(θ)|
  return planeArea * abs_sinTheta;
}

} // namespace openpni::experimental::algorithms
