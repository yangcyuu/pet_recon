#pragma once
#include <array>
#include <memory>
#include <numbers>

#include "../basic/Math.hpp"
#include "../misc/Platform-Independent.hpp"
namespace openpni::math {
template <FloatingPoint_c ValueType, int SampleNum>
struct ERFSample {
  static_assert(SampleNum > 0, "SampleNum must be greater than 0");

  ValueType samples[SampleNum + 1];
  ValueType samplePrecision;

  ValueType at(
      ValueType x) const noexcept {
    unsigned index = std::floor(std::abs(x * samplePrecision));
    ValueType interpolation = std::abs(x * samplePrecision) - index; // 计算插值部分
    ValueType sgn = x < ValueType(0) ? -1 : 1;
    if (index >= SampleNum)
      return samples[SampleNum] * sgn;
    return (samples[index] * (1 - interpolation) + samples[index + 1] * interpolation) * sgn;
  }

  ValueType operator[](
      ValueType x) const noexcept {
    return at(x);
  }

  ValueType guassianIntegral(
      ValueType a, ValueType b) const noexcept {
    return ((*this)[b / std::numbers::sqrt2] - (*this)[a / std::numbers::sqrt2]) / ValueType(2);
  }
};

template <FloatingPoint_c ValueType, int SampleNumIn01,
          int Sigma = 5> // 这里的 SampleNum 是[0, 1.f)范围采样点数量
constexpr ERFSample<ValueType, SampleNumIn01 * Sigma> erfSample() {
  static_assert(Sigma > 0, "Sigma must be greater than 0");
  ERFSample<ValueType, SampleNumIn01 * Sigma> result;
  result.samplePrecision = SampleNumIn01;
  for (int i = 0; i <= SampleNumIn01 * Sigma; ++i) {
    ValueType x = static_cast<ValueType>(i) / static_cast<ValueType>(SampleNumIn01);
    result.samples[i] = std::erf(x);
  }
  return result;
}

template <FloatingPoint_c ValueType, typename Func>
constexpr inline std::vector<ValueType> importanceSampling(
    unsigned __totalSampleNum, ValueType __begin, ValueType __end,
    Func __funcImportance) // funcImportance必须总是正数
{
  if (__totalSampleNum <= 0)
    return {};
  double sumImportance = 0;
  std::vector<ValueType> result{};
  std::vector<ValueType> stepSize{};
  result.resize(__totalSampleNum);
  stepSize.resize(__totalSampleNum);

  ValueType step =
      (static_cast<ValueType>(__end) - static_cast<ValueType>(__begin)) / static_cast<ValueType>(__totalSampleNum);
  for (unsigned i = 0; i < __totalSampleNum; ++i) {
    ValueType x = static_cast<ValueType>(__begin) + static_cast<ValueType>(i + 0.5) * step;
    ValueType importance = 1. / __funcImportance(x);
    stepSize[i] = importance;
    sumImportance += importance;
  }

  // 将步长归一化
  for (unsigned i = 0; i < __totalSampleNum; ++i)
    stepSize[i] /= static_cast<ValueType>(sumImportance);
  // 计算步长累积
  for (unsigned i = 1; i < __totalSampleNum; ++i)
    stepSize[i] += stepSize[i - 1];

  // 应用到采样点，采样点的值是区间中点
  for (unsigned i = 0; i < __totalSampleNum; ++i) {
    if (i == 0)
      result[i] = __begin + step * (stepSize[i] / 2);
    else
      result[i] = __begin + step * (stepSize[i - 1] + stepSize[i]) / 2;
  }

  return result;
}

template <FloatingPoint_c ValueType>
inline constexpr std::vector<ValueType> gaussianImportanceSampling(
    unsigned __totalSampleNum,
    ValueType rangeBySigma) // 范围是[-sigma, +sigma]的多少倍
{
  rangeBySigma = std::abs(rangeBySigma);
  return importanceSampling<ValueType>(__totalSampleNum, -rangeBySigma, rangeBySigma, [](ValueType x) -> ValueType {
    return std::exp(-x * x / ValueType(2)) / std::numbers::sqrt2;
  });
}

template <FloatingPoint_c ValueType>
struct LayeredGuassianSampling {
  std::unique_ptr<ValueType[]> samples;
  unsigned sampleDepth;
  std::unique_ptr<uint64_t[]> sampleBegins;
  std::unique_ptr<uint64_t[]> sampleNums;
  ValueType rangeBySigma; // 范围是[-sigma, +sigma]的多少倍
};

template <FloatingPoint_c ValueType>
struct LayeredGuassianSampling<ValueType> generateLayeredGuassianSamplingByPower2(
    unsigned sampleDepth, ValueType rangeBySigma) {
  const uint64_t totalSampleNum = (1ull << sampleDepth) - 1; // 1 + 2 + 4 + ... == 2^sampleDepth - 1
  LayeredGuassianSampling<ValueType> result;
  result.sampleDepth = sampleDepth;
  result.rangeBySigma = rangeBySigma;
  result.samples = std::make_unique<ValueType[]>(totalSampleNum);
  result.sampleBegins = std::make_unique<uint64_t[]>(sampleDepth);
  result.sampleNums = std::make_unique<uint64_t[]>(sampleDepth);
  for (unsigned i = 1; i <= sampleDepth; i++) {
    result.sampleBegins[i - 1] = (1ull << i) - 1; // 1 + 2 + 4 + ... == 2^i - 1
    result.sampleNums[i - 1] = (1ull << (i - 1)); // 2^(i-1)
    const auto gaussianSamples = gaussianImportanceSampling<ValueType>(result.sampleNums[i - 1], rangeBySigma);
    std::copy(gaussianSamples.begin(), gaussianSamples.end(), result.samples.get() + result.sampleBegins[i - 1]);
  }
}

__PNI_CUDA_MACRO__ inline float calTOFGaussianCenter(
    int16_t deltaTime, float distance_cry1_cry2) {
  return 0.5 * ((static_cast<float>(deltaTime) * misc::speed_of_light_ps / distance_cry1_cry2) + 1.f);
} // 参数方程01之间的结果

} // namespace openpni::math
