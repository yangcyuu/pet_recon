#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <thread>

#include "../basic/CpuInfo.hpp"
#include "../basic/DataView.hpp"
#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
#include "../math/Convolution.hpp"
#include "../math/Geometry.hpp"
#include "../process/Foreach.hpp"
#include "Interpolation.hpp"
#include "Sampling.hpp"
namespace openpni::math {
template <typename _SamplingMethod, typename _InterpolationMethod>
struct MethodEMCalculation {
  using SamplingMethod = _SamplingMethod;
  using InterpolstionMethod = _InterpolationMethod;

  _SamplingMethod sampler;
  _InterpolationMethod interpolator;
};

using ProjectionMethodUniformTOF = MethodEMCalculation<SamplingUniformWithTOF<float>, InterpolationNearest3D>;
using ProjectionMethodUniform = MethodEMCalculation<SamplingUniform<float>, InterpolationNearest3D>;
using ProjectionMethodSiddon = MethodEMCalculation<SamplingIntersection<float>, InterpolationNearest3D>;

template <typename GainType, typename T, typename ProjectionMethod>
__PNI_CUDA_MACRO__ inline T EMSum_impl(
    basic::Event<GainType> __Event, const T *__in_img_3d, const basic::Image3DGeometry *__image3dSize,
    const basic::Vec2<basic::Vec3<float>> *__roi, ProjectionMethod __projectionMethod) {
  auto &sampler = __projectionMethod.sampler;
  if (!sampler.setInfo(__Event, __image3dSize, __roi))
    return 0;
  sampler.reset();
  T sum = T(0);
  while (!sampler.isEnd()) {
    const auto samplePoint = sampler.next();
    const auto stepSize = sampler.getlastStepSize();
    if (stepSize == 0)
      continue; // 跳过无效采样点
    sum += __projectionMethod.interpolator(samplePoint, __in_img_3d, *__image3dSize) * stepSize;
  }
  return sum;
}

template <typename EMFactorType, typename ImageValueType, typename GainType, typename ProjectionMethod>
__PNI_CUDA_MACRO__ inline void EMDistribute_impl(
    basic::Event<GainType> __Event, EMFactorType emFactor, ImageValueType *__out_img_3d,
    const basic::Image3DGeometry *__image3dSize, const cubef *__roi, ProjectionMethod __projectionMethod) {
  auto &sampler = __projectionMethod.sampler;
  if (!sampler.setInfo(__Event, __image3dSize, __roi))
    return;
  sampler.reset();
  while (!sampler.isEnd()) {
    const auto samplePoint = sampler.next();
    const auto stepSize = sampler.getlastStepSize();
    if (stepSize == 0)
      continue; // 跳过无效采样点
    __projectionMethod.interpolator.add_value(
        samplePoint, __out_img_3d, emFactor * (__Event.gain ? *__Event.gain : GainType(1.)) * stepSize, *__image3dSize);
  }
}

template <typename GainType, typename ProjectionMethod>
__PNI_CUDA_MACRO__ inline int EMTest_impl(
    basic::Event<GainType> __Event, const basic::Image3DGeometry *__image3dSize,
    const basic::Vec2<basic::Vec3<float>> *__roi, ProjectionMethod __projectionMethod) {
  return __projectionMethod.sampler.test(__Event, __image3dSize, __roi);
}

} // namespace openpni::math
namespace openpni::process {
template <typename EMCalculationMethod, typename ValueTypeResult, typename DeviceImageType, typename DataViewType>
void EMSum(
    Image3DInputSpan<DeviceImageType> __inImage3DSpan, const basic::Vec2<basic::Vec3<float>> &__roi,
    ValueTypeResult *__out_EMSum, const DataViewType &__dataView, EMCalculationMethod __emMethod,
    basic::CpuMultiThread __cpuMultiThread) {
  const auto datasize = __dataView.size();
  process::for_each(
      datasize,
      [&](std::size_t i) {
        __out_EMSum[i] +=
            EMSum_impl(__dataView.at(i), __inImage3DSpan.ptr, &__inImage3DSpan.geometry, &__roi, __emMethod);
      },
      __cpuMultiThread);
}

template <typename EMCalculationMethod, typename ValueTypeEMSum, typename DeviceImageType, typename DataViewType>
void EMDistribute(
    const ValueTypeEMSum *__in_EMSum, DeviceImageType *__out_Img3D, const basic::Image3DGeometry &__image3dSize,
    const basic::Vec2<basic::Vec3<float>> &__roi, const DataViewType &__dataView, EMCalculationMethod __emMethod,
    basic::CpuMultiThread __cpuMultiThread) {
  const auto datasize = __dataView.size();
  process::for_each(
      datasize,
      [&](std::size_t i) {
        math::EMDistribute_impl(__dataView.at(i), __in_EMSum ? __in_EMSum[i] : ValueTypeEMSum(1.), __out_Img3D,
                                &__image3dSize, &__roi, __emMethod);
      },
      __cpuMultiThread);
}

struct _EMUpdate {
  template <typename ImageValueType, typename DataViewType>
  bool checkBufferSize(
      DataViewType &__dataView, const basic::Image3DGeometry &__image3dSize, std::size_t &__bufferSize) const {
    ImageValueType *imageUpdate1 = reinterpret_cast<ImageValueType *>(0);
    ImageValueType *imageUpdate2 = reinterpret_cast<ImageValueType *>(imageUpdate1 + __image3dSize.totalVoxelNum());
    ImageValueType *vecEMSum = reinterpret_cast<ImageValueType *>(imageUpdate2 + __image3dSize.totalVoxelNum());
    char *endOfBuffer = reinterpret_cast<char *>(vecEMSum + __dataView.size());
    if (std::size_t usedBufferSize = endOfBuffer - reinterpret_cast<char *>(0); usedBufferSize > __bufferSize) {
      __bufferSize = usedBufferSize;
      return false; // Buffer size is not enough
    }
    return true; // Buffer size is enough
  }

  template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType,
            typename EMSumHandleMethod, typename ImageUpdateMethod>
  [[nodiscard]] bool operator()(
      const DataViewType &__dataView, Image3DIOSpan<ImageValueType> __image3dIOSpan, const ConvKernel &__kernel,
      ImageValueType *__senmap, void *__buffer, std::size_t &__bufferSize, EMCalculationMethod __emMethod,
      EMSumHandleMethod __emSumHandleMethod, ImageUpdateMethod __imageUpdateMethod,
      basic::CpuMultiThread __cpuMultiThread) const {
    if (!checkBufferSize<ImageValueType>(__dataView, __image3dIOSpan.geometry, __bufferSize))
      return false; // Buffer size is not enough
    ImageValueType *imageUpdate1 = reinterpret_cast<ImageValueType *>(__buffer);
    ImageValueType *imageUpdate2 =
        reinterpret_cast<ImageValueType *>(imageUpdate1 + __image3dIOSpan.geometry.totalVoxelNum());
    ImageValueType *vecEMSum =
        reinterpret_cast<ImageValueType *>(imageUpdate2 + __image3dIOSpan.geometry.totalVoxelNum());
    Image3DSpan<const ImageValueType> __EMSum3DSpan{__image3dIOSpan.geometry, __image3dIOSpan.ptr_out};
    process::convolution3d(__kernel, __image3dIOSpan.geometry, __image3dIOSpan.ptr_in, __image3dIOSpan.ptr_out,
                           __cpuMultiThread);
    process::EMSum(__EMSum3DSpan, vecEMSum, __dataView, __emMethod, __cpuMultiThread);
    __emSumHandleMethod(vecEMSum, __dataView.size());
    process::EMDistribute(vecEMSum, imageUpdate1, __image3dIOSpan.geometry, __dataView, __emMethod, __cpuMultiThread);
    process::convolution3d(__kernel, __image3dIOSpan.geometry, imageUpdate1, imageUpdate2, __cpuMultiThread);
    __imageUpdateMethod(__image3dIOSpan.ptr_in, __image3dIOSpan.ptr_out, __senmap,
                        __image3dIOSpan.geometry.totalVoxelNum());
    return true;
  }
};
inline constexpr _EMUpdate EMUpdate{};
} // namespace openpni::process
