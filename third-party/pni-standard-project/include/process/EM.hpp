#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <ranges>
#include <thread>

#include "../basic/DataView.hpp"
#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
#include "../math/Convolution.hpp"
#include "../math/EMStep.hpp"
#include "Foreach.hpp"
#define PNI_TOSTRING(content) ((std::stringstream() << content).str())
namespace openpni::process {
struct EMSumSimpleUpdate {
  void operator()(
      auto *__emSum, std::size_t count) {
    for (std::size_t i = 0; i < count; i++) {
      __emSum[i] = 1.0f / __emSum[i];
    }
  }
};
struct ImageSimpleUpdate {
  void operator()(
      auto *__in_img3D, auto *__outImg3D, auto *__senMap, std::size_t count) {
    for (std::size_t i = 0; i < count; i++) {
      __outImg3D[i] = __outImg3D[i] / __senMap[i] * __in_img3D[i];
    }
  }
};

template <typename ImageValueType>
struct ImageUpdateMethod {
  ImageValueType *outImg;
  ImageValueType *updateImage;
  ImageValueType *senMap;
  __PNI_CUDA_MACRO__ ImageValueType operator()(
      std::size_t i) const {
    return outImg[i] * updateImage[i] / senMap[i];
  }
};

struct SetOutOfViewZero {
  float *outImg;
  uint64_t rr;
  const basic::Image3DGeometry *image3dSize;

  __PNI_CUDA_MACRO__ float operator()(
      uint64_t i) const {
    auto index = image3dSize->at(i);
    if ((index.x - image3dSize->voxelNum.x / 2) * (index.x - image3dSize->voxelNum.x / 2) +
            ((index.y - image3dSize->voxelNum.y / 2) * (index.y - image3dSize->voxelNum.y / 2)) >
        rr)
      return 0.;
    else
      return outImg[i];
  }
};

template <typename ImageValueType>
struct FixSenmapValue {
  ImageValueType *senMap;
  float senMapPickupThreshold;

  __PNI_CUDA_MACRO__ ImageValueType operator()(
      std::size_t i) const {
    if (senMap[i] <= senMapPickupThreshold)
      return INFINITY; // this will set outImg to zero
    else
      return senMap[i];
  }
};

template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType>
inline void calSenmap(
    DataViewType __dataView, Image3DSpan<ImageValueType> __in_senmapImage3dSpan, const ConvKernel &__kernel,
    EMCalculationMethod __emMethod, basic::CpuMultiThread __cpuMultiThread) {
  std::unique_ptr<ImageValueType[]> tmpSenmap =
      std::make_unique_for_overwrite<ImageValueType[]>(__in_senmapImage3dSpan.geometry.totalVoxelNum());
  process::EMDistribute((ImageValueType *)nullptr, __in_senmapImage3dSpan.ptr, __in_senmapImage3dSpan.geometry,
                        __in_senmapImage3dSpan.geometry.roi(), __dataView, __emMethod, __cpuMultiThread);
  // process::convolution3d(__kernel, __in_senmapImage3dSpan.geometry, tmpSenmap.get(), __in_senmapImage3dSpan.ptr,
  // __cpuMultiThread);
}

struct _SEM_simple {
  template <typename ImageValueType, typename DataViewType>
  [[nodiscard]] inline bool checkBufferSize(
      std::vector<DataViewType> __dataViews, const basic::Image3DGeometry &__image3dSize,
      std::size_t &__bufferSize) const {
    ImageValueType *tempImage1 = reinterpret_cast<ImageValueType *>(0);
    ImageValueType *tempImage2 = reinterpret_cast<ImageValueType *>(tempImage1 + __image3dSize.totalVoxelNum());
    char *endOfBufferThisLayer = reinterpret_cast<char *>(tempImage2 + __image3dSize.totalVoxelNum());
    std::size_t bufferSizeForEMUpdate = 0;
    for (const auto &dataView : __dataViews)
      EMUpdate.checkBufferSize<ImageValueType>(dataView, __image3dSize, bufferSizeForEMUpdate);
    std::size_t usedBufferSize = (endOfBufferThisLayer - reinterpret_cast<char *>(0)) + bufferSizeForEMUpdate;
    if (usedBufferSize > __bufferSize) {
      __bufferSize = usedBufferSize;
      return false; // Buffer size is not enough
    }
    return true; // Buffer size is enough
  }
  template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType,
            typename EMSumHandleMethod, typename ImageUpdateMethod>
  [[nodiscard]] inline bool operator()(
      std::vector<DataViewType> __dataViews, Image3DSpan<ImageValueType> __out_osemImage3dSpan,
      const ConvKernel &__kernel, std::vector<ImageValueType *> &__senmaps, int __iterations, void *__buffer,
      std::size_t &__bufferSize, EMCalculationMethod __emMethod, EMSumHandleMethod __emSumHandleMethod,
      ImageUpdateMethod __imageUpdateMethod, basic::CpuMultiThread __cpuMultiThread) const {
    if (!checkBufferSize<ImageValueType, DataViewType>(__dataViews, __out_osemImage3dSpan.geometry, __bufferSize))
      return false; // Buffer size is not enough

    ImageValueType *tempImage1 = reinterpret_cast<ImageValueType *>(__buffer);
    ImageValueType *tempImage2 =
        reinterpret_cast<ImageValueType *>(tempImage1 + __out_osemImage3dSpan.geometry.totalVoxelNum());
    char *bufferForEMUpdate = reinterpret_cast<char *>(tempImage2 + __out_osemImage3dSpan.geometry.totalVoxelNum());
    std::size_t bufferSizeForEMUpdate = reinterpret_cast<char *>(__buffer) + __bufferSize - bufferForEMUpdate;
    std::fill(tempImage1, tempImage1 + __out_osemImage3dSpan.geometry.totalVoxelNum(), 1.);

    for (const auto iterId : std::views::iota(0, __iterations))
      for (const auto dataViewId : std::views::iota(0ull, __dataViews.size())) {
        EMUpdate(__dataViews[dataViewId], tempImage1, tempImage2, __out_osemImage3dSpan.geometry, __kernel,
                 __senmaps[dataViewId], bufferForEMUpdate, bufferSizeForEMUpdate, __emMethod, __emSumHandleMethod,
                 __imageUpdateMethod, __cpuMultiThread);
        std::swap(tempImage1, tempImage2);
        PNI_DEBUG(PNI_TOSTRING("OSEM iteration " << iterId << ", dataView " << dataViewId << " completed."));
      }
    std::copy(tempImage2, tempImage2 + __out_osemImage3dSpan.geometry.totalVoxelNum(), __out_osemImage3dSpan.ptr);
    return true;
  }
};
inline constexpr _SEM_simple SEM_simple{};
} // namespace openpni::process
