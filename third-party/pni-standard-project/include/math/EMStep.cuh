#pragma once
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/shuffle.h>
#include <vector>

#include "../PnI-Config.hpp"
#include "../basic/CudaPtr.hpp"
#include "Convolution.cuh"
#include "EMStep.hpp"
namespace openpni::process {
template <int N = 2>
struct SubROISelect1 {
  static_assert(N >= 2, "SubROISelect1 can only be used with N = 1");
  static std::vector<cubef> select(
      const cubef &__roiAll) {
    std::vector<cubef> result;
    const auto roiBegin000 = __roiAll.x;
    const auto roiEnd000 = roiBegin000 + (__roiAll.y - __roiAll.x) / N;
    const auto roiSize = roiEnd000 - roiBegin000;
    for (int i = 0; i < N * N * N; i++) {
      basic::Vec3<int> bxbybz = basic::Vec3<int>(i % N, (i / N) % N, i / (N * N));
      auto currentROIBegin = roiBegin000 + roiSize * bxbybz;
      auto currentROIEnd = roiEnd000 + roiSize * bxbybz;
      result.push_back(basic::Vec2<basic::Vec3<float>>(currentROIBegin, currentROIEnd));
    }
    return result;
  }
};

template <int N = 4>
struct SubROISelect2 {
  static std::vector<cubef> select(
      const cubef &__roiAll) // Z方向分为 4层
  {
    std::vector<cubef> result;
    const auto Zdirect = (__roiAll.y.z - __roiAll.x.z) / N;
    const auto roiBegin000 = __roiAll.x;
    auto roiEnd000 = basic::make_vec3<float>(__roiAll.y.x, __roiAll.y.y, __roiAll.x.z + Zdirect);
    for (int i = 0; i < N; i++) {
      auto currentROIBegin = basic::make_vec3<float>(roiBegin000.x, roiBegin000.y, roiBegin000.z + Zdirect * i);
      auto currentROIEnd = basic::make_vec3<float>(roiEnd000.x, roiEnd000.y, roiEnd000.z + Zdirect * i);
      result.push_back(basic::Vec2<basic::Vec3<float>>(currentROIBegin, currentROIEnd));
    }
    return result;
  }
};

template <typename EMCalculationMethod, typename ValueTypeResult, typename DeviceImageType, typename DataViewType>
__global__ void EMSum_KERNEL(
    const DeviceImageType *__in_Img3D, basic::Image3DGeometry __image3dSize, basic::Vec2<basic::Vec3<float>> __roi,
    ValueTypeResult *__out_EMSum, DataViewType __dataView, EMCalculationMethod __emMethod) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __dataView.size())
    return;
  __out_EMSum[idx] += math::EMSum_impl(__dataView.at(idx), __in_Img3D, &__image3dSize, &__roi, __emMethod);
}

template <typename EMCalculationMethod, typename ValueTypeResult, typename DeviceImageType, typename DataViewType>
__global__ void EMSumFiltered_KERNEL(
    const DeviceImageType *__in_Img3D, const std::size_t *__in_validIndexes, std::size_t __count,
    basic::Image3DGeometry __image3dSize, basic::Vec2<basic::Vec3<float>> __roi, ValueTypeResult *__out_EMSum,
    DataViewType __dataView, EMCalculationMethod __emMethod) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __count)
    return;

  const auto index = __in_validIndexes[idx];
  __out_EMSum[index] += math::EMSum_impl(__dataView.at(index), __in_Img3D, &__image3dSize, &__roi, __emMethod);
}

template <typename ValueTypeEMSum, typename DeviceImageType, typename DataViewType, typename EMCalculationMethod>
__global__ void EMDistribute_KERNEL(
    const ValueTypeEMSum *__in_EMSum, DeviceImageType *__out_Img3D, basic::Image3DGeometry __image3dSize,
    basic::Vec2<basic::Vec3<float>> __roi, DataViewType __dataView, EMCalculationMethod __emMethod) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __dataView.size())
    return;
  math::EMDistribute_impl(__dataView.at(idx), __in_EMSum ? __in_EMSum[idx] : ValueTypeEMSum(1.), __out_Img3D,
                          &__image3dSize, &__roi, __emMethod);
}

template <typename EMCalculationMethod, typename ValueTypeEMSum, typename DeviceImageType, typename DataViewType>
__global__ void EMDistributeFiltered_KERNEL(
    const ValueTypeEMSum *__in_EMSum, const std::size_t *__in_validIndexes, std::size_t __count,
    DeviceImageType *__out_Img3D, basic::Image3DGeometry __image3dSize, cubef __roi, DataViewType __dataView,
    EMCalculationMethod __emMethod) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __count)
    return;

  const auto index = __in_validIndexes[idx];
  math::EMDistribute_impl(__dataView.at(index), __in_EMSum ? __in_EMSum[index] : ValueTypeEMSum(1.), __out_Img3D,
                          &__image3dSize, &__roi, __emMethod);
}

template <typename EMCalculationMethod, typename DeviceImageType, typename DataViewType>
void EMSum_CUDA(
    Image3DIOSpan<DeviceImageType> __IO_d_Image3dSpan, const basic::Vec2<basic::Vec3<float>> &__roi,
    DataViewType __dataView, EMCalculationMethod __emMethod) {
  EMSum_KERNEL<<<((__dataView.size() + 255) / 256), 256>>>(__IO_d_Image3dSpan.ptr_in, __IO_d_Image3dSpan.geometry,
                                                           __roi, __IO_d_Image3dSpan.ptr_out, __dataView, __emMethod);
}

template <typename EMCalculationMethod, typename ValueTypeResult, typename DeviceImageType, typename DataViewType>
void EMSumFiltered_CUDA(
    const DeviceImageType *__in_Img3D, std::size_t *__d_validIndexes, basic::Image3DGeometry __image3dSize,
    const basic::Vec2<basic::Vec3<float>> &__roi, ValueTypeResult *__out_EMSum, DataViewType __dataView,
    EMCalculationMethod __emMethod, bool __skipFilter, cudaStream_t __stream = cudaStreamDefault) {
  thrust::transform(thrust::cuda::par.on(__stream), thrust::counting_iterator<std::size_t>(0),
                    thrust::counting_iterator<std::size_t>(__dataView.size()),
                    thrust::device_pointer_cast(__d_validIndexes), [] __device__(std::size_t i) { return i; });
  auto end =
      __skipFilter
          ? thrust::device_pointer_cast(__d_validIndexes + __dataView.size())
          : thrust::partition(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(__d_validIndexes),
                              thrust::device_pointer_cast(__d_validIndexes + __dataView.size()),
                              [__dataView, __image3dSize, __roi, __emMethod] __device__(std::size_t i) {
                                return math::EMTest_impl(__dataView.at(i), &__image3dSize, &__roi, __emMethod) > 0;
                              });
  thrust::for_each(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(__d_validIndexes), end,
                   [__in_Img3D, __image3dSize, __roi, __dataView, __emMethod, __out_EMSum] __device__(std::size_t i) {
                     __out_EMSum[i] +=
                         math::EMSum_impl(__dataView.at(i), __in_Img3D, &__image3dSize, &__roi, __emMethod);
                   });
}

template <typename EMCalculationMethod, typename ValueTypeEMSum, typename DeviceImageType, typename DataViewType>
void EMDistribute_CUDA(
    const ValueTypeEMSum *__in_EMSum, DeviceImageType *__out_Img3D, basic::Image3DGeometry __image3dSize,
    basic::Vec2<basic::Vec3<float>> __roi, DataViewType __dataView, EMCalculationMethod __emMethod) {
  EMDistribute_KERNEL<<<((__dataView.size() + 255) / 256), 256>>>(__in_EMSum, __out_Img3D, __image3dSize, __roi,
                                                                  __dataView, __emMethod);
}

template <typename EMCalculationMethod, typename ValueTypeEMSum, typename DeviceImageType, typename DataViewType>
void EMDistributeFiltered_CUDA(
    const ValueTypeEMSum *__in_EMSum, std::size_t *__d_validIndexes, DeviceImageType *__out_Img3D,
    const basic::Image3DGeometry &__image3dSize, const basic::Vec2<basic::Vec3<float>> &__roi,
    const DataViewType &__dataView, EMCalculationMethod __emMethod, bool __skipFilter,
    cudaStream_t __stream = cudaStreamDefault) {
  thrust::transform(thrust::cuda::par.on(__stream), thrust::counting_iterator<std::size_t>(0),
                    thrust::counting_iterator<std::size_t>(__dataView.size()),
                    thrust::device_pointer_cast(__d_validIndexes), [] __device__(std::size_t i) { return i; });
  auto end =
      __skipFilter
          ? thrust::device_pointer_cast(__d_validIndexes + __dataView.size())
          : thrust::partition(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(__d_validIndexes),
                              thrust::device_pointer_cast(__d_validIndexes + __dataView.size()),
                              [__dataView, __image3dSize, __roi, __emMethod] __device__(std::size_t i) {
                                return math::EMTest_impl(__dataView.at(i), &__image3dSize, &__roi, __emMethod) > 0;
                              });
  thrust::for_each(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(__d_validIndexes), end,
                   [__in_EMSum, __out_Img3D, __image3dSize, __roi, __dataView, __emMethod] __device__(std::size_t i) {
                     math::EMDistribute_impl(__dataView.at(i), __in_EMSum ? __in_EMSum[i] : ValueTypeEMSum(1.),
                                             __out_Img3D, &__image3dSize, &__roi, __emMethod);
                   });
}

struct _EMUpdate_CUDA {
  template <typename ImageValueType>
  bool checkCacheOptimize(
      const basic::Image3DGeometry &__image3dSize) const {
    return sizeof(ImageValueType) * __image3dSize.totalVoxelNum() > 1024 * 1024 * 64; // 64MB
  }

  template <typename ImageValueType, typename DataViewType>
  struct _MemoryAccess {
    _MemoryAccess(
        void *__buffer, const DataViewType &__dataView, const basic::Image3DGeometry &__image3dSize) {
      m_imageUpdate1 = openpni::aligned_as<ImageValueType>(__buffer);
      m_imageUpdate2 = openpni::aligned_as<ImageValueType>(m_imageUpdate1 + __image3dSize.totalVoxelNum());
      m_vecEMSum = openpni::aligned_as<ImageValueType>(m_imageUpdate2 + __image3dSize.totalVoxelNum());
      m_vecValidIndexes = openpni::aligned_as<std::size_t>(m_vecEMSum + __dataView.size());
      m_bufferSize =
          reinterpret_cast<char *>(m_vecValidIndexes + __dataView.size()) - reinterpret_cast<char *>(__buffer);
    }

    static std::size_t buffer_size(
        void *__buffer, const DataViewType &__dataView, const basic::Image3DGeometry &__image3dSize) {
      return _MemoryAccess(__buffer, __dataView, __image3dSize).size();
    }

    ImageValueType *&imageUpdate1() { return m_imageUpdate1; }

    ImageValueType *&imageUpdate2() { return m_imageUpdate2; }

    ImageValueType *&vecEMSum() { return m_vecEMSum; }

    std::size_t *&vecValidIndexes() { return m_vecValidIndexes; }

    std::size_t size() const { return m_bufferSize; }

    bool capacity_ok(
        std::size_t &__size) const {
      if (__size >= m_bufferSize)
        return true;
      __size = m_bufferSize;
      return false;
    }

  private:
    ImageValueType *m_imageUpdate1;
    ImageValueType *m_imageUpdate2;
    ImageValueType *m_vecEMSum;
    std::size_t *m_vecValidIndexes;
    std::size_t m_bufferSize;
  };

  template <typename ImageValueType, typename DataViewType>
  bool checkBufferSize(
      const DataViewType &__dataView, const basic::Image3DGeometry &__image3dSize, void *__buffer,
      std::size_t &__bufferSize) const {
    const auto bufferSize =
        _MemoryAccess<ImageValueType, DataViewType>::buffer_size(__buffer, __dataView, __image3dSize);
    if (bufferSize < __bufferSize)
      return true;
    __bufferSize = bufferSize;
    return false;
  }

  template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType,
            typename EMSumHandleMethod, typename ImageUpdateMethod>
  [[nodiscard]] bool operator()(
      const DataViewType &__dataView, const ImageValueType *__d_in_Img3D, ImageValueType *__d_out_Img3D,
      const basic::Image3DGeometry &__image3dSize, const ConvKernel *__d_kernel, ImageValueType *__d_senmap,
      void *__d_buffer, std::size_t &__bufferSize, EMCalculationMethod __emMethod,
      EMSumHandleMethod __emSumHandleMethod, ImageUpdateMethod __imageUpdateMethod,
      cudaStream_t __stream = cudaStreamDefault) const {
    auto memoryAccess = _MemoryAccess<ImageValueType, DataViewType>(__d_buffer, __dataView, __image3dSize);
    if (!memoryAccess.capacity_ok(__bufferSize))
      return false; // Buffer size is not enough

    ImageValueType *imageUpdate1 = memoryAccess.imageUpdate1();
    ImageValueType *imageUpdate2 = memoryAccess.imageUpdate2();
    ImageValueType *vecEMSum = memoryAccess.vecEMSum();
    std::size_t *vecValidIndexes = memoryAccess.vecValidIndexes();

    process::convolution3d_CUDA(__d_kernel, __image3dSize, __d_in_Img3D, __d_out_Img3D);
    cudaMemset(vecEMSum, 0, __dataView.size() * sizeof(ImageValueType));

    const auto roiSubsets = checkCacheOptimize<ImageValueType>(__image3dSize)
                                ? SubROISelect1<2>::select(__image3dSize.roi())
                                : std::vector<cubef>{__image3dSize.roi()};
    for (auto roi : roiSubsets)
      process::EMSumFiltered_CUDA(__d_out_Img3D, vecValidIndexes, __image3dSize, roi, vecEMSum, __dataView, __emMethod,
                                  roiSubsets.size() == 1, __stream);

    __emSumHandleMethod(__dataView, vecEMSum, __stream);

    cudaMemset(imageUpdate1, 0, __image3dSize.totalVoxelNum() * sizeof(ImageValueType));
    for (auto roi : roiSubsets)
      process::EMDistributeFiltered_CUDA(vecEMSum, vecValidIndexes, imageUpdate1, __image3dSize, roi, __dataView,
                                         __emMethod, roiSubsets.size() == 1, __stream);

    process::convolution3d_CUDA(__d_kernel, __image3dSize, imageUpdate1, imageUpdate2, __stream);
    __imageUpdateMethod(__d_in_Img3D, __d_senmap, imageUpdate2, __d_out_Img3D, __image3dSize.totalVoxelNum(), __stream);
    return true;
  }
};
inline constexpr _EMUpdate_CUDA EMUpdate_CUDA{};
} // namespace openpni::process
