#pragma once
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "../PnI-Config.hpp"
#include "../basic/Math.hpp"
#include "../math/Convolution.cuh"
#include "../math/EMStep.cuh"
#include "EM.hpp"
#include "Foreach.cuh"
#include "Scatter.hpp"
namespace openpni::process {

namespace cuda_private {
__global__ void kernel_ImageSimpleUpdate_CUDA(
    const auto *__in_img3D, const auto *__senMap, const auto *__updateImage, auto *__out_img3D, std::size_t count) {
  std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  __out_img3D[idx] = __in_img3D[idx] * __updateImage[idx] / __senMap[idx];
}

template <typename T>
__global__ void kernel_ImageRatedUpdate_CUDA(
    const T *__in_img3D, const T *__senMap, const T *__updateImage, T *__out_img3D, T __learningRate, T __geometricMean,
    std::size_t count) {
  std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  __out_img3D[idx] = __in_img3D[idx] * powf(__updateImage[idx] / __senMap[idx] /
                                                powf(__geometricMean, (__learningRate - 1) / __learningRate),
                                            __learningRate);
}
template <typename T>
__global__ void halfImage_kernel(
    const T *__d_in_image3D, T *__d_out_image3D, basic::Image3DGeometry __image3dSize,
    basic::Image3DGeometry __image3dHalfSize) {
  const auto idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __image3dHalfSize.totalVoxelNum())
    return;

  const auto xyz = __image3dHalfSize.at(idx);
  const auto _2xyz = xyz * 2u;
  // printf("Half image at idx: %zu, xyz: (%u, %u, %u)\n", idx, xyz.x, xyz.y, xyz.z);
  T sum = 0;
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x, _2xyz.y, _2xyz.z)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x + 1, _2xyz.y, _2xyz.z)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x, _2xyz.y + 1, _2xyz.z)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x + 1, _2xyz.y + 1, _2xyz.z)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x, _2xyz.y, _2xyz.z + 1)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x + 1, _2xyz.y, _2xyz.z + 1)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x, _2xyz.y + 1, _2xyz.z + 1)];
  sum += __d_in_image3D[__image3dSize.at(_2xyz.x + 1, _2xyz.y + 1, _2xyz.z + 1)];
  __d_out_image3D[idx] = sum; // Keep sum.
}

template <typename T>
__global__ void doubleImage_kernel(
    const T *__d_in_image3D, T *__d_out_image3D, basic::Image3DGeometry __image3dSize,
    basic::Image3DGeometry __image3dDoubleSize) {
  const auto idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= __image3dDoubleSize.totalVoxelNum())
    return;

  const auto point = (basic::make_vec3<float>(__image3dDoubleSize.at(idx)) + 0.5) * __image3dDoubleSize.voxelSize +
                     __image3dDoubleSize.imgBegin;
  __d_out_image3D[idx] = math::InterpolationTrilinear3D()(point, __d_in_image3D, __image3dSize);
}

} // namespace cuda_private
template <typename MultiplyFactorAdapter, typename AdditionFactorAdapter>
struct EMSumUpdator_CUDA {
  MultiplyFactorAdapter multiplyFactorAdapter;
  AdditionFactorAdapter additionFactorAdapter;
  float emSumCutRatio = 1e-8;
  template <typename DataViewType, typename EmSumType>
  void operator()(
      DataViewType __dataView, EmSumType *__d_emSum, cudaStream_t __stream = cudaStreamDefault) const {
    const auto elementNum = __dataView.size();
    if (elementNum == 0)
      return;
    const auto max_value =
        thrust::reduce(thrust::device_pointer_cast(__d_emSum), thrust::device_pointer_cast(__d_emSum + elementNum),
                       0.0f, thrust::maximum<std::remove_pointer_t<decltype(__d_emSum)>>());
    const auto cutValue = max_value * emSumCutRatio;
    for_each_CUDA(
        elementNum,
        [this_ = *this, cutValue, __d_emSum] __device__(std::size_t idx) {
          if (__d_emSum[idx] < cutValue)
            __d_emSum[idx] = cutValue;
        },
        __stream);
    for_each_CUDA(
        __dataView.size(),
        [this_ = *this, __d_emSum, __dataView] __device__(std::size_t idx) {
          const auto event = __dataView.at(idx);
          const auto additionFactor = this_.additionFactorAdapter(event);
          const auto multiplyFactor = this_.multiplyFactorAdapter(event);

          if (multiplyFactor > 1e-8f) {
            __d_emSum[idx] = 1 / (__d_emSum[idx] + additionFactor / multiplyFactor);
            // printf("EMSumUpdator_CUDA at idx: %d, emSum: %f, multiplyFactor: %f, additionFactor: %f\n", int(idx),
            //        __d_emSum[idx], multiplyFactor, additionFactor);
          } else {
            __d_emSum[idx] = 0;
          }
        },
        __stream);
  }
};
template <typename sssAdaptor, typename MultiplyFactorAdapter, typename AdditionFactorAdapter>
struct EMSumUpdatorTOF_CUDA {
  MultiplyFactorAdapter multiplyFactorAdapter;
  AdditionFactorAdapter randFactorAdapter;
  sssAdaptor sssFactorAdapter;
  float emSumCutRatio = 1e-8;
  template <typename DataViewType, typename EmSumType>
  void operator()(
      DataViewType __dataView, EmSumType *__d_emSum, cudaStream_t __stream = cudaStreamDefault) const {
    const auto elementNum = __dataView.size();
    if (elementNum == 0)
      return;
    const auto max_value =
        thrust::reduce(thrust::device_pointer_cast(__d_emSum), thrust::device_pointer_cast(__d_emSum + elementNum),
                       0.0f, thrust::maximum<std::remove_pointer_t<decltype(__d_emSum)>>());
    const auto cutValue = max_value * emSumCutRatio;
    for_each_CUDA(
        elementNum,
        [this_ = *this, cutValue, __d_emSum] __device__(std::size_t idx) {
          if (__d_emSum[idx] < cutValue)
            __d_emSum[idx] = cutValue;
        },
        __stream);
    for_each_CUDA(
        __dataView.size(),
        [this_ = *this, __d_emSum, __dataView] __device__(std::size_t idx) {
          const auto event = __dataView.at(idx);
          const auto randFactor = this_.randFactorAdapter(event);
          const auto multiplyFactor = this_.multiplyFactorAdapter(event);
          const auto sssFactor = this_.sssFactorAdapter(event);
          const auto additionFactor = randFactor + sssFactor; // ?
          if (multiplyFactor <= 1e-8f)
            __d_emSum[idx] = 0;
          else
            __d_emSum[idx] = 1 / (__d_emSum[idx] + additionFactor / multiplyFactor);
        },
        __stream);
  }
};

struct ImageSimpleUpdate_CUDA {
  void operator()(
      const auto *__in_img3D, const auto *__senMap, const auto *__updateImage, auto *__out_img3D, std::size_t count,
      cudaStream_t __stream = cudaStreamDefault) const {
    const std::size_t blockSize = 256;
    const std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    cuda_private::kernel_ImageSimpleUpdate_CUDA<<<numBlocks, blockSize, 0, __stream>>>(
        __in_img3D, __senMap, __updateImage, __out_img3D, count);
  }
};

template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType>
inline void calSenmap_CUDA(
    DataViewType __dataView, basic::Image3DGeometry __image3dSize, ImageValueType *__out_senmap, ConvKernel *__kernel,
    EMCalculationMethod __emMethod) {
  cuda_sync_ptr<ImageValueType> tmpSenmap = make_cuda_sync_ptr<ImageValueType>(__image3dSize.totalVoxelNum());
  process::EMDistribute_CUDA((const ImageValueType *)nullptr, tmpSenmap.get(), __image3dSize, __image3dSize.roi(),
                             __dataView, __emMethod);
  process::convolution3d_CUDA(__kernel, __image3dSize, tmpSenmap.get(), __out_senmap);
}

template <typename ImageValueType>
inline void fixSenmap_simple_CUDA(
    ImageValueType *__d_senmap, const basic::Image3DGeometry &__image3dSize, ImageValueType __fixRatio = 0.02) {
  // find the maximum value in the senmap
  thrust::device_ptr<ImageValueType> d_senmap_ptr(__d_senmap);
  ImageValueType average_value = thrust::reduce(d_senmap_ptr, d_senmap_ptr + __image3dSize.totalVoxelNum(),
                                                ImageValueType(0), thrust::plus<ImageValueType>()) /
                                 __image3dSize.totalVoxelNum();
  // set the senmap value below the threshold to INF
  ImageValueType threshold = average_value * __fixRatio;
  thrust::transform(d_senmap_ptr, d_senmap_ptr + __image3dSize.totalVoxelNum(), d_senmap_ptr,
                    [threshold, average_value] __device__(ImageValueType value) -> ImageValueType {
                      return value < threshold ? average_value : value;
                    });
}

template <typename ImageValueType>
inline void fixSenmap_smooth_CUDA(
    ImageValueType *__d_senmap, const basic::Image3DGeometry &__image3dSize, ImageValueType __fixRatio = 0.02) {
  // find the average value in the senmap
  thrust::device_ptr<ImageValueType> d_senmap_ptr(__d_senmap);
  ImageValueType average_value = thrust::reduce(d_senmap_ptr, d_senmap_ptr + __image3dSize.totalVoxelNum(),
                                                ImageValueType(0), thrust::plus<ImageValueType>()) /
                                 __image3dSize.totalVoxelNum();
  // set the senmap value below the threshold to INF
  thrust::transform(d_senmap_ptr, d_senmap_ptr + __image3dSize.totalVoxelNum(), d_senmap_ptr,
                    [average_value, __fixRatio] __device__(ImageValueType value) -> ImageValueType {
                      const auto sigmoidInput = value / (average_value * __fixRatio);
                      if (sigmoidInput < 1e-6)
                        return average_value;
                      return value / basic::pos_sigmoid(sigmoidInput);
                    });
}

struct _SEM_simple_CUDA {
  template <typename ImageValueType, typename DataViewType>
  struct _MemoryAccess {
    _MemoryAccess(
        void *__buffer, const std::vector<DataViewType> &__dataViews, const basic::Image3DGeometry &__image3dSize) {
      m_tempImage1 = reinterpret_cast<ImageValueType *>(__buffer);
      m_tempImage2 = reinterpret_cast<ImageValueType *>(m_tempImage1 + __image3dSize.totalVoxelNum());
      m_endOfBufferThisLayer = reinterpret_cast<char *>(m_tempImage2 + __image3dSize.totalVoxelNum());
      std::size_t bufferSizeForEMUpdate = 0;
      for (const auto &dataView : __dataViews)
        (void)EMUpdate_CUDA.checkBufferSize<ImageValueType, DataViewType>(
            dataView, __image3dSize, m_endOfBufferThisLayer, bufferSizeForEMUpdate);
      m_bufferSize = (m_endOfBufferThisLayer - reinterpret_cast<char *>(__buffer)) + bufferSizeForEMUpdate;
      m_endOfBuffer = m_endOfBufferThisLayer + bufferSizeForEMUpdate;
    }
    static std::size_t buffer_size(
        void *__buffer, const DataViewType &__dataView, const basic::Image3DGeometry &__image3dSize) {
      return _MemoryAccess(__buffer, __dataView, __image3dSize).m_bufferSize;
    }
    ImageValueType *&tempImage1() { return m_tempImage1; }
    ImageValueType *&tempImage2() { return m_tempImage2; }
    char *&endOfBufferThisLayer() { return m_endOfBufferThisLayer; }
    char *&endOfBuffer() { return m_endOfBuffer; }
    std::size_t size() const { return m_bufferSize; }
    bool capacity_ok(
        std::size_t &__size) const {
      if (__size >= m_bufferSize)
        return true;
      __size = m_bufferSize;
      return false;
    }

  private:
    ImageValueType *m_tempImage1;
    ImageValueType *m_tempImage2;
    char *m_endOfBufferThisLayer;
    char *m_endOfBuffer;
    std::size_t m_bufferSize;
  };

  template <typename ImageValueType, typename DataViewType>
  inline bool checkBufferSize(
      std::vector<DataViewType> __dataViews, const basic::Image3DGeometry &__image3dSize, void *__buffer,
      std::size_t &__bufferSize) const {
    return _MemoryAccess<ImageValueType, DataViewType>(__buffer, __dataViews, __image3dSize).capacity_ok(__bufferSize);
  }
  template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType,
            typename EMSumHandleMethod, typename ImageUpdateMethod>
  [[nodiscard]] inline bool operator()(
      std::vector<DataViewType> __d_dataViews, const basic::Image3DGeometry &__image3dSize,
      ImageValueType *__d_out_Img3D, const ConvKernel *__d_kernel, std::vector<ImageValueType *> __d_senmaps,
      int __iterations, void *__d_buffer, std::size_t &__bufferSize, EMCalculationMethod __emMethod,
      std::vector<EMSumHandleMethod> __emSumHandleMethod, ImageUpdateMethod __imageUpdateMethod,
      cudaStream_t __stream = cudaStreamDefault) const {
    auto memoryAccess = _MemoryAccess<ImageValueType, DataViewType>(__d_buffer, __d_dataViews, __image3dSize);
    if (!memoryAccess.capacity_ok(__bufferSize))
      return false; // Buffer size is not enough

    ImageValueType *tempImage1 = memoryAccess.tempImage1();
    ImageValueType *tempImage2 = memoryAccess.tempImage2();
    char *bufferForEMUpdate = memoryAccess.endOfBufferThisLayer();
    std::size_t bufferSizeForEMUpdate = memoryAccess.endOfBuffer() - bufferForEMUpdate;
    thrust::fill(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(tempImage1),
                 thrust::device_pointer_cast(tempImage1 + __image3dSize.totalVoxelNum()), ImageValueType(1.));

    for (const auto iterId : std::views::iota(0, __iterations))
      for (const auto dataViewId : std::views::iota(0ull, __d_dataViews.size())) {
        const auto start = std::chrono::steady_clock::now();
        std::cout << "OSEM iteration " << iterId << ", dataView " << dataViewId << " started." << std::endl;
        (void)EMUpdate_CUDA(__d_dataViews[dataViewId], tempImage1, tempImage2, __image3dSize, __d_kernel,
                            __d_senmaps[dataViewId], bufferForEMUpdate, bufferSizeForEMUpdate, __emMethod,
                            __emSumHandleMethod[dataViewId], __imageUpdateMethod, __stream);
        std::swap(tempImage1, tempImage2);
        const auto time =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        const auto speed = __d_dataViews[dataViewId].size() / std::max(time, int64_t(1));
        PNI_DEBUG(PNI_TOSTRING("OSEM iteration " << iterId << ", dataView " << dataViewId
                                                 << " completed, speed: " << speed << " events/ms, time: " << time
                                                 << " ms, event: " << __d_dataViews[dataViewId].size() << "."));
      }
    thrust::copy(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(tempImage1),
                 thrust::device_pointer_cast(tempImage1 + __image3dSize.totalVoxelNum()),
                 thrust::device_pointer_cast(__d_out_Img3D));
    return true;
  }
};

inline constexpr _SEM_simple_CUDA SEM_simple_CUDA{};

} // namespace openpni::process

namespace openpni::process::experimental {

struct _SEM_fine_CUDA {
  struct EMSumSmoothUpdate_CUDA {
    template <typename DataView, typename T>
    void operator()(
        DataView __dataView, T *__d_emSum) const {
      const auto count = __dataView.size();
      thrust::device_ptr<T> d_emSum_ptr(__d_emSum);
      auto max_value = thrust::reduce(d_emSum_ptr, d_emSum_ptr + count, T(0), thrust::maximum<T>());
      constexpr T min_value = 1e-4;
      const auto threshold = max_value * std::max<T>(cut_ratio, min_value);
      thrust::transform(d_emSum_ptr, d_emSum_ptr + count, d_emSum_ptr, [threshold] __device__(T value) -> T {
        const auto sigmoidInput = value / threshold;
        if (sigmoidInput < 1e-6)
          return 0;
        return basic::pos_sigmoid(sigmoidInput) / value;
      });
      // normalize the sum
    }
    float cut_ratio;
  };
  struct ImageNomalizedUpdate_CUDA {
    template <typename T>
    void operator()(
        const T *__in_img3D, const T *__senMap, const T *__updateImage, T *__out_img3D, std::size_t count) const {
      const std::size_t blockSize = 256;
      const std::size_t numBlocks = (count + blockSize - 1) / blockSize;
      cuda_private::kernel_ImageSimpleUpdate_CUDA<<<numBlocks, blockSize>>>(__in_img3D, __senMap, __updateImage,
                                                                            __out_img3D, count);
    }
  };

  struct ImageRatedUpdate_CUDA {
    template <typename T>
    struct _log_plus {
      __PNI_CUDA_MACRO__ T operator()(
          const T &__l, const T &__r) const {
        return logf(__l) + logf(__r);
      }
    };
    template <typename T>
    void operator()(
        const T *__in_img3D, const T *__senMap, const T *__updateImage, T *__out_img3D, std::size_t count,
        cudaStream_t __stream = cudaStreamDefault) const {
      // Calculate the geometric mean of the update image
      T logSum = 0;
      thrust::reduce(thrust::device_pointer_cast(__updateImage), thrust::device_pointer_cast(__updateImage + count),
                     logSum, _log_plus<T>());
      T geometricMean = expf(logSum / count);

      const std::size_t blockSize = 256;
      const std::size_t numBlocks = (count + blockSize - 1) / blockSize;
      cuda_private::kernel_ImageRatedUpdate_CUDA<<<numBlocks, blockSize>>>(
          __in_img3D, __senMap, __updateImage, __out_img3D, T(update_ratio), geometricMean, count);
    }
    float update_ratio = 2.0f;
  };

  template <typename T>
  inline void halfImage(
      const T *__d_in_image3D, T *__d_out_image3D, const basic::Image3DGeometry &__image3dSize) const {
    auto halfSize = __image3dSize;
    halfSize.voxelNum.x /= 2;
    halfSize.voxelNum.y /= 2;
    halfSize.voxelNum.z /= 2;
    const std::size_t blockSize = 256;
    const std::size_t numBlocks = (halfSize.totalVoxelNum() + blockSize - 1) / blockSize;
    cuda_private::halfImage_kernel<<<numBlocks, blockSize>>>(__d_in_image3D, __d_out_image3D, __image3dSize, halfSize);
  }
  template <typename T>
  inline void doubleImage(
      const T *__d_in_image3D, T *__d_out_image3D, const basic::Image3DGeometry &__image3dSize,
      cudaStream_t __stream = cudaStreamDefault) const {
    auto doubleSize = __image3dSize;
    doubleSize.voxelNum.x *= 2;
    doubleSize.voxelNum.y *= 2;
    doubleSize.voxelNum.z *= 2;
    doubleSize.voxelSize.x /= 2;
    doubleSize.voxelSize.y /= 2;
    doubleSize.voxelSize.z /= 2;
    const std::size_t blockSize = 256;
    const std::size_t numBlocks = (doubleSize.totalVoxelNum() + blockSize - 1) / blockSize;
    cuda_private::doubleImage_kernel<<<numBlocks, blockSize>>>(__d_in_image3D, __d_out_image3D, __image3dSize,
                                                               doubleSize);
    static int index = 0;
    index++;
  }

  std::optional<basic::Image3DGeometry> tryHalfImageSize(
      basic::Image3DGeometry __image3dSize) const {
    if (__image3dSize.voxelNum.x % 2 == 0 && __image3dSize.voxelNum.y % 2 == 0 && __image3dSize.voxelNum.z % 2 == 0) {
      __image3dSize.voxelNum.x /= 2;
      __image3dSize.voxelNum.y /= 2;
      __image3dSize.voxelNum.z /= 2;
      __image3dSize.voxelSize.x *= 2;
      __image3dSize.voxelSize.y *= 2;
      __image3dSize.voxelSize.z *= 2;
      return __image3dSize;
    }
    return std::nullopt; // Cannot halve the image size evenly
  }

  basic::Image3DGeometry doubleImageSize(
      basic::Image3DGeometry __image3dSize) const {
    __image3dSize.voxelNum.x *= 2;
    __image3dSize.voxelNum.y *= 2;
    __image3dSize.voxelNum.z *= 2;
    __image3dSize.voxelSize.x /= 2;
    __image3dSize.voxelSize.y /= 2;
    __image3dSize.voxelSize.z /= 2;
    return __image3dSize;
  }
  int calMaxHalfImageNum(
      const basic::Image3DGeometry &__image3dSize) const {
    const auto halfImageSize = tryHalfImageSize(__image3dSize);
    if (halfImageSize)
      return 1 + calMaxHalfImageNum(*halfImageSize);
    return 0;
  }

  template <typename ImageValueType, typename DataViewType>
  [[nodiscard]] inline bool checkBufferSize(
      std::vector<DataViewType> __dataViews, const basic::Image3DGeometry &__image3dSize, int __iterations,
      int __maxHalfImageNum, std::size_t &__bufferSize) const {
    const int osemSteps = __dataViews.size() * __iterations;
    __maxHalfImageNum =
        std::clamp<int>(__maxHalfImageNum, 0, std::min<int>(calMaxHalfImageNum(__image3dSize), osemSteps - 1));

    ImageValueType *tempImage1 = reinterpret_cast<ImageValueType *>(0);
    ImageValueType *tempImage2 = reinterpret_cast<ImageValueType *>(tempImage1 + __image3dSize.totalVoxelNum());
    ImageValueType *tempImage3 = reinterpret_cast<ImageValueType *>(tempImage2 + __image3dSize.totalVoxelNum());

    std::size_t halfVoxelNum = __image3dSize.totalVoxelNum();
    ImageValueType *senmapBuffer = tempImage3 + __image3dSize.totalVoxelNum();
    for (int i = __maxHalfImageNum - 1; i >= 0; i--) {
      halfVoxelNum /= 8;
      senmapBuffer += halfVoxelNum * __dataViews.size();
    }
    char *endOfBufferThisLayer = reinterpret_cast<char *>(senmapBuffer);
    std::size_t bufferSizeForEMUpdate = 0;
    for (const auto &dataView : __dataViews)
      (void)EMUpdate_CUDA.checkBufferSize<ImageValueType>(dataView, __image3dSize, endOfBufferThisLayer,
                                                          bufferSizeForEMUpdate);
    std::cout << "Buffer size for EM update: " << bufferSizeForEMUpdate << std::endl;
    std::size_t usedBufferSize = (endOfBufferThisLayer - reinterpret_cast<char *>(0)) + bufferSizeForEMUpdate;
    if (usedBufferSize > __bufferSize) {
      __bufferSize = usedBufferSize;
      return false; // Buffer size is not enough
    }
    return true; // Buffer size is enough
  }

  template <typename EMCalculationMethod, typename ConvKernel, typename ImageValueType, typename DataViewType>
  [[nodiscard]] inline bool operator()(
      std::vector<DataViewType> __d_dataViews, const basic::Image3DGeometry &__image3dSize,
      ImageValueType *__d_out_Img3D,
      // basic::Point3D<ImageValueType> __gaussianKernelHWHM, // 高斯半高半宽
      ConvKernel *__d_convKernel, const std::vector<ImageValueType *> &__d_senmaps, int __iterations,
      int __maxHalfImageNum, void *__d_buffer, std::size_t &__bufferSize, EMCalculationMethod __emMethod,
      float __emSumCutUpdateRatio, cudaStream_t __stream = cudaStreamDefault) const {
    if (!checkBufferSize<ImageValueType, DataViewType>(__d_dataViews, __image3dSize, __iterations, __maxHalfImageNum,
                                                       __bufferSize))
      return false; // Buffer size is not enough

    const int osemSteps = __d_dataViews.size() * __iterations;
    __maxHalfImageNum =
        std::clamp<int>(__maxHalfImageNum, 0, std::min<int>(calMaxHalfImageNum(__image3dSize), osemSteps - 1));
    const int osemStage = __maxHalfImageNum + 1;
    const int stepsPerStage = osemSteps / osemStage;

    ImageValueType *tempImage1 = reinterpret_cast<ImageValueType *>(__d_buffer);
    ImageValueType *tempImage2 = reinterpret_cast<ImageValueType *>(tempImage1 + __image3dSize.totalVoxelNum());
    ImageValueType *tempImage3 = reinterpret_cast<ImageValueType *>(tempImage2 + __image3dSize.totalVoxelNum());
    cudaMemset(tempImage3, 0, __image3dSize.totalVoxelNum() * sizeof(ImageValueType));
    std::vector<std::vector<ImageValueType *>> d_senmaps_half(__maxHalfImageNum + 1);
    d_senmaps_half[__maxHalfImageNum] = __d_senmaps;
    ImageValueType *senmapBuffer = tempImage3 + __image3dSize.totalVoxelNum();
    basic::Image3DGeometry currentImageSize = __image3dSize;
    for (int i = __maxHalfImageNum - 1; i >= 0; i--) {
      d_senmaps_half[i].resize(__d_senmaps.size());
      currentImageSize = *tryHalfImageSize(currentImageSize);
      for (std::size_t j = 0; j < __d_senmaps.size(); j++) {
        d_senmaps_half[i][j] = senmapBuffer;
        halfImage(d_senmaps_half[i + 1][j], d_senmaps_half[i][j], doubleImageSize(currentImageSize));
        senmapBuffer += currentImageSize.totalVoxelNum();
      }
    }
    char *bufferForEMUpdate = reinterpret_cast<char *>(senmapBuffer);
    std::size_t bufferSizeForEMUpdate = reinterpret_cast<char *>(__d_buffer) + __bufferSize - bufferForEMUpdate;

    thrust::fill(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(tempImage1),
                 thrust::device_pointer_cast(tempImage1 + __image3dSize.totalVoxelNum()), ImageValueType(1.));

    EMSumSmoothUpdate_CUDA emSumUpdate;
    emSumUpdate.cut_ratio = 1.;
    ImageRatedUpdate_CUDA imageUpdateMethod;
    imageUpdateMethod.update_ratio = 3;
    int averageStepNum = 0;
    float logSum = 0;
    for (const auto step : std::views::iota(0, osemSteps)) {
      if (step == osemSteps - 1)
        imageUpdateMethod.update_ratio = 1;
      const auto iterId = step / __d_dataViews.size();
      const auto dataViewId = step % __d_dataViews.size();
      const auto osemStageId = std::min(step / stepsPerStage, osemStage - 1);
      const auto osemStepInStage = step % stepsPerStage;

      const auto start = std::chrono::high_resolution_clock::now();
      (void)EMUpdate_CUDA(__d_dataViews[dataViewId], tempImage1, tempImage2, currentImageSize, __d_convKernel,
                          d_senmaps_half[std::min<int>(osemStageId, d_senmaps_half.size() - 1)][dataViewId],
                          bufferForEMUpdate, bufferSizeForEMUpdate, __emMethod, emSumUpdate, imageUpdateMethod,
                          __stream);
      std::swap(tempImage1, tempImage2);
      PNI_DEBUG("OSEM iteration " << iterId << ", dataView " << dataViewId << " completed, stage " << osemStageId
                                  << ", step in stage " << osemStepInStage << ", step per stage " << stepsPerStage
                                  << ", speed: "
                                  << __d_dataViews[dataViewId].count /
                                         std::max<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                               std::chrono::high_resolution_clock::now() - start)
                                                               .count(),
                                                           1)
                                  << " events/ms");
      emSumUpdate.cut_ratio *= __emSumCutUpdateRatio;
      imageUpdateMethod.update_ratio = (0.5 + imageUpdateMethod.update_ratio) / 1.5f;
      if (osemStepInStage == stepsPerStage - 1 && osemStageId < __maxHalfImageNum) {
        doubleImage(tempImage1, tempImage2, currentImageSize, __stream);
        currentImageSize = doubleImageSize(currentImageSize);
        std::swap(tempImage1, tempImage2);
        imageUpdateMethod.update_ratio = 2;
        PNI_DEBUG(PNI_TOSTRING("OSEM stage " << osemStageId << " completed, image size: " << currentImageSize.voxelNum.x
                                             << "x" << currentImageSize.voxelNum.y << "x"
                                             << currentImageSize.voxelNum.z));
      }
      if (osemStageId == osemStage - 1) {
        // Add log of each tempImage1 value to tempImage3
        thrust::transform(thrust::cuda::par.on(__stream), thrust::counting_iterator<std::size_t>(0),
                          thrust::counting_iterator<std::size_t>(currentImageSize.totalVoxelNum()),
                          thrust::device_pointer_cast(tempImage3),
                          [tempImage1, tempImage3, averageStepNum] __device__(std::size_t i) -> ImageValueType {
                            return log(tempImage1[i]) * (averageStepNum + 1) + tempImage3[i];
                          });
        logSum += averageStepNum + 1;
        averageStepNum++;
      }
    }

    thrust::transform(thrust::cuda::par.on(__stream), thrust::device_pointer_cast(tempImage3),
                      thrust::device_pointer_cast(tempImage3 + currentImageSize.totalVoxelNum()),
                      thrust::device_pointer_cast(__d_out_Img3D),
                      [logSum] __device__(ImageValueType value) -> ImageValueType { return exp(value / logSum); });

    return true;
  }
};
inline constexpr _SEM_fine_CUDA SEM_fine_CUDA{};
} // namespace openpni::process::experimental
