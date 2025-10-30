#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

#include "../basic/CpuInfo.hpp"
#include "../basic/Image.hpp"
#include "../basic/Point.hpp"
namespace openpni::process {
template <int U, int V, int W, typename Precision>
struct ConvolutionKernelStatic {
  static_assert(U % 2 == 1 && V % 2 == 1 && W % 2 == 1, "Kernel size must be odd");
  static_assert(U > 0 && V > 0 && W > 0, "Kernel size must be greater than zero");
  using _ValueType = Precision;
  static constexpr std::size_t kernelSize = std::size_t(U) * std::size_t(V) * std::size_t(W);
  static constexpr int kernelHalfU = U / 2;
  static constexpr int kernelHalfV = V / 2;
  static constexpr int kernelHalfW = W / 2;

  __PNI_CUDA_MACRO__ constexpr int sizeU() const { return U; }
  __PNI_CUDA_MACRO__ constexpr int sizeV() const { return V; }
  __PNI_CUDA_MACRO__ constexpr int sizeW() const { return W; }

  __PNI_CUDA_MACRO__ Precision &at(
      int x, int y, int z) {
    return kernel[x][y][z];
  }
  __PNI_CUDA_MACRO__ const Precision &at(
      int x, int y, int z) const {
    return kernel[x][y][z];
  }

private:
  Precision kernel[U][V][W];
};

template <typename ConvKernel>
inline ConvKernel flip(
    const ConvKernel &kernel) {
  ConvKernel flippedKernel;
  for (int kz = -kernel.kernelHalfW; kz <= kernel.kernelHalfW; ++kz)
    for (int ky = -kernel.kernelHalfV; ky <= kernel.kernelHalfV; ++ky)
      for (int kx = -kernel.kernelHalfU; kx <= kernel.kernelHalfU; ++kx) {
        flippedKernel.kernel[kx + kernel.kernelHalfU][ky + kernel.kernelHalfV][kz + kernel.kernelHalfW] =
            kernel.kernel[-kx + kernel.kernelHalfU][-ky + kernel.kernelHalfV][-kz + kernel.kernelHalfW];
      }
  return flippedKernel;
}

template <typename ConvKernel, typename ImageValueType>
__PNI_CUDA_MACRO__ inline ImageValueType convolution3d_impl(
    const ConvKernel *__kernel, const basic::Image3DGeometry &__image3dSize, const ImageValueType *__in_Img3D,
    ImageValueType *__out_Img3D, int x, int y, int z) {
  ImageValueType sum = 0;
  for (int kz = -__kernel->sizeW() / 2; kz <= __kernel->sizeW() / 2; ++kz)
    for (int ky = -__kernel->sizeV() / 2; ky <= __kernel->sizeV() / 2; ++ky)
      for (int kx = -__kernel->sizeU() / 2; kx <= __kernel->sizeU() / 2; ++kx)
        if (const auto _index = basic::make_vec3<unsigned>(x + kx, y + ky, z + kz); __image3dSize.in(_index))
          sum += __in_Img3D[__image3dSize.at(_index)] *
                 __kernel->at(kx + __kernel->sizeU() / 2, ky + __kernel->sizeV() / 2, kz + __kernel->sizeW() / 2);
  return sum;
}

template <typename ConvKernel, typename ImageValueType>
inline void convolution3d(
    const ConvKernel &__kernel, const basic::Image3DGeometry &__image3dSize, const ImageValueType *__in_Img3D,
    ImageValueType *__out_Img3D, basic::CpuMultiThread __cpuMultiThread) {
  const auto kernelHalfU = __kernel.kernelHalfU;
  const auto kernelHalfV = __kernel.kernelHalfV;
  const auto kernelHalfW = __kernel.kernelHalfW;

#pragma omp parallel for num_threads(__cpuMultiThread.threadNum()) schedule(dynamic, 64)
  for (uint64_t uniformIndex = 0; uniformIndex < __image3dSize.totalVoxelNum(); ++uniformIndex) {
    const auto index3D = __image3dSize.at(uniformIndex);
    __out_Img3D[uniformIndex] =
        convolution3d_impl(__kernel, __image3dSize, __in_Img3D, __out_Img3D, index3D.x, index3D.y, index3D.z);
  }
}

template <typename ConvKernel, typename ImageValueType>
inline void deconvolution3d(
    const ConvKernel &kernel, const basic::Image3DGeometry &image3dSize, const ImageValueType *in_Img3D,
    ImageValueType *out_Img3D, std::size_t maxThreadNum = std::thread::hardware_concurrency()) {
  return convolution3d(flip(kernel), image3dSize, in_Img3D, out_Img3D, maxThreadNum);
}

template <typename T>
struct ConvolutionKernel3D {
public:
  ConvolutionKernel3D(
      const basic::Vec3<unsigned> &kernelSize)
      : m_kernelSize(kernelSize)
      , m_kernel(std::make_unique_for_overwrite<T[]>(kernelSize.x * kernelSize.y * kernelSize.z)) {}
  ConvolutionKernel3D(
      unsigned xSize, unsigned ySize, unsigned zSize)
      : ConvolutionKernel3D(basic::make_vec3<unsigned>(xSize, ySize, zSize)) {}

public:
  template <typename V>
  void apply(
      Image3DIOSpan<V> __span, auto __cpuMultiThread = cpu_threads.singleThread()) {
    for_each(
        __span.geometry.totalVoxelNum(),
        [=, this](std::size_t idx) {
          V sum = 0;
          const auto halfKernelSizeX = static_cast<int>(m_kernelSize.x) / 2;
          const auto halfKernelSizeY = static_cast<int>(m_kernelSize.y) / 2;
          const auto halfKernelSizeZ = static_cast<int>(m_kernelSize.z) / 2;
          const auto indexXYZ = __span.geometry.at(idx);
          for (int kz = -halfKernelSizeZ; kz <= halfKernelSizeZ; ++kz)
            for (int ky = -halfKernelSizeY; ky <= halfKernelSizeY; ++ky)
              for (int kx = -halfKernelSizeX; kx <= halfKernelSizeX; ++kx)
                if (const auto _index = basic::make_vec3<int>(indexXYZ.x + kx, indexXYZ.y + ky, indexXYZ.z + kz);
                    __span.geometry.in(_index))
                  sum += __span.ptr_in[__span.geometry.at(_index)] *
                         m_kernel[(kx + halfKernelSizeX) +
                                  m_kernelSize.x * ((ky + halfKernelSizeY) + m_kernelSize.y * (kz + halfKernelSizeZ))];
          __span.ptr_out[idx] = sum;
        },
        __cpuMultiThread);
  }

private:
  basic::Vec3<unsigned> m_kernelSize;
  std::unique_ptr<T[]> m_kernel;
};
} // namespace openpni::process
