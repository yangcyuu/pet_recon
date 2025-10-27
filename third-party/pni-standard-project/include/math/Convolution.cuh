#pragma once
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../basic/CudaPtr.hpp"
#include "../basic/Vector.hpp"
#include "../process/Foreach.cuh"
#include "Convolution.hpp"
namespace openpni::process {
template <typename ConvKernel>
inline cuda_sync_ptr<ConvKernel> toCUDAKernel(
    const ConvKernel &kernel) {
  return make_cuda_sync_ptr_from_hcopy(&kernel, 1);
}

// CUDA kernel for 3D convolution
template <typename ConvKernel, typename ImageValueType>
__global__ void convolution3dKernel(
    const ConvKernel *kernel, const basic::Image3DGeometry image3dSize, const ImageValueType *d_in_Img3D,
    ImageValueType *out_Img3D) {
  constexpr int U = ConvKernel::kernelHalfU * 2 + 1;
  constexpr int V = ConvKernel::kernelHalfV * 2 + 1;

  __shared__ typename ConvKernel::_ValueType sharedKernel[ConvKernel::kernelSize]; // 使用共享内存存储kernel数据

  const int threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  const int totalThreads = blockDim.x * blockDim.y * blockDim.z;

  // 多轮加载，确保所有kernel元素都被加载
  for (int i = threadId; i < ConvKernel::kernelSize; i += totalThreads) {
    const int kz = i / (U * V);
    const int ky = (i % (U * V)) / U;
    const int kx = i % U;
    sharedKernel[i] = kernel->kernel[kx][ky][kz];
  }

  // 同步线程，确保共享内存中的数据已加载完成
  __syncthreads();

  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= image3dSize.voxelNum.x || y >= image3dSize.voxelNum.y || z >= image3dSize.voxelNum.z)
    return;

  const auto kernelHalfU = kernel->kernelHalfU;
  const auto kernelHalfV = kernel->kernelHalfV;
  const auto kernelHalfW = kernel->kernelHalfW;

  ImageValueType sum = 0;

  for (int kz = -kernelHalfW; kz <= kernelHalfW; ++kz)
    for (int ky = -kernelHalfV; ky <= kernelHalfV; ++ky)
      for (int kx = -kernelHalfU; kx <= kernelHalfU; ++kx) {
        const unsigned xIndex = x + kx;
        const unsigned yIndex = y + ky;
        const unsigned zIndex = z + kz;

        if (image3dSize.in(xIndex, yIndex, zIndex)) {
          int kernelIdx = (kz + kernelHalfW) * U * V + (ky + kernelHalfV) * U + (kx + kernelHalfU);
          sum += d_in_Img3D[image3dSize.at(xIndex, yIndex, zIndex)] * sharedKernel[kernelIdx];
        }
      }

  out_Img3D[image3dSize.at(x, y, z)] = sum;
}

template <typename ConvKernel, typename ImageValueType>
inline void convolution3d_CUDA(
    const ConvKernel *__d_kernel, const basic::Image3DGeometry &__image3dSize, const ImageValueType *__d_in_Img3D,
    ImageValueType *__d_out_Img3D, cudaStream_t __stream = cudaStreamDefault) {
  // Not Used Old Codes.

  // // Define grid and block dimensions
  // dim3 blockSize(8, 8, 8);
  // dim3 gridSize((__image3dSize.voxelNum.x + blockSize.x - 1) / blockSize.x,
  //               (__image3dSize.voxelNum.y + blockSize.y - 1) / blockSize.y,
  //               (__image3dSize.voxelNum.z + blockSize.z - 1) / blockSize.z);

  // // Launch kernel
  // convolution3dKernel<<<gridSize, blockSize>>>(
  //     __d_kernel,
  //     __image3dSize,
  //     __d_in_Img3D,
  //     __d_out_Img3D);
  // cudaDeviceSynchronize();

  // New Version:
  thrust::for_each(thrust::cuda::par.on(__stream), thrust::counting_iterator<uint64_t>(0),
                   thrust::counting_iterator<uint64_t>(__image3dSize.totalVoxelNum()),
                   [=] __device__(uint64_t uniformIndex) {
                     const auto index3D = __image3dSize.at(uniformIndex);
                     __d_out_Img3D[uniformIndex] = convolution3d_impl(__d_kernel, __image3dSize, __d_in_Img3D,
                                                                      __d_out_Img3D, index3D.x, index3D.y, index3D.z);
                   });
}

template <typename ConvKernel, typename ImageValueType>
inline void deconvolution3d_CUDA(
    const ConvKernel &kernel, const basic::Image3DGeometry &image3dSize, const ImageValueType *in_Img3D,
    ImageValueType *out_Img3D) {
  convolution3d_CUDA(flip(kernel), image3dSize, in_Img3D, out_Img3D);
}

} // namespace openpni::process