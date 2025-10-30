#pragma once
#include "../process/Foreach.cuh"
#include "KEMkernelGenerate.hpp"

namespace openpni::example::knn {
template <typename ImageValueType>
inline void knnConvolution3d(
    const _KNNConvolution<ImageValueType> &d_knnKernel, openpni::Image3DIOSpan<ImageValueType> d_convImg,
    cudaStream_t __stream = cudaStreamDefault) {
  for_each_CUDA(
      d_convImg.geometry.totalVoxelNum(),
      [&](std::size_t imgIndx) {
        d_convImg.ptr_out[imgIndx] = d_knnKernel.convolution_impl(d_convImg.ptr_in, imgIndx);
      },
      __stream);
}
template <typename ImageValueType>
inline void knnDeconvolution3d(
    const _KNNConvolution<ImageValueType> &d_knnKernel, openpni::Image3DIOSpan<ImageValueType> d_convImg,
    cudaStream_t __stream = cudaStreamDefault) {

  for_each_CUDA(
      d_convImg.geometry.totalVoxelNum() * d_knnKernel.__kernelNumber,
      [&](std::size_t convIdx) {
        std::size_t imgIdx = convIdx / d_knnKernel.__kernelNumber;
        d_convImg.ptr_out[d_knnKernel.__KNNto[convIdx]] +=
            d_knnKernel.deconvolution_impl(d_convImg.ptr_in, imgIdx, convIdx);
      },
      __stream);
}
} // namespace openpni::example::knn
