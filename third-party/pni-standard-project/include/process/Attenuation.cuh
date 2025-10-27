#pragma once
#include "../basic/Point.hpp"
#include "../math/EMStep.cuh"
#include "Attenuation.hpp"
#include "Foreach.cuh"
namespace openpni::process {
namespace attn {
template <typename EMCalculationMethod, FloatingPoint_c ImageValueType, typename DataViewType,
          typename AttenuationModel>
inline void cal_attn_coff_CUDA(
    DataViewType __dataView, Image3DSpan<const ImageValueType> __d_image3dSpan_in, ImageValueType *__d_out_AttnFactor,
    AttenuationModel __attnModel, EMCalculationMethod __emMethod, cudaStream_t __stream = cudaStreamDefault) {
  auto d_imgAttnValue =
      openpni::make_cuda_sync_ptr<ImageValueType>(std::size_t(__d_image3dSpan_in.geometry.totalVoxelNum()));
  for_each_CUDA(
      __d_image3dSpan_in.geometry.totalVoxelNum(),
      [d_imgAttnValue = d_imgAttnValue.get(), __attnModel, d_imgIn = __d_image3dSpan_in.ptr] __device__(
          std::size_t idx) { d_imgAttnValue[idx] = __attnModel(d_imgIn[idx]); },
      __stream);
  Image3DIOSpan<ImageValueType> d_attnImg3DIOSpan = {__d_image3dSpan_in.geometry, d_imgAttnValue.get(),
                                                     __d_out_AttnFactor};
  process::EMSum_CUDA<EMCalculationMethod, ImageValueType, DataViewType>(
      d_attnImg3DIOSpan, d_attnImg3DIOSpan.geometry.roi(), __dataView, __emMethod);
  for_each_CUDA(
      __dataView.size(),
      [__d_out_AttnFactor] __device__(std::size_t idx) {
        __d_out_AttnFactor[idx] = basic::FMath<ImageValueType>::fexp(-__d_out_AttnFactor[idx]);
      },
      __stream);
}

} // namespace attn

} // namespace openpni::process
