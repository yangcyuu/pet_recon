#pragma once
#include "../basic/Math.hpp"
#include "../math/EMStep.hpp"
#include "Foreach.hpp"
namespace openpni::process {

namespace attn {
struct AttenuationModel {
  struct _v0 {
    __PNI_CUDA_MACRO__ float operator()(
        float HUValue) const {
      return HUValue;
    }
  };
  auto v0() const { return _v0{}; }

  struct _v1 {
    __PNI_CUDA_MACRO__ float operator()(
        float HUValue) const {
      constexpr float WATER_ATTENUATION_MM_511 = 0.009687;
      return basic::FMath<float>::max(0, HUValue * 1e-3 + 1) * WATER_ATTENUATION_MM_511;
    }
  };
  auto v1() const { return _v1{}; }
};
inline constexpr AttenuationModel attn_model{};

template <typename EMCalculationMethod, FloatingPoint_c ImageValueType, typename DataViewType,
          typename AttenuationModel>
inline void cal_attn_coff(
    DataViewType __dataView, Image3DSpan<const ImageValueType> __image3dSpan_in, ImageValueType *__out_AttnFactor,
    AttenuationModel __attnModel, EMCalculationMethod __emMethod, basic::CpuMultiThread __cpuMultiThread) {
  std::unique_ptr<ImageValueType[]> imgAttnValue =
      std::make_unique_for_overwrite<ImageValueType[]>(__image3dSpan_in.geometry.totalVoxelNum());
  // reCal HU
  for_each(__image3dSpan_in.geometry.totalVoxelNum(),
           [&](std::size_t idx) { imgAttnValue[idx] = __attnModel(__image3dSpan_in.ptr[idx]); });
  // fwd to cal Coefficience
  Image3DSpan<const ImageValueType> imgAttnSpan = {__image3dSpan_in.geometry, imgAttnValue.get()};
  process::EMSum(imgAttnSpan, __image3dSpan_in.geometry.roi(), __out_AttnFactor, __dataView, __emMethod,
                 __cpuMultiThread);
  for_each(__dataView.size(), [&](std::size_t idx) {
    __out_AttnFactor[idx] = basic::FMath<ImageValueType>::fexp(-__out_AttnFactor[idx]);
  });
}
} // namespace attn

} // namespace openpni::process
