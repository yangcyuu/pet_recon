#include "MichFactors.hpp"
#include "MichScatterImpl.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
void d_calNormFactorsAll(
    core::MichDefine __mich, std::span<core::MichStandardEvent const> events, float *__output, float const *__cryFct,
    float const *__blockFctA, float const *__blockFctT, float const *__planeFct, float const *__radialFct,
    float const *__interferenceFct, float const *__dtComponent, FactorBitMask __factorBitMask) {
  const auto indexConverter = core::IndexConverter::create(__mich);
  tools::parallel_for_each_CUDA(events.size(), [=, d_events = events.data()] __device__(std::size_t idx) {
    auto lorIndex = indexConverter.getLORIDFromRectangleID(d_events[idx].crystal1, d_events[idx].crystal2);
    __output[idx] = impl::calNormFactorsAll(__mich, lorIndex, __cryFct, __blockFctA, __blockFctT, __planeFct,
                                            __radialFct, __interferenceFct, __dtComponent, __factorBitMask);
  });
}
void d_getDScatterFactorsBatch(
    float *__outFct, const float *__sssValue, std::span<core::MichStandardEvent const> events,
    core::MichDefine __michDefine) {
  const auto coverter = core::IndexConverter::create(__michDefine);
  tools::parallel_for_each_CUDA(events.size(), [=, d_events = events.data()] __device__(size_t idx) {
    auto lorIndex = coverter.getLORIDFromRectangleID(d_events[idx].crystal1, d_events[idx].crystal2);
    __outFct[idx] = __sssValue[lorIndex];
  });
}
void d_getDScatterFactorsBatchTOF(
    float *__outFct, const float *__sssTOFTable, const core::CrystalGeom *__d_crystalGeometry,
    const core::CrystalGeom *__d_dsCrystalGeometry, std::span<core::MichStandardEvent const> __events,
    core::MichDefine __michDefine, core::MichDefine __dsmichDefine, float __tofBinWidth_ns, int __tofBinNum) {
  const auto michInfo = MichInfoHub(__michDefine);
  const auto dsMichInfo = MichInfoHub(__dsmichDefine);
  const auto coverter = core::IndexConverter::create(__michDefine);

  auto fullSliceDsLorNum = dsMichInfo.getBinNum() * dsMichInfo.getViewNum() * michInfo.getSliceNum();

  tools::parallel_for_each_CUDA(__events.size(), [=, d_events = __events.data()] __device__(size_t idx) {
    int tofbinIdx = int(floor((d_events[idx].tof *1e-3)/ __tofBinWidth_ns + 0.5)) + int(__tofBinNum / 2);
    if(tofbinIdx <0 || tofbinIdx >= __tofBinNum){
      //printf("tofbinIdx out of range: %d for tofBinNum %d,at tof %f,tofBinWidth = %f\n", tofbinIdx,__tofBinNum, d_events[idx].tof*1e-3, __tofBinWidth_ns);
      __outFct[idx] = 0.0f;
      return;
    }
    auto lorIndex = coverter.getLORIDFromRectangleID(d_events[idx].crystal1, d_events[idx].crystal2);
   float value = get2DInterpolationUpsamplingValue(__sssTOFTable + tofbinIdx * fullSliceDsLorNum, lorIndex, __michDefine,
                                          __dsmichDefine, __d_crystalGeometry, __d_dsCrystalGeometry);
    __outFct[idx] = value;
  });
}
} // namespace openpni::experimental::node::impl
