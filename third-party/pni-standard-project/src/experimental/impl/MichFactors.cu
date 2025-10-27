#include "MichFactors.hpp"
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
} // namespace openpni::experimental::node::impl
