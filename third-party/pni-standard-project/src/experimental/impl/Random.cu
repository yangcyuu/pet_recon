#include "Random.h"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
void dumpFactorsAsDMich(
    float const *factors, openpni::experimental::core::MichDefine mich, float *d_out, int minSectorDifference) {
  const auto indexConverter = core::IndexConverter::create(mich);
  const auto michSize = core::MichInfoHub::create(mich).getMichSize();
  tools::parallel_for_each_CUDA(michSize, [result_ptr = d_out, factors_ptr = factors, mich = mich,
                                           minSectorDifference = minSectorDifference,
                                           indexConverter] __device__(std::size_t lor) {
    auto [rid1, rid2] = indexConverter.getCrystalIDFromLORID(lor);
    result_ptr[lor] = get_factor(rid1, rid2, factors_ptr, mich, minSectorDifference);
  });
}

void getDRandomFactors(
    std::span<core::MichStandardEvent const> d_events, float const *factors, core::MichDefine mich,
    int minSectorDifference, float *d_out) {
  const auto indexConverter = core::IndexConverter::create(mich);

  tools::parallel_for_each_CUDA(d_events.size(),
                                [factors_ptr = factors, mich = mich, minSectorDifference = minSectorDifference,
                                 event_ptr = d_events.data(), out_ptr = d_out] __device__(std::size_t i) {
                                  const auto &event = event_ptr[i];
                                  out_ptr[i] = impl::get_factor(event, factors_ptr, mich, minSectorDifference);
                                });
}

} // namespace openpni::experimental::node::impl
