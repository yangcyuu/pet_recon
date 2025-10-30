#include "LORBatch.h"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
void d_fillLORBatch(
    std::size_t begin, std::size_t end, int binCut, int subsetNum, int currentSubset,
    std::span<core::Vector<int64_t, 2> const> d_ringPair, core::MichDefine mich, std::size_t *d_out) {
  const auto michInfoHub = core::MichInfoHub::create(mich);
  const auto indexConverter = core::IndexConverter::create(mich);
  core::MDSpan<3> span = core::MDSpan<3>::create(
      michInfoHub.getBinNum() - 2 * binCut,
      core::mich::calViewNumInSubset(mich.polygon, mich.detector, subsetNum, currentSubset), d_ringPair.size());
  tools::parallel_for_each_CUDA(
      begin, end,
      [binCut = binCut, subsetNum = subsetNum, currentSubset = currentSubset, ptr_ringpair = d_ringPair.data(),
       mich = mich, ptr_out = d_out, span = span, indexConverter, begin] __device__(std::size_t index) {
        const auto &[binCutted, viewInSubset, ringPairIndex] = span.toIndex(index);
        const int binIndex = binCutted + binCut;
        const int viewIndex = viewInSubset * subsetNum + currentSubset;
        const auto &[ring1, ring2] = ptr_ringpair[ringPairIndex];
        const int sliceIndex = indexConverter.getSliceFromRing1Ring2(ring1, ring2);
        const std::size_t lorId = indexConverter.getLORFromBVS(binIndex, viewIndex, sliceIndex);
        ptr_out[index - begin] = lorId;
      });
}
} // namespace openpni::experimental::node::impl
