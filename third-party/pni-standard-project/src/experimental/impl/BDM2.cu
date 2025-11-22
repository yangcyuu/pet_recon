#include "BDM2.hpp"
#include "include/Exceptions.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
using namespace openpni::device::bdm2;
void d_r2s(
    uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t channelIndex,
    BDM2ConvertSingleContext d_ctx, interface::LocalSingle *d_out, uint64_t count,
    uint32_t const *d_packetInclusiveSum) {
  tools::parallel_for_each_CUDA(count, [=] __device__(uint64_t index) {
    auto *outBegin = d_out + (index == 0 ? 0 : d_packetInclusiveSum[index - 1]) * SINGLE_NUM_PER_PACKET;
    auto *outEnd = d_out + d_packetInclusiveSum[index] * SINGLE_NUM_PER_PACKET;
    if (outBegin == outEnd)
      return;
    auto *raw = d_raw + d_offset[index];
    auto length = d_length[index];
    impl_r2s(raw, length, channelIndex, d_ctx, outBegin, index);
  });
}
void d_r2s_converged(
    uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t const *d_channel,
    BDM2ConvertSingleContext const *d_ctx, uint16_t const *d_channelContextMap, interface::LocalSingle *d_out,
    uint64_t count, uint32_t const *d_packetInclusiveSum) {
  tools::parallel_for_each_CUDA(count, [=] __device__(uint64_t index) {
    auto *outBegin = d_out + (index == 0 ? 0 : d_packetInclusiveSum[index - 1]) * SINGLE_NUM_PER_PACKET;
    auto *outEnd = d_out + d_packetInclusiveSum[index] * SINGLE_NUM_PER_PACKET;
    if (outBegin == outEnd)
      return;
    auto *raw = d_raw + d_offset[index];
    auto length = d_length[index];
    auto channelIndex = d_channel[index];
    auto ctxIndex = d_channelContextMap[channelIndex];
    if (ctxIndex == 0xFFFF)
      return;
    impl_r2s(raw, length, channelIndex, d_ctx[ctxIndex], outBegin, index);
  });
}
} // namespace openpni::experimental::node::impl
