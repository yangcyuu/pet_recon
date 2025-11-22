#pragma once
#include "include/detector/BDM2.hpp"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/interface/SingleGenerator.hpp"
namespace openpni::experimental::node::impl {
struct BDM2ConvertSingleContext {
  float const *energyCoef;
  uint8_t const *positionTable;
};
__PNI_CUDA_MACRO__
inline constexpr auto energy_coef_span() {
  using namespace openpni::device::bdm2;
  return core::MDSpan<2>::create(CRYSTAL_NUM_ONE_BLOCK, BLOCK_NUM);
}
__PNI_CUDA_MACRO__
inline constexpr auto position_table_span() {
  using namespace openpni::device::bdm2;
  return core::MDSpan<3>::create(CRYSTAL_RAW_POSITION_RANGE, CRYSTAL_RAW_POSITION_RANGE, BLOCK_NUM);
}
openpni::device::bdm2::CalibrationTable read_bdm2_calibration_table(std::string filename);
__PNI_CUDA_MACRO__
inline void impl_r2s(
    uint8_t const *__raw, uint16_t __length, uint16_t __channelIndex, BDM2ConvertSingleContext __ctx,
    interface::LocalSingle *__out, uint64_t __index) {
  using namespace openpni::device::bdm2;
  if (__length != UDP_PACKET_SIZE) {
    for (int i = 0; i < SINGLE_NUM_PER_PACKET; i++) {
      __out[i] = interface::LocalSingle::invalid();
    }
    return;
  }

  const auto begin = reinterpret_cast<const DataFrameV2 *>(__raw);
  auto [__energyCoef, __positionTable] = __ctx;
  float everageEnergy = 0;
  for (int i = 0; i < SINGLE_NUM_PER_PACKET; i++) {
    const auto &curElem = begin[i];
    interface::LocalSingle single;
    single.channelIndex = __channelIndex;

    int energyIndex = 0;
    uint8_t duIndex = curElem.nHeadAndDU & 0x03;
    uint8_t positionInBlock = __positionTable[position_table_span()(curElem.X, curElem.Y, duIndex)];
    if (positionInBlock >= CRYSTAL_NUM_ONE_BLOCK) {
      single.crystalIndex = 0xFFFF;
      positionInBlock = 0;
    } else {
      uint8_t crystalU = positionInBlock % CRYSTAL_LINE + duIndex * CRYSTAL_LINE;
      uint8_t crystalV = positionInBlock / CRYSTAL_LINE;
      // crystalU = 52 - 1 - crystalU; // Invert U axis
      // crystalV = 13 - 1 - crystalV; // Invert V axis
      single.crystalIndex = crystalU + crystalV * BLOCK_NUM * CRYSTAL_LINE;
      energyIndex = energy_coef_span()(positionInBlock, duIndex);
    }

    /// Energy correction & converting
    uint16_t tempEnergy = uint16_t(curElem.Energy[0]) << 8 | curElem.Energy[1];
    single.energy = tempEnergy * __energyCoef[energyIndex];
    everageEnergy += single.energy;

    /// Time converting
    uint64_t nTimeTempPico;
    nTimeTempPico = curElem.nTime[0];
    for (unsigned ii = 1; ii <= 7; ++ii) {
      nTimeTempPico <<= 8;
      nTimeTempPico |= curElem.nTime[ii];
    }
    single.timevalue_pico = nTimeTempPico;

    __out[i] = single;
  }
}

void h_r2s(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t channelIndex,
           BDM2ConvertSingleContext h_ctx, interface::LocalSingle *h_out, uint64_t count,
           uint32_t const *h_packetInclusiveSum);
void h_r2s_converged(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length,
                     uint16_t const *h_channel, BDM2ConvertSingleContext const *h_ctx,
                     uint16_t const *h_channelContextMap, interface::LocalSingle *h_out, uint64_t count,
                     uint32_t const *h_packetInclusiveSum);
void d_r2s(uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t channelIndex,
           BDM2ConvertSingleContext d_ctx, interface::LocalSingle *d_out, uint64_t count,
           uint32_t const *d_packetInclusiveSum);
void d_r2s_converged(uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length,
                     uint16_t const *d_channel, BDM2ConvertSingleContext const *d_ctx,
                     uint16_t const *d_channelContextMap, interface::LocalSingle *d_out, uint64_t count,
                     uint32_t const *d_packetInclusiveSum);
} // namespace openpni::experimental::node::impl
