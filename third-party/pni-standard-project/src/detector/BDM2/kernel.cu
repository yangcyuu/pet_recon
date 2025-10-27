#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
#include "kernel.hpp"
namespace openpni::device::bdm2 {
__global__ void kernel_r2s_cuda(
    const void *d_raw, const device::PacketPositionInfo *d_position, uint64_t count, basic::LocalSingle_t *d_out,
    const float *d_energyCoef, const uint8_t *d_positionTable) {
  const auto pktIndex = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pktIndex >= count)
    return;
  const auto &position = d_position[pktIndex];
  if (position.length != UDP_PACKET_SIZE)
    return;

  const auto begin = reinterpret_cast<const DataFrameV2 *>(static_cast<const char *>(d_raw) + position.offset);
  for (int i = 0; i < SINGLE_NUM_PER_PACKET; i++) {
    const auto &curElem = begin[i];
    basic::LocalSingle_t single;
    int energyIndex = 0;

    uint8_t duIndex = curElem.nHeadAndDU & 0x03;
    uint8_t positionInBlock = d_positionTable[duIndex * 256 * 256 + curElem.Y * 256 + curElem.X];
    if (positionInBlock >= CRYSTAL_NUM_ONE_BLOCK) {
      single.crystalIndex = 0xFFFF;
    } else {
      uint8_t crystalU = positionInBlock % CRYSTAL_LINE + duIndex * CRYSTAL_LINE;
      uint8_t crystalV = positionInBlock / CRYSTAL_LINE;
      // crystalU = 52 - 1 - crystalU; // Invert U axis
      // crystalV = 13 - 1 - crystalV; // Invert V axis
      single.crystalIndex = crystalU + crystalV * BLOCK_NUM * CRYSTAL_LINE;
      energyIndex = positionInBlock + duIndex * CRYSTAL_NUM_ONE_BLOCK;
    }

    /// Energy correction & converting
    uint16_t tempEnergy = uint16_t(curElem.Energy[0]) << 8 | curElem.Energy[1];
    single.energy = tempEnergy * d_energyCoef[energyIndex];

    /// Time converting
    uint64_t nTimeTempPico;
    nTimeTempPico = curElem.nTime[0];
    for (unsigned ii = 1; ii <= 7; ++ii) {
      nTimeTempPico <<= 8;
      nTimeTempPico |= curElem.nTime[ii];
    }
    single.timevalue_pico = nTimeTempPico;

    d_out[i + pktIndex * MAX_SINGLE_NUM_PER_PACKET] = single;
  }
}
void r2s_cuda_impl(
    const void *d_raw, const device::PacketPositionInfo *d_position, uint64_t count, basic::LocalSingle_t *d_out,
    const float *d_energyCoef, const uint8_t *d_positionTable) {
  kernel_r2s_cuda<<<(count + 255) / 256, 256>>>(d_raw, d_position, count, d_out, d_energyCoef, d_positionTable);
}
} // namespace openpni::device::bdm2

#endif