#pragma once
#include "BDM2_impl.hpp"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
namespace openpni::device::bdm2 {
void r2s_cuda_impl(const void *d_raw, const device::PacketPositionInfo *d_position, uint64_t count,
                   basic::LocalSingle_t *d_out, const float *d_energyCoef, const uint8_t *d_positionTable);
} // namespace openpni::device::bdm2
#endif