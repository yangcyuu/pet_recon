#pragma once
#include <memory>
// #include "../Define.h"
#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"
#include "../basic/PetDataType.h"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace openpni::process::r2cTools {
namespace kernel {
static __global__ void compressCoin_kernel(
    const basic::GlobalSingle_t *__d_coinSinglesIn, basic::CoinListmode *__d_compressedCoinOut,
    const unsigned __coinNum, const uint64_t __delayTimePicoSec) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __coinNum) {
    return;
  }

  // Compress the coin singles into the CoinListmode format
  uint64_t timeDiff =
      __d_coinSinglesIn[tid * 2 + 1].timeValue_pico - __d_coinSinglesIn[tid * 2].timeValue_pico - __delayTimePicoSec;

  // Clamp to int16_t range and negate
  int16_t clampedTimeDiff = int16_t(0) - static_cast<int16_t>(timeDiff);

  // Store the compressed coin
  __d_compressedCoinOut[tid] = basic::CoinListmode{__d_coinSinglesIn[tid * 2].globalCrystalIndex,
                                                   __d_coinSinglesIn[tid * 2 + 1].globalCrystalIndex, clampedTimeDiff};
}
} // namespace kernel

static inline cudaError_t compressCoin(
    const basic::GlobalSingle_t *__d_coinSinglesIn, basic::CoinListmode *__d_compressedCoinOut,
    const unsigned __coinNum, const uint64_t __delayTimePicoSec, cudaStream_t __stream = cudaStreamDefault) {
  if (__coinNum == 0) {
    return cudaSuccess;
  }

  const unsigned blockSize = 256;
  const unsigned gridSize = (unsigned)ceil(static_cast<float>(__coinNum) / blockSize);

  kernel::compressCoin_kernel<<<gridSize, blockSize, 0, __stream>>>(__d_coinSinglesIn, __d_compressedCoinOut, __coinNum,
                                                                    __delayTimePicoSec);

  cudaStreamSynchronize(__stream);
  return cudaGetLastError();
}
} // namespace openpni::process::r2cTools

#endif