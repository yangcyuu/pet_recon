
#pragma once
#include <memory>

#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../basic/CudaUniquePtr.cuh"
#include "../basic/PetDataType.h"

#define CALL_AND_RETURN_IF_CUDA_ERR(func, err)                                                                         \
  if (err) {                                                                                                           \
    if (func) {                                                                                                        \
      func("err at file: ");                                                                                           \
      func(std::string(__FILE__));                                                                                     \
      func("err at line: ");                                                                                           \
      func(std::to_string(__LINE__));                                                                                  \
      func("err string: ");                                                                                            \
      func(cudaGetErrorString(err));                                                                                   \
    }                                                                                                                  \
    return err;                                                                                                        \
  }

#define CALL_IF(func, msg)                                                                                             \
  if (func) {                                                                                                          \
    func(msg);                                                                                                         \
  }

#define DEBUG_MODE false
#define THREADNUM 128
#define CRY_MAP_LAYER_NUM 4

#if DEBUG_MODE
#include <fstream>
#include <iostream>
static const std::string debugFileFolder = "/media/lenovo/1TB/a_new_envir/v4_coin/data/Z50100ArrayDeBugNew/";
#endif

namespace openpni::process::coin {
using namespace openpni;
namespace constants {
// 此处的能量符合率为（能量选择后的单次事件数量 / 单事件数量）
constexpr float MAX_ENERGY_RATE_EXPECTED = 1.0f;
// 此处的符合率为（符合事件数量 * 2 / 能量选择后的单次事件数量）
constexpr float MAX_PROMPT_COIN_RATE_EXPECTED = 0.5f;
constexpr float MAX_DELAY_COIN_RATE_EXPECTED = 0.25f;
} // namespace constants

struct energyCoinSelectOp {
  float2 energyWindow;

  __host__ __device__ __forceinline__ energyCoinSelectOp(
      float __energyWindowLow, float __energyWindowHigh)
      : energyWindow({__energyWindowLow, __energyWindowHigh}) {}

  __host__ __device__ __forceinline__ bool operator()(
      const basic::GlobalSingle_t &a) const {
    return (energyWindow.x <= a.energy && a.energy <= energyWindow.y);
  }
};

struct singlesTimeSortOp {
  __host__ __device__ bool operator()(
      const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) {
    return a.timeValue_pico < b.timeValue_pico;
  }
};

struct singlesCrystalPairDefaultOp {
  __host__ __device__ bool operator()(
      const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) {
    return a.globalCrystalIndex != b.globalCrystalIndex;
  }
};

namespace coin_functions {
static __global__ void countSingles_kernel(
    const basic::GlobalSingle_t *__d_singlesIn, unsigned *__d_countMap, const int __singlesNum) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __singlesNum) {
    return;
  }

  atomicAdd(__d_countMap + __d_singlesIn[tid].globalCrystalIndex, 1);
  return;
}

static __global__ void countCoins_kernel(
    const basic::CoinListmode *__d_coinsIn, unsigned *__d_countMap, const int __coinsNum) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __coinsNum) {
    return;
  }

  atomicAdd(__d_countMap + __d_coinsIn[tid].globalCrystalIndex1, 1);
  atomicAdd(__d_countMap + __d_coinsIn[tid].globalCrystalIndex2, 1);
  return;
}

static __global__ void convertLocalToGlobal_kernel(
    const basic::LocalSingle_t *__d_in, basic::GlobalSingle_t *__d_out, uint64_t __num,
    unsigned __crystalNumPrefixSum) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __num)
    return;

  __d_out[tid].energy = __d_in[tid].energy;
  __d_out[tid].globalCrystalIndex = __crystalNumPrefixSum + __d_in[tid].crystalIndex;
  __d_out[tid].timeValue_pico = __d_in[tid].timevalue_pico;
}

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

template <typename isGoodPairOp>
static __global__ void coinLocateGeneral_kernel(
    basic::GlobalSingle_t *__d_singlesIn, unsigned *__d_promptNum, unsigned *__d_delayNum,
    unsigned *__d_delayFirstIndex, const uint64_t __timeWindowPicoSec, const uint64_t __delayWindowPicoSec,
    const int __singlesNum, isGoodPairOp __goodPairOp) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= __singlesNum) {
    return;
  }

  unsigned promptNumCount = 0;
  unsigned delayNumCount = 0;

  unsigned promptEnd = tid;
  unsigned delayStartIndex = tid;

  // prompt coin locate, assume first single to search is the next single
  for (unsigned nextIdx = tid + 1; nextIdx < __singlesNum; ++nextIdx) {
    uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico;
    if (timeDiff < __timeWindowPicoSec) {
      // in time window
      // is a good pair
      if (__goodPairOp(__d_singlesIn[nextIdx], __d_singlesIn[tid])) {
        ++promptNumCount;
      }
    } else {
      // not in window
      promptEnd = nextIdx;
      break;
    }
  }
  __d_promptNum[tid] = promptNumCount;

  // delay coin locate, do not know first single positon
  bool foundFirstDelayIndex = false;
  for (unsigned nextIdx = promptEnd; nextIdx < __singlesNum; ++nextIdx) {
    // dt = t1 - delay - t0
    // to avoid overflow, we use the following logic:
    // if t1 - t0 < delay, then it is before the delayed time window
    // uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico -
    // __delayWindowPicoSec;
    uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico;

    if (timeDiff < __delayWindowPicoSec) {
      // Before delayed time window
      continue;
    } else if (timeDiff < __timeWindowPicoSec + __delayWindowPicoSec) {
      // In delayed time window
      if (__goodPairOp(__d_singlesIn[nextIdx], __d_singlesIn[tid])) {
        ++delayNumCount;
        if (!foundFirstDelayIndex) {
          // Remember the first delay event in-window.
          delayStartIndex = nextIdx;
          foundFirstDelayIndex = true;
        }
      }
    } else {
      // After delayed time window, finish this event and go to the next.
      break;
    }
  }
  __d_delayNum[tid] = delayNumCount;
  __d_delayFirstIndex[tid] = delayStartIndex;

  return;
}

template <typename __isGoodPairOp>
static __global__ void coinTimeGeneral_kernel(
    basic::GlobalSingle_t *__d_singlesIn, basic::GlobalSingle_t *__d_promptCoinSinglesOut,
    basic::GlobalSingle_t *__d_delayCoinSinglesOut, unsigned *__d_prefixedPromptNum, unsigned *__d_prefixedDelayNum,
    unsigned *__d_delayFirstIndex, uint64_t __delayTimeWindowPicoSec, const int __singlesNum,
    __isGoodPairOp __goodPairOp) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __singlesNum) {
    return;
  }

  const unsigned promptCoinNum = __d_prefixedPromptNum[tid + 1] - __d_prefixedPromptNum[tid];
  const unsigned delayCoinNum = __d_prefixedDelayNum[tid + 1] - __d_prefixedDelayNum[tid];

  const unsigned promptStartIndex = tid + 1;
  const unsigned delayStartIndex = __d_delayFirstIndex[tid];

  unsigned processedPromptNum = 0;
  unsigned processedDelayNum = 0;

  // prompt coins
  for (unsigned i = 0; processedPromptNum < promptCoinNum && i + promptStartIndex < __singlesNum; i++) {
    if (!__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i + promptStartIndex])) {
      continue;
    }
    __d_promptCoinSinglesOut[(__d_prefixedPromptNum[tid] + processedPromptNum) * 2] = __d_singlesIn[tid];
    __d_promptCoinSinglesOut[(__d_prefixedPromptNum[tid] + processedPromptNum) * 2 + 1] =
        __d_singlesIn[i + promptStartIndex];

    // new format
    // __d_promptCoinSinglesOut[(__d_prefixedPromptNum[tid] + processedPromptNum)] =
    // basic::CoinListmode{__d_singlesIn[tid].globalCrystalIndex,
    //                                                                                                   __d_singlesIn[i
    //                                                                                                   +
    //                                                                                                   promptStartIndex].globalCrystalIndex,
    //                                                                                                   -1 *
    //                                                                                                   static_cast<int16_t>(__d_singlesIn[i
    //                                                                                                   +
    //                                                                                                   promptStartIndex].timeValue_pico
    //                                                                                                   -
    //                                                                                                                             __d_singlesIn[tid].timeValue_pico)};
    ++processedPromptNum;
  }

  // delay coins
  for (unsigned i = 0; processedDelayNum < delayCoinNum && i + delayStartIndex < __singlesNum; i++) {
    if (!__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i + delayStartIndex])) {
      continue;
    }
    __d_delayCoinSinglesOut[(__d_prefixedDelayNum[tid] + processedDelayNum) * 2] = __d_singlesIn[tid];
    __d_delayCoinSinglesOut[(__d_prefixedDelayNum[tid] + processedDelayNum) * 2 + 1] =
        __d_singlesIn[i + delayStartIndex];

    // new format
    // __d_delayCoinSinglesOut[(__d_prefixedDelayNum[tid] + processedDelayNum)] =
    // basic::CoinListmode{__d_singlesIn[tid].globalCrystalIndex,
    //                                                                                                __d_singlesIn[i +
    //                                                                                                delayStartIndex].globalCrystalIndex,
    //                                                                                                -1 *
    //                                                                                                static_cast<int16_t>(__d_singlesIn[i
    //                                                                                                +
    //                                                                                                delayStartIndex].timeValue_pico
    //                                                                                                -
    //                                                                                                                          __d_singlesIn[tid].timeValue_pico
    //                                                                                                                          -
    //                                                                                                                          __delayTimeWindowPicoSec)};
    ++processedDelayNum;
  }

  return;
}

static inline cudaError_t singlesSelectByEnergyCub(
    basic::GlobalSingle_t *d_singlesIn, basic::GlobalSingle_t *d_singlesOut, unsigned *d_numSelectedOut,
    const int singlesNum, const float energyWindowLowInKeV, const float energyWindowUpInKeV, size_t &buffer_bytes,
    void *d_buffer, cudaStream_t stream = 0) {

  return cub::DeviceSelect::If(d_buffer, buffer_bytes, d_singlesIn, d_singlesOut, d_numSelectedOut, singlesNum,
                               energyCoinSelectOp(energyWindowLowInKeV, energyWindowUpInKeV), stream);
}

static inline cudaError_t singlesSortByTimeCub(
    basic::GlobalSingle_t *d_singlesIn, int singlesNum, size_t &buffer_bytes, void *d_buffer, cudaStream_t stream = 0) {
  return cub::DeviceMergeSort::StableSortKeys(d_buffer, buffer_bytes, d_singlesIn, singlesNum, singlesTimeSortOp(),
                                              stream);
}
// using __isGoodPairOp = singlesCrystalPairDefaultOp; // IGNORE
template <typename __isGoodPairOp>
static inline cudaError_t singlesTimeCoinGeneral(
    basic::GlobalSingle_t *__d_singlesIn, basic::GlobalSingle_t *__d_promptCoinsOut,
    basic::GlobalSingle_t *__d_delayCoinsOut, const uint64_t __timeWindowPicoSec, const uint64_t __delayWindowPicoSec,
    const int __singlesNum, size_t &__buffer_bytes, void *__d_buffer, const __isGoodPairOp __goodPairOp,
    unsigned &__h_promptNum, unsigned &__h_delayNum, cudaStream_t __stream = 0,
    std::function<void(std::string)> __callBackFunc = std::function<void(std::string)>()) {
  cudaError_t err = cudaSuccess;
  const unsigned prefixSize = unsigned(__singlesNum) + 1;

  unsigned *d_promptCoinNum = nullptr;
  unsigned *d_delayCoinNum = nullptr;
  unsigned *d_firstDelayIndex = nullptr;

  // get buffer in bytes and return
  if (__d_buffer == nullptr) {
    err = cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_promptCoinNum, d_promptCoinNum, prefixSize,
                                        __stream);

    __buffer_bytes += prefixSize * 3 * sizeof(unsigned);
    return err;
  }

  // find the pointer to the first element of the prefix sum
  d_promptCoinNum = reinterpret_cast<unsigned *>(__d_buffer);
  d_delayCoinNum = d_promptCoinNum + prefixSize;
  d_firstDelayIndex = d_delayCoinNum + prefixSize;

  // Check if the buffer is large enough
  void *d_cub_temp = reinterpret_cast<void *>(d_firstDelayIndex + (prefixSize));
  size_t d_cub_temp_bytes = __buffer_bytes - prefixSize * sizeof(unsigned) * 3;

  // set block number
  const unsigned blockNum = (__singlesNum - 1) / THREADNUM + 1;

  // 1. Count prompt and delay coins
  coinLocateGeneral_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, d_promptCoinNum, d_delayCoinNum,
                                                                 d_firstDelayIndex, __timeWindowPicoSec,
                                                                 __delayWindowPicoSec, __singlesNum, __goodPairOp);
  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, cudaPeekAtLastError());

#if DEBUG_MODE
  {
    std::unique_ptr<unsigned[]> h_promptCoinNum = std::make_unique<unsigned[]>(prefixSize);
    std::unique_ptr<unsigned[]> h_delayCoinNum = std::make_unique<unsigned[]>(prefixSize);
    err = cudaMemcpyAsync(h_promptCoinNum.get(), d_promptCoinNum, prefixSize * sizeof(unsigned), cudaMemcpyDeviceToHost,
                          __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    err = cudaMemcpyAsync(h_delayCoinNum.get(), d_delayCoinNum, prefixSize * sizeof(unsigned), cudaMemcpyDeviceToHost,
                          __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 输出prompt符合数据到单独文件
    std::ofstream promptDebugFile(debugFileFolder + "debug_prompt_counts.txt");
    if (promptDebugFile.is_open()) {
      promptDebugFile << "=== Prompt Coin Count Debug ===" << std::endl;
      promptDebugFile << "Singles Number: " << __singlesNum << std::endl;
      promptDebugFile << "Prefix Size: " << prefixSize << std::endl;
      promptDebugFile << std::endl;

      // 输出每个单事件对应的prompt符合数量
      for (unsigned i = 0; i < prefixSize - 1; ++i) {
        promptDebugFile << "Single[" << i << "]: promptCount=" << h_promptCoinNum[i] << std::endl;
      }

      promptDebugFile << std::endl;
      promptDebugFile << "=== Prefix Sum Results ===" << std::endl;
      promptDebugFile << "Total Prompt Coins: " << h_promptCoinNum[prefixSize - 1] << std::endl;

      promptDebugFile.close();
    } else {
      std::cerr << "Failed to open debug_prompt_counts.txt for writing." << std::endl;
    }

    // 输出delay符合数据到单独文件
    std::ofstream delayDebugFile(debugFileFolder + "debug_delay_counts.txt");
    if (delayDebugFile.is_open()) {
      delayDebugFile << "=== Delay Coin Count Debug ===" << std::endl;
      delayDebugFile << "Singles Number: " << __singlesNum << std::endl;
      delayDebugFile << "Prefix Size: " << prefixSize << std::endl;
      delayDebugFile << std::endl;

      // 输出每个单事件对应的delay符合数量
      for (unsigned i = 0; i < prefixSize - 1; ++i) {
        delayDebugFile << "Single[" << i << "]: delayCount=" << h_delayCoinNum[i] << std::endl;
      }

      delayDebugFile << std::endl;
      delayDebugFile << "=== Prefix Sum Results ===" << std::endl;
      delayDebugFile << "Total Delay Coins: " << h_delayCoinNum[prefixSize - 1] << std::endl;

      delayDebugFile.close();
    } else {
      std::cerr << "Failed to open debug_delay_counts.txt for writing." << std::endl;
    }
  }
#endif

  // 2. Scan the prefix sum of prompt and delay coins
  err = cub::DeviceScan::ExclusiveSum(d_cub_temp, d_cub_temp_bytes, d_promptCoinNum, d_promptCoinNum, prefixSize,
                                      __stream);
  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

  err =
      cub::DeviceScan::ExclusiveSum(d_cub_temp, d_cub_temp_bytes, d_delayCoinNum, d_delayCoinNum, prefixSize, __stream);
  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

  // 3. Copy the last element of the prefix sum to the last element of the output arrays
  basic::cuda::cuda_pinned_unique_ptr<unsigned> h_promptNum = basic::cuda::make_cuda_pinned_unique_ptr<unsigned>(1);

  basic::cuda::cuda_pinned_unique_ptr<unsigned> h_delayNum = basic::cuda::make_cuda_pinned_unique_ptr<unsigned>(1);

  err = cudaMemcpyAsync(h_promptNum.get(), d_promptCoinNum + prefixSize - 1, sizeof(unsigned), cudaMemcpyDeviceToHost,
                        __stream);
  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

  err = cudaMemcpyAsync(h_delayNum.get(), d_delayCoinNum + prefixSize - 1, sizeof(unsigned), cudaMemcpyDeviceToHost,
                        __stream);

  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

  // Check for unrealistic coin rates that might indicate errors
  const float promptCoinRate =
      (__singlesNum > 0) ? static_cast<float>(h_promptNum.at(0) * 2) / static_cast<float>(__singlesNum) : 0.0f;
  const float delayCoinRate =
      (__singlesNum > 0) ? static_cast<float>(h_delayNum.at(0) * 2) / static_cast<float>(__singlesNum) : 0.0f;

  if (promptCoinRate > constants::MAX_PROMPT_COIN_RATE_EXPECTED) {
    if (__callBackFunc) {
      __callBackFunc("Error: prompt coin num / energy selected singles (" + std::to_string(promptCoinRate * 100.0f) +
                     "%) exceeds expected maximum (" +
                     std::to_string(constants::MAX_PROMPT_COIN_RATE_EXPECTED * 100.0f) + "%)");
    }
    return cudaErrorInvalidValue;
  }

  if (delayCoinRate > constants::MAX_DELAY_COIN_RATE_EXPECTED) {
    if (__callBackFunc) {
      __callBackFunc("Error: Delay coin num / energy selected singles(" + std::to_string(delayCoinRate * 100.0f) +
                     "%) exceeds expected maximum (" +
                     std::to_string(constants::MAX_DELAY_COIN_RATE_EXPECTED * 100.0f) + "%)");
    }
    return cudaErrorInvalidValue;
  }

  coinTimeGeneral_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, __d_promptCoinsOut, __d_delayCoinsOut,
                                                               d_promptCoinNum, d_delayCoinNum, d_firstDelayIndex,
                                                               __delayWindowPicoSec, __singlesNum, __goodPairOp);

  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, cudaPeekAtLastError());

  __h_promptNum = h_promptNum.at(0);
  __h_delayNum = h_delayNum.at(0);

  return err;
}
} // namespace coin_functions

class s2cHandle {
public:
  s2cHandle() {};
  s2cHandle(const s2cHandle &) = delete;
  s2cHandle(s2cHandle &&) = delete;

  // Disable assignment operators
  s2cHandle &operator=(const s2cHandle &) = delete;
  s2cHandle &operator=(s2cHandle &&) = delete;

  ~s2cHandle() {};

private:
  using LogCallback = std::function<void(std::string)>;
  cudaError_t countCryMapSingles(
      const basic::GlobalSingle_t *__d_singlesIn, unsigned *__d_countMap, const unsigned __singlesNum,
      cudaStream_t __stream = 0) {
    const unsigned blockNum = (__singlesNum - 1) / THREADNUM + 1;

    coin_functions::countSingles_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, __d_countMap,
                                                                              __singlesNum);
    cudaStreamSynchronize(__stream);
    return cudaPeekAtLastError();
  }

  cudaError_t countCryMapCoins(
      const basic::CoinListmode *__d_coinsIn, unsigned *__d_countMap, const unsigned __coinsNum,
      cudaStream_t __stream = 0) {
    const unsigned blockNum = (__coinsNum - 1) / THREADNUM + 1;

    coin_functions::countCoins_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_coinsIn, __d_countMap, __coinsNum);
    cudaStreamSynchronize(__stream);
    return cudaPeekAtLastError();
  }

private:
  static constexpr int m_headerUnsignedNum = 32; // 128字节对齐，使用4字节的unsigned

  template <bool NEED_SELECT_ENERGY = true, bool NEED_SORT_TIME = true>
  std::size_t getBufferNeededBytesNew(
      const int __singlesNum) const {
    std::size_t bufferSizeTotal = 0;
    bufferSizeTotal += m_headerUnsignedNum * sizeof(unsigned); // 为了128字节对齐，目前仅使用了4字节

    // 中间变量：全局单次事件
    bufferSizeTotal += __singlesNum * sizeof(basic::GlobalSingle_t);

    std::size_t bufferSize = 0; // 内部公用内存大小
    if (NEED_SELECT_ENERGY) {
      std::size_t bufferTemp = 0;
      basic::GlobalSingle_t *globalSinglePtr = nullptr;
      unsigned *unsignedPtr = nullptr;
      coin_functions::singlesSelectByEnergyCub(globalSinglePtr, globalSinglePtr, unsignedPtr, __singlesNum, 0.0f, 0.0f,
                                               bufferTemp, nullptr);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    if (NEED_SORT_TIME) {
      std::size_t bufferTemp = 0;
      basic::GlobalSingle_t *globalSinglePtr = nullptr;
      coin_functions::singlesSortByTimeCub(globalSinglePtr, __singlesNum, bufferTemp, nullptr);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    {
      std::size_t bufferTemp = 0;
      basic::GlobalSingle_t *globalSinglePtr = nullptr;
      unsigned *unsignedPtr = nullptr;
      coin_functions::singlesTimeCoinGeneral(globalSinglePtr, globalSinglePtr, globalSinglePtr, 0, 0, __singlesNum,
                                             bufferTemp, nullptr, singlesCrystalPairDefaultOp(), *unsignedPtr,
                                             *unsignedPtr);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    bufferSizeTotal += bufferSize;
    return bufferSizeTotal;
  }

public:
  template <typename __isGoodPairOp, bool NEED_SELECT_ENERGY = true, bool NEED_SORT_TIME = true>
  inline cudaError_t exc(
      basic::GlobalSingle_t *__d_singlesIn, basic::GlobalSingle_t *__d_promptCoinsOut,
      basic::GlobalSingle_t *__d_delayCoinsOut, unsigned *__d_cryMap, const float __energyWindowLow,
      const float __energyWindowHigh, const uint64_t __timeWindowPicoSec, const uint64_t __delayWindowPicoSec,
      const int __cryNum, const int __singlesNum, unsigned &__promptCoinsNum, unsigned &__delayCoinsNum,
      std::size_t &__bufferSize, void *__d_buffer = nullptr, const int __deviceId = 0, cudaStream_t __stream = 0,
      __isGoodPairOp __goodPairOp = singlesCrystalPairDefaultOp(), LogCallback __callBackFunc = LogCallback()) {
    // 0. 若__d_buffer为nullptr，则获取所需的buffer大小并返回
    if (__d_buffer == nullptr) {
      __bufferSize = getBufferNeededBytesNew<NEED_SELECT_ENERGY, NEED_SORT_TIME>(__singlesNum);
      return cudaSuccess;
    }

    if (__bufferSize < getBufferNeededBytesNew<NEED_SELECT_ENERGY, NEED_SORT_TIME>(__singlesNum)) {
      CALL_IF(__callBackFunc, "Error: Provided buffer size is smaller than required.");
      return cudaErrorInvalidValue;
    }

    // 1. 设置设备
    cudaError_t err = cudaSuccess;
    int originalDeviceId = 0;
    err = cudaGetDevice(&originalDeviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    err = cudaSetDevice(__deviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 2. 初始化必要的变量
    unsigned *d_validSinglesNum = nullptr;           // 有效的singles数量指针
    basic::GlobalSingle_t *d_validSingles = nullptr; // 有效的singles指针
    char *d_buffer = nullptr;                        // 中间buffer指针，初始化为nullptr
    size_t bufferSize = __bufferSize;                // 中间buffer的大小
    {
      // 计算buffer的起始位置
      uint8_t *bufferPtr = static_cast<uint8_t *>(__d_buffer);
      d_validSinglesNum = reinterpret_cast<unsigned *>(bufferPtr);
      bufferPtr += m_headerUnsignedNum * sizeof(unsigned);
      bufferSize -= m_headerUnsignedNum * sizeof(unsigned);

      d_validSingles = reinterpret_cast<basic::GlobalSingle_t *>(bufferPtr);
      bufferPtr += __singlesNum * sizeof(basic::GlobalSingle_t);
      bufferSize -= __singlesNum * sizeof(basic::GlobalSingle_t);

      // 中间buffer的剩余部分用于排序和计数
      d_buffer = reinterpret_cast<char *>(bufferPtr);
    }

    basic::cuda::cuda_pinned_unique_ptr<unsigned> h_validSinglesNum =
        basic::cuda::make_cuda_pinned_unique_ptr<unsigned>(1);

    // 3. 若需要选择能量，则进行能量选择
    if (NEED_SELECT_ENERGY) {
      // 使用cub的DeviceSelect进行能量选择
      err = coin_functions::singlesSelectByEnergyCub(__d_singlesIn, d_validSingles, d_validSinglesNum, __singlesNum,
                                                     __energyWindowLow, __energyWindowHigh, bufferSize, d_buffer,
                                                     __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

      err = cudaMemcpyAsync(h_validSinglesNum.get(), d_validSinglesNum, sizeof(unsigned), cudaMemcpyDeviceToHost,
                            __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
      const float energyRate =
          __singlesNum > 0 ? static_cast<float>(h_validSinglesNum.at(0)) / static_cast<float>(__singlesNum) : 0.0f;

      CALL_IF(__callBackFunc,
              ("Energy selected singles num / input singles num: " + std::to_string(energyRate * 100.0f) + "%"));

      if (energyRate > constants::MAX_ENERGY_RATE_EXPECTED) {
        if (__callBackFunc) {
          __callBackFunc("Error: Energy selected singles num / input singles num (" +
                         std::to_string(energyRate * 100.0f) + "%) is more than expected minimum (" +
                         std::to_string(constants::MAX_ENERGY_RATE_EXPECTED * 100.0f) + "%)");
        }
        return cudaErrorInvalidValue;
      }
    } else {
      // 如果不需要能量选择，但需要其他处理，则直接将输入的singles数量作为有效singles数量
      h_validSinglesNum[0] = __singlesNum;

      // 直接将输入的singles复制到有效singles
      err = cudaMemcpyAsync(d_validSingles, __d_singlesIn, __singlesNum * sizeof(basic::GlobalSingle_t),
                            cudaMemcpyDeviceToDevice, __stream);

      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

#if DEBUG_MODE
    // {
    //     cudaDeviceSynchronize();
    //     std::cout << "Valid singles num: " << h_validSinglesNum.at(0) << std::endl;
    //     std::cout << "Valid singles rate: " << (static_cast<float>(h_validSinglesNum.at(0)) / __singlesNum) * 100.0f
    //               << "%" << std::endl;

    //     std::unique_ptr<basic::GlobalSingle_t[]> h_debugSingles =
    //         std::make_unique<basic::GlobalSingle_t[]>(h_validSinglesNum.at(0));

    //     err = cudaMemcpy(h_debugSingles.get(),
    //                      d_validSingles,
    //                      h_validSinglesNum.at(0) * sizeof(basic::GlobalSingle_t),
    //                      cudaMemcpyDeviceToHost);
    //     CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    //     std::ofstream debugFile(debugFileFolder + "debug_valid_singles.txt");
    //     if (debugFile.is_open())
    //     {
    //         for (unsigned i = 0; i < h_validSinglesNum.at(0); i++)
    //         {
    //             debugFile << "Single " << i << ": (" << h_debugSingles[i].globalCrystalIndex << ", "
    //                       << h_debugSingles[i].timeValue_pico << " ps), "
    //                       << h_debugSingles[i].energy << " kev)" << std::endl;
    //         }
    //         debugFile.close();
    //     }
    //     else
    //     {
    //         std::cerr << "Error opening debug file for writing." << std::endl;
    //     }
    // }
#endif

    // 4. 若需要排序时间，则进行时间排序
    if (NEED_SORT_TIME) {
      // 使用cub在中间变量d_validSingles上进行时间排序
      err =
          coin_functions::singlesSortByTimeCub(d_validSingles, h_validSinglesNum.at(0), bufferSize, d_buffer, __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

    //  5 符合计算
    err = coin_functions::singlesTimeCoinGeneral(d_validSingles, __d_promptCoinsOut, __d_delayCoinsOut,
                                                 __timeWindowPicoSec, __delayWindowPicoSec, h_validSinglesNum.at(0),
                                                 bufferSize, d_buffer, __goodPairOp, __promptCoinsNum, __delayCoinsNum,
                                                 __stream, __callBackFunc);

    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    const unsigned promptCoinsNumLocal = __promptCoinsNum;
    const unsigned delayCoinsNumLocal = __delayCoinsNum;

    CALL_IF(__callBackFunc, ("Prompt coins num: " + std::to_string(promptCoinsNumLocal)));
    CALL_IF(__callBackFunc,
            ("Prompt coins rate: " +
             std::to_string(static_cast<float>(promptCoinsNumLocal * 2) / __singlesNum * 100.0f) + "%"));
    CALL_IF(__callBackFunc, ("Delay coins num: " + std::to_string(delayCoinsNumLocal)));
    CALL_IF(__callBackFunc, ("Delay coins rate: " +
                             std::to_string(static_cast<float>(delayCoinsNumLocal * 2) / __singlesNum * 100.0f) + "%"));
#if DEBUG_MODE
    // txt输出完整符合文件
    // {
    //     cudaDeviceSynchronize();
    //     std::unique_ptr<basic::GlobalSingle_t[]> h_debugSingles =
    //     std::make_unique<basic::GlobalSingle_t[]>(promptCoinsNumLocal * 2); err = cudaMemcpy(h_debugSingles.get(),
    //                      __d_promptCoinsOut,
    //                      promptCoinsNumLocal * 2 * sizeof(basic::GlobalSingle_t),
    //                      cudaMemcpyDeviceToHost);
    //     CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    //     std::ofstream debugFilePrompt(debugFileFolder + "debug_prompt_coins.txt");
    //     if (debugFilePrompt.is_open())
    //     {
    //         for (unsigned i = 0; i < promptCoinsNumLocal * 2; i++)
    //         {
    //             debugFilePrompt << "Coin " << i << ": (" << h_debugSingles[i].globalCrystalIndex << ", "
    //                             << h_debugSingles[i].timeValue_pico << " ps), "
    //                             << h_debugSingles[i].energy << " kev)" << std::endl;
    //         }
    //         debugFilePrompt.close();
    //     }

    //     h_debugSingles = std::make_unique<basic::GlobalSingle_t[]>(delayCoinsNumLocal * 2);
    //     err = cudaMemcpy(h_debugSingles.get(),
    //                      __d_delayCoinsOut,
    //                      delayCoinsNumLocal * 2 * sizeof(basic::GlobalSingle_t),
    //                      cudaMemcpyDeviceToHost);
    //     CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    //     cudaDeviceSynchronize();

    //     std::ofstream debugFileDelay(debugFileFolder + "debug_delay_coins.txt");
    //     if (debugFileDelay.is_open())
    //     {
    //         for (unsigned i = 0; i < delayCoinsNumLocal * 2; i++)
    //         {
    //             debugFileDelay << "Coin " << i << ": (" << h_debugSingles[i].globalCrystalIndex << ", "
    //                            << h_debugSingles[i].timeValue_pico << " ps), "
    //                            << h_debugSingles[i].energy << " kev)" << std::endl;
    //         }
    //         debugFileDelay.close();
    //     }
    // }
#endif

    // 6. 统计cryMap
    // 6.1 计算能量窗前的有效singles数量， 若无需能量选择，则不统计（因为所有singles都有效）
    if (NEED_SELECT_ENERGY) {
      err = countCryMapSingles(__d_singlesIn, __d_cryMap, __singlesNum, __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

    // 6.2 计算能量窗后的有效singles数量，若无需能量选择，则统计所有singles（因为所有singles都有效）
    err = countCryMapSingles(d_validSingles, __d_cryMap + __cryNum, h_validSinglesNum.at(0), __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 6.3 计算prompt coins数量
    err = countCryMapSingles(__d_promptCoinsOut, __d_cryMap + __cryNum * 2, promptCoinsNumLocal * 2, __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    err = countCryMapSingles(__d_delayCoinsOut, __d_cryMap + __cryNum * 3, delayCoinsNumLocal * 2, __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 5. reset deviceId
    err = cudaSetDevice(originalDeviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    cudaStreamSynchronize(__stream);

#if DEBUG_MODE

    {
      cudaDeviceSynchronize();
      std::unique_ptr<unsigned[]> h_debugCryMap = std::make_unique<unsigned[]>(__cryNum * 4);
      err = cudaMemcpy(h_debugCryMap.get(), __d_cryMap, __cryNum * 4 * sizeof(unsigned), cudaMemcpyDeviceToHost);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
      std::ofstream debugFileCryMap(debugFileFolder + "debug_cry_map.txt");
      if (debugFileCryMap.is_open()) {
        for (unsigned i = 0; i < __cryNum * 4; i++) {
          debugFileCryMap << "CryMap[" << i << "]: " << h_debugCryMap[i] << std::endl;
        }
        debugFileCryMap.close();
      } else {
        std::cerr << "Error opening debug file for writing." << std::endl;
      }
    }

#endif

    return cudaSuccess;
  }

  static inline cudaError_t listModeCompress(
      basic::GlobalSingle_t *__d_coinSinglesIn, basic::CoinListmode *__d_compressedCoinOut,
      const unsigned __coinSinglesNum, const uint64_t __delayTimePicoSec = 0, const int __deviceId = 0,
      cudaStream_t __stream = 0, LogCallback __callBackFunc = LogCallback()) {
    int originalDeviceId = 0;
    cudaGetDevice(&originalDeviceId);
    cudaSetDevice(__deviceId);

    const unsigned blockNum = (__coinSinglesNum - 1) / THREADNUM + 1;

    coin_functions::compressCoin_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_coinSinglesIn, __d_compressedCoinOut,
                                                                              __coinSinglesNum, __delayTimePicoSec);

    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, cudaPeekAtLastError());

    cudaSetDevice(originalDeviceId);
    return cudaSuccess;
  }

private:
  // basic::cuda::cuda_pinned_unique_ptr<unsigned> h_validSinglesNum =
  //     basic::cuda::make_cuda_pinned_unique_ptr<unsigned>(1);
};
} // namespace openpni::process::coin

#undef THREADNUM
#undef CALL_AND_RETURN_IF_CUDA_ERR
#undef CRY_MAP_LAYER_NUM
#undef CALL_IF
#undef DEBUG_MODE

#endif
