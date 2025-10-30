#pragma once
#include <functional>
#include <memory>
// #include "../Define.h"
#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"
#include "../basic/PetDataType.h"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>

#include "../basic/CudaUniquePtr.cuh"

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

namespace openpni::process::s2c_any {
namespace constants {
// 此处的能量符合率为（能量选择后的单次事件数量 / 单事件数量）
constexpr float MAX_ENERGY_RATE_EXPECTED = 1.0f;

// 此处的符合率为（符合事件数量 * 2 / 能量选择后的单次事件数量）
constexpr float MAX_PROMPT_COIN_RATE_EXPECTED = 0.5f;
constexpr float MAX_DELAY_COIN_RATE_EXPECTED = 0.25f;
} // namespace constants

typedef struct SingleAnyEnergySelectOp {
  float2 energyWindow;

  __host__ __device__ __forceinline__ SingleAnyEnergySelectOp(
      float __energyWindowLow, float __energyWindowHigh)
      : energyWindow({__energyWindowLow, __energyWindowHigh}) {}

  template <typename T>
  __host__ __device__ __forceinline__ bool operator()(
      const T &a) const {
    return (energyWindow.x <= a.energy && a.energy <= energyWindow.y);
  }
} SingleAnyEnergySelectOp_t;

typedef struct SingleAnyTimeSortOp {
  template <typename T>
  __host__ __device__ __forceinline__ bool operator()(
      const T &a, const T &b) const {
    return a.timeValue_pico < b.timeValue_pico;
  }
} SingleAnyTimeSortOp_t;

namespace functions {
template <typename T, typename SINGLE_CRYMAP_OP>
static __global__ void countSingleAny_kernel(
    const T *__d_in, uint32_t *__d_out, const uint32_t __singleNum, const SINGLE_CRYMAP_OP __mapConvertOp) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= __singleNum) {
    return;
  }

  const uint32_t pos = __mapConvertOp(__d_in[tid]);
  atomicAdd(__d_out + pos, 1);
  return;
}

template <typename T, typename IsGoodPairOp>
static __global__ void coinLocateAny_kernel(
    T *__d_singlesIn, uint32_t *__d_promptNum, uint32_t *__d_delayNum, uint32_t *__d_delayFirstIndex,
    const uint64_t __timeWindowPicoSec, const uint64_t __delayWindowPicoSec, const uint32_t __singlesNum,
    const IsGoodPairOp __goodPairOp) {
  static_assert(std::is_same_v<decltype(std::declval<T>().timeValue_pico), uint64_t>,
                "coinLocateAny_kernel only supports T with timeValue_pico as uint64_t");

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= __singlesNum) {
    return;
  }

  uint32_t promptNumCount = 0;
  uint32_t delayNumCount = 0;

  uint32_t promptEnd = tid;
  uint32_t delayStartIndex = tid;

  // prompt coin locate, assume first single to search is the next single
  for (uint32_t nextIdx = tid + 1; nextIdx < __singlesNum; ++nextIdx) {
    const uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico;

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
  for (uint32_t nextIdx = promptEnd; nextIdx < __singlesNum; ++nextIdx) {
    // dt = t1 - delay - t0
    // to avoid overflow, we use the following logic:
    // if t1 - t0 < delay, then it is before the delayed time window
    // uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico -
    // __delayWindowPicoSec;
    const uint64_t timeDiff = __d_singlesIn[nextIdx].timeValue_pico - __d_singlesIn[tid].timeValue_pico;

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

template <typename T, typename IsGoodPairOp>
static __global__ void coinTimeAny_kernel(
    T *__d_singlesIn, T *__d_promptCoinSinglesOut, T *__d_delayCoinSinglesOut, uint32_t *__d_prefixedPromptNum,
    uint32_t *__d_prefixedDelayNum, uint32_t *__d_delayFirstIndex, const uint64_t __delayTimeWindowPicoSec,
    const uint32_t __singlesNum, const IsGoodPairOp __goodPairOp) {
  static_assert(std::is_same_v<decltype(std::declval<T>().timeValue_pico), uint64_t>,
                "coinTimeAny_kernel only supports T with timeValue_pico as uint64_t");

  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __singlesNum) {
    return;
  }

  const uint32_t promptCoinNum = __d_prefixedPromptNum[tid + 1] - __d_prefixedPromptNum[tid];
  const uint32_t delayCoinNum = __d_prefixedDelayNum[tid + 1] - __d_prefixedDelayNum[tid];

  const uint32_t promptStartIndex = tid + 1;
  const uint32_t delayStartIndex = __d_delayFirstIndex[tid];

  uint32_t processedPromptNum = 0;
  uint32_t processedDelayNum = 0;

  // prompt coins
  for (uint32_t i = 0; processedPromptNum < promptCoinNum && i + promptStartIndex < __singlesNum; i++) {
    if (!__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i + promptStartIndex])) {
      continue;
    }
    __d_promptCoinSinglesOut[(__d_prefixedPromptNum[tid] + processedPromptNum) * 2] = __d_singlesIn[tid];
    __d_promptCoinSinglesOut[(__d_prefixedPromptNum[tid] + processedPromptNum) * 2 + 1] =
        __d_singlesIn[i + promptStartIndex];
    ++processedPromptNum;
  }

  // delay coins
  for (uint32_t i = 0; processedDelayNum < delayCoinNum && i + delayStartIndex < __singlesNum; i++) {
    if (!__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i + delayStartIndex])) {
      continue;
    }
    __d_delayCoinSinglesOut[(__d_prefixedDelayNum[tid] + processedDelayNum) * 2] = __d_singlesIn[tid];
    __d_delayCoinSinglesOut[(__d_prefixedDelayNum[tid] + processedDelayNum) * 2 + 1] =
        __d_singlesIn[i + delayStartIndex];
    ++processedDelayNum;
  }

  return;
}

template <typename T>
static inline cudaError_t singleAnySelectEnergyCub(
    T *__d_singlesIn, T *__d_singlesOut, uint32_t *__d_numSelectedOut, const int32_t __singlesNum,
    const float __energyWindowLowInKeV, const float __energyWindowUpInKeV, size_t &__buffer_bytes, void *__d_buffer,
    cudaStream_t __stream = 0) {
  static_assert(std::is_same_v<decltype(std::declval<T>().energy), float>,
                "singleAnySelectEnergyCub only supports T with energy as float");

  SingleAnyEnergySelectOp op(__energyWindowLowInKeV, __energyWindowUpInKeV);
  return cub::DeviceSelect::If(__d_buffer, __buffer_bytes, __d_singlesIn, __d_singlesOut, __d_numSelectedOut,
                               __singlesNum, op, __stream);
}

template <typename T>
static inline cudaError_t singleAnySortTimeCub(
    T *__d_singlesIn, int32_t __singlesNum, size_t &__buffer_bytes, void *__d_buffer, cudaStream_t __stream = 0) {
  static_assert(std::is_same_v<decltype(std::declval<T>().timeValue_pico), uint64_t>,
                "singleAnySortTimeCub only supports T with timeValue_pico as uint64_t");

  return cub::DeviceMergeSort::StableSortKeys(__d_buffer, __buffer_bytes, __d_singlesIn, __singlesNum,
                                              SingleAnyTimeSortOp(), __stream);
}

// using T = basic::GlobalSingle_t;
// using IsGoodPairOp = singlesCrystalPairDefaultOp;
template <typename T, typename IsGoodPairOp>
static inline cudaError_t singleAnyTimeCoin(
    T *__d_singlesIn, T *__d_promptCoinsOut, T *__d_delayCoinsOut, const uint64_t __timeWindowPicoSec,
    const uint64_t __delayWindowPicoSec, const int32_t __singlesNum, size_t &__buffer_bytes, void *__d_buffer,
    const IsGoodPairOp __goodPairOp, uint32_t &__h_promptNum, uint32_t &__h_delayNum, cudaStream_t __stream = 0,
    std::function<void(std::string)> __callBackFunc = std::function<void(std::string)>()) {
  static_assert(std::is_same_v<decltype(std::declval<T>().timeValue_pico), uint64_t>,
                "singleAnyTimeCoin only supports T with timeValue_pico as uint64_t");

  cudaError_t err = cudaSuccess;
  const uint32_t prefixSize = uint32_t(__singlesNum) + 1;

  uint32_t *d_promptCoinNum = nullptr;
  uint32_t *d_delayCoinNum = nullptr;
  uint32_t *d_firstDelayIndex = nullptr;
  void *d_cub_temp = nullptr;
  size_t d_cub_temp_bytes = 0;

  // get buffer in bytes and return
  if (__d_buffer == nullptr) {
    err = cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_promptCoinNum, d_promptCoinNum, prefixSize,
                                        __stream);

    __buffer_bytes += prefixSize * 3 * sizeof(uint32_t);
    return err;
  }

  // Buffer layout optimization: organize memory layout for better cache performance
  {
    // Calculate required buffer sizes
    const size_t promptCoinNumSize = prefixSize * sizeof(uint32_t);
    const size_t delayCoinNumSize = prefixSize * sizeof(uint32_t);
    const size_t delayFirstIndexSize = prefixSize * sizeof(uint32_t);
    const size_t totalPrefixSize = promptCoinNumSize + delayCoinNumSize + delayFirstIndexSize;

    // Ensure buffer alignment and size validation
    if (__buffer_bytes < totalPrefixSize) {
      return cudaErrorInvalidValue;
    }

    // Setup memory layout with proper pointer arithmetic
    char *buffer_ptr = reinterpret_cast<char *>(__d_buffer);

    d_promptCoinNum = reinterpret_cast<uint32_t *>(buffer_ptr);
    buffer_ptr += promptCoinNumSize;

    d_delayCoinNum = reinterpret_cast<uint32_t *>(buffer_ptr);
    buffer_ptr += delayCoinNumSize;

    d_firstDelayIndex = reinterpret_cast<uint32_t *>(buffer_ptr);
    buffer_ptr += delayFirstIndexSize;

    // Remaining buffer for CUB operations
    d_cub_temp = reinterpret_cast<void *>(buffer_ptr);
    d_cub_temp_bytes = __buffer_bytes - totalPrefixSize;
  }

  // 1. Count prompt and delay coins
  {
    const uint32_t blockNum = (__singlesNum - 1) / THREADNUM + 1;
    coinLocateAny_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, d_promptCoinNum, d_delayCoinNum,
                                                               d_firstDelayIndex, __timeWindowPicoSec,
                                                               __delayWindowPicoSec, __singlesNum, __goodPairOp);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, cudaPeekAtLastError());
  }

#if DEBUG_MODE
  {
    std::unique_ptr<uint32_t[]> h_promptCoinNum = std::make_unique<uint32_t[]>(prefixSize);
    std::unique_ptr<uint32_t[]> h_delayCoinNum = std::make_unique<uint32_t[]>(prefixSize);
    err = cudaMemcpyAsync(h_promptCoinNum.get(), d_promptCoinNum, prefixSize * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                          __stream);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    err = cudaMemcpyAsync(h_delayCoinNum.get(), d_delayCoinNum, prefixSize * sizeof(uint32_t), cudaMemcpyDeviceToHost,
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
      for (uint32_t i = 0; i < prefixSize - 1; ++i) {
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
      for (uint32_t i = 0; i < prefixSize - 1; ++i) {
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
  basic::cuda::cuda_pinned_unique_ptr<uint32_t> h_promptNum = basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);

  basic::cuda::cuda_pinned_unique_ptr<uint32_t> h_delayNum = basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);

  err = cudaMemcpyAsync(h_promptNum.get(), d_promptCoinNum + prefixSize - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                        __stream);
  cudaStreamSynchronize(__stream);
  CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

  err = cudaMemcpyAsync(h_delayNum.get(), d_delayCoinNum + prefixSize - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost,
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

  // 4. Prepare output arrays for prompt and delay coins
  {
    const uint32_t blockNum = (__singlesNum - 1) / THREADNUM + 1;
    coinTimeAny_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, __d_promptCoinsOut, __d_delayCoinsOut,
                                                             d_promptCoinNum, d_delayCoinNum, d_firstDelayIndex,
                                                             __delayWindowPicoSec, __singlesNum, __goodPairOp);

    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, cudaPeekAtLastError());
  }

  __h_promptNum = h_promptNum.at(0);
  __h_delayNum = h_delayNum.at(0);

  return err;
}
} // namespace functions
class s2cAnyHandle {
public:
  s2cAnyHandle() {};
  s2cAnyHandle(const s2cAnyHandle &) = delete;
  s2cAnyHandle(s2cAnyHandle &&) = delete;
  s2cAnyHandle &operator=(const s2cAnyHandle &) = delete;
  s2cAnyHandle &operator=(s2cAnyHandle &&) = delete;
  ~s2cAnyHandle() {};

private:
  template <typename T, typename SINGLE_CRYMAP_OP>
  static cudaError_t countSingleAnyCryMap(
      T *__d_singlesIn, uint32_t *__d_cryMapOut, const uint32_t __singleNum, const SINGLE_CRYMAP_OP __mapConvertOp,
      cudaStream_t __stream = 0, std::function<void(std::string)> __callBackFunc = std::function<void(std::string)>()) {
    if (__singleNum == 0) {
      return cudaSuccess;
    }

    const uint32_t blockNum = (__singleNum - 1) / THREADNUM + 1;

    // Launch the counting kernel
    functions::countSingleAny_kernel<<<blockNum, THREADNUM, 0, __stream>>>(__d_singlesIn, __d_cryMapOut, __singleNum,
                                                                           __mapConvertOp);

    // Synchronize and check for execution errors
    cudaStreamSynchronize(__stream);
    cudaError_t syncError = cudaPeekAtLastError();
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, syncError);

    return cudaSuccess;
  }

private:
  using LogCallback = std::function<void(std::string)>;
  static constexpr int32_t m_headerUnsignedNum = 32; // 128字节对齐，使用4字节的uint32_t

  template <typename T, typename IS_GOOD_PAIR_OP, bool NEED_SELECT_ENERGY, bool NEED_SORT_TIME>
  static std::size_t getBufferNeeded(
      const int32_t __singleNum) {
    std::size_t bufferSizeTotal = 0;

    // Header alignment buffer (128-byte alignment using 4-byte uint32_t)
    bufferSizeTotal += m_headerUnsignedNum * sizeof(uint32_t);

    // buffer for intermediate variables
    bufferSizeTotal += __singleNum * sizeof(T);

    // Internal shared memory size
    std::size_t bufferSize = 0;

    // Energy selection buffer requirement
    if (NEED_SELECT_ENERGY) {
      std::size_t bufferTemp = 0;
      T *singlesPtr = nullptr;
      uint32_t *unsignedPtr = nullptr;
      functions::singleAnySelectEnergyCub(singlesPtr, singlesPtr, unsignedPtr, __singleNum, 0.0f, 0.0f, bufferTemp,
                                          nullptr);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    // Time sorting buffer requirement
    if (NEED_SORT_TIME) {
      std::size_t bufferTemp = 0;
      T *singlesPtr = nullptr;
      functions::singleAnySortTimeCub(singlesPtr, __singleNum, bufferTemp, nullptr);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    // Coincidence processing buffer requirement
    {
      std::size_t bufferTemp = 0;
      T *singlesPtr = nullptr;
      // uint32_t *unsignedPtr = nullptr;
      uint32_t dummyPromptNum = 0;
      uint32_t dummyDelayNum = 0;
      functions::singleAnyTimeCoin(singlesPtr, singlesPtr, singlesPtr, 0, 0, __singleNum, bufferTemp, nullptr,
                                   IS_GOOD_PAIR_OP(), dummyPromptNum, dummyDelayNum);
      bufferSize = std::max(bufferSize, bufferTemp);
    }

    bufferSizeTotal += bufferSize;
    return bufferSizeTotal;
  }

public:
  template <typename T, typename IS_GOOD_PAIR_OP, typename SINGLE_CRYMAP_OP, bool NEED_SELECT_ENERGY,
            bool NEED_SORT_TIME>
  cudaError_t exc(
      T *__d_singlesIn, T *__d_promptCoinsOut, T *__d_delayCoinsOut, uint32_t *__d_cryMap,
      const float __energyWindowLow, const float __energyWindowHigh, const uint64_t __timeWindowPicoSec,
      const uint64_t __delayWindowPicoSec, const int32_t __cryNum, const int32_t __singlesNum,
      uint32_t &__promptCoinsNum, uint32_t &__delayCoinsNum, std::size_t &__bufferSize, void *__d_buffer,
      IS_GOOD_PAIR_OP __goodPairOp, SINGLE_CRYMAP_OP __cryMapOp, const int32_t __deviceId = 0,
      cudaStream_t __stream = 0, LogCallback __callBackFunc = LogCallback()) {
    // 0. 若__d_buffer为nullptr，则获取所需的buffer大小并返回
    if (__d_buffer == nullptr) {
      __bufferSize = getBufferNeeded<T, IS_GOOD_PAIR_OP, NEED_SELECT_ENERGY, NEED_SORT_TIME>(__singlesNum);
      return cudaSuccess;
    }

    if (__bufferSize < getBufferNeeded<T, IS_GOOD_PAIR_OP, NEED_SELECT_ENERGY, NEED_SORT_TIME>(__singlesNum)) {
      CALL_IF(__callBackFunc, "Error: Provided buffer size is smaller than required.");
      return cudaErrorInvalidValue;
    }

    // 1. 设置设备
    cudaError_t err = cudaSuccess;
    int32_t originalDeviceId = 0;
    err = cudaGetDevice(&originalDeviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    err = cudaSetDevice(__deviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 2. 初始化必要的变量
    uint32_t *d_validSinglesNum = nullptr; // 有效的singles数量指针
    T *d_validSingles = nullptr;           // 有效的singles指针
    char *d_buffer = nullptr;              // 中间buffer指针，初始化为nullptr
    size_t bufferSize = __bufferSize;      // 中间buffer的大小
    {
      // 计算buffer的起始位置
      uint8_t *bufferPtr = static_cast<uint8_t *>(__d_buffer);
      d_validSinglesNum = reinterpret_cast<uint32_t *>(bufferPtr);
      bufferPtr += m_headerUnsignedNum * sizeof(uint32_t);
      bufferSize -= m_headerUnsignedNum * sizeof(uint32_t);

      d_validSingles = reinterpret_cast<T *>(bufferPtr);
      bufferPtr += __singlesNum * sizeof(T);
      bufferSize -= __singlesNum * sizeof(T);

      // 中间buffer的剩余部分用于排序和计数
      d_buffer = reinterpret_cast<char *>(bufferPtr);
    }

    basic::cuda::cuda_pinned_unique_ptr<uint32_t> h_validSinglesNum =
        basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);

    // 3. 若需要选择能量，则进行能量选择
    if (NEED_SELECT_ENERGY) {
      // 使用cub的DeviceSelect进行能量选择
      err = functions::singleAnySelectEnergyCub(__d_singlesIn, d_validSingles, d_validSinglesNum, __singlesNum,
                                                __energyWindowLow, __energyWindowHigh, bufferSize, d_buffer, __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

      err = cudaMemcpyAsync(h_validSinglesNum.get(), d_validSinglesNum, sizeof(uint32_t), cudaMemcpyDeviceToHost,
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
      err =
          cudaMemcpyAsync(d_validSingles, __d_singlesIn, __singlesNum * sizeof(T), cudaMemcpyDeviceToDevice, __stream);

      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

#if DEBUG_MODE
    // Debug output for valid singles (commented out but available)
    // {
    //     cudaDeviceSynchronize();
    //     std::cout << "Valid singles num: " << h_validSinglesNum.at(0) << std::endl;
    //     std::cout << "Valid singles rate: " << (static_cast<float>(h_validSinglesNum.at(0)) / __singlesNum) * 100.0f
    //               << "%" << std::endl;
    // }
#endif

    // 4. 若需要排序时间，则进行时间排序
    if (NEED_SORT_TIME) {
      // 使用cub在中间变量d_validSingles上进行时间排序
      err = functions::singleAnySortTimeCub(d_validSingles, h_validSinglesNum.at(0), bufferSize, d_buffer, __stream);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

    // 5. 符合计算
    err = functions::singleAnyTimeCoin(d_validSingles, __d_promptCoinsOut, __d_delayCoinsOut, __timeWindowPicoSec,
                                       __delayWindowPicoSec, h_validSinglesNum.at(0), bufferSize, d_buffer,
                                       __goodPairOp, __promptCoinsNum, __delayCoinsNum, __stream, __callBackFunc);

    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    const uint32_t promptCoinsNumLocal = __promptCoinsNum;
    const uint32_t delayCoinsNumLocal = __delayCoinsNum;

    CALL_IF(__callBackFunc, ("Prompt coins num: " + std::to_string(promptCoinsNumLocal)));
    CALL_IF(__callBackFunc,
            ("Prompt coins rate: " +
             std::to_string(static_cast<float>(promptCoinsNumLocal * 2) / __singlesNum * 100.0f) + "%"));
    CALL_IF(__callBackFunc, ("Delay coins num: " + std::to_string(delayCoinsNumLocal)));
    CALL_IF(__callBackFunc, ("Delay coins rate: " +
                             std::to_string(static_cast<float>(delayCoinsNumLocal * 2) / __singlesNum * 100.0f) + "%"));

#if DEBUG_MODE
    // Debug output for coins (commented out but available)
    // {
    //     cudaDeviceSynchronize();
    //     // Debug code for prompt and delay coins would go here
    // }
#endif

    // 6. 统计cryMap
    // 6.1 计算能量窗前的有效singles数量，若无需能量选择，则不统计（因为所有singles都有效）
    if (NEED_SELECT_ENERGY) {
      err = countSingleAnyCryMap(__d_singlesIn, __d_cryMap, __singlesNum, __cryMapOp, __stream, __callBackFunc);
      cudaStreamSynchronize(__stream);
      CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    }

    // 6.2 计算能量窗后的有效singles数量，若无需能量选择，则统计所有singles（因为所有singles都有效）
    err = countSingleAnyCryMap(d_validSingles, __d_cryMap + __cryNum, h_validSinglesNum.at(0), __cryMapOp, __stream,
                               __callBackFunc);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 6.3 计算prompt coins数量
    err = countSingleAnyCryMap(__d_promptCoinsOut, __d_cryMap + __cryNum * 2, promptCoinsNumLocal * 2, __cryMapOp,
                               __stream, __callBackFunc);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 6.4 计算delay coins数量
    err = countSingleAnyCryMap(__d_delayCoinsOut, __d_cryMap + __cryNum * 3, delayCoinsNumLocal * 2, __cryMapOp,
                               __stream, __callBackFunc);
    cudaStreamSynchronize(__stream);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

    // 7. 恢复原设备
    err = cudaSetDevice(originalDeviceId);
    CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    cudaStreamSynchronize(__stream);

#if DEBUG_MODE
    // Debug output for cryMap (available if needed)
    // {
    //     cudaDeviceSynchronize();
    //     std::unique_ptr<uint32_t[]> h_debugCryMap = std::make_unique<uint32_t[]>(__cryNum * 4);
    //     err = cudaMemcpy(h_debugCryMap.get(),
    //                      __d_cryMap,
    //                      __cryNum * 4 * sizeof(uint32_t),
    //                      cudaMemcpyDeviceToHost);
    //     CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
    //     std::ofstream debugFileCryMap(debugFileFolder + "debug_cry_map.txt");
    //     if (debugFileCryMap.is_open())
    //     {
    //         for (uint32_t i = 0; i < __cryNum * 4; i++)
    //         {
    //             debugFileCryMap << "CryMap[" << i << "]: " << h_debugCryMap[i] << std::endl;
    //         }
    //         debugFileCryMap.close();
    //     }
    // }
#endif

    return cudaSuccess;
  }
};

} // namespace openpni::process::s2c_any

#undef THREADNUM
#undef CALL_AND_RETURN_IF_CUDA_ERR
#undef CRY_MAP_LAYER_NUM
#undef CALL_IF
#undef DEBUG_MODE
#endif