#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <numeric>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

#include "include/basic/Point.hpp"
#include "include/process/ListmodeProcessing.hpp"
#include "src/common/CudaTemplates.hpp"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
// #define PRINT_CUDA_ERROR \
//   { \
//     cudaDeviceSynchronize(); \
//     cudaError_t err = cudaGetLastError(); \
//     if (err != cudaSuccess) { \
//       std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
//       exit(1); \
//     } \
//   }
#define PRINT_CUDA_ERROR
namespace openpni::process {
__global__ void kernel_countSingles(
    const basic::GlobalSingle_t *__d_singlesIn, unsigned *__d_countMap, uint64_t __singlesNum) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singlesNum)
    return;

  atomicAdd(__d_countMap + __d_singlesIn[tid].globalCrystalIndex, 1);
  return;
}

__global__ void kernel_countCoincidence(
    const basic::Listmode_t *__d_coinIn, unsigned *__d_countMap, uint64_t __coinNum) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __coinNum)
    return;

  atomicAdd(__d_countMap + __d_coinIn[tid].globalCrystalIndex1, 1);
  atomicAdd(__d_countMap + __d_coinIn[tid].globalCrystalIndex2, 1);
  return;
}

__global__ void kernel_convertLocalToGlobal(
    const basic::LocalSingle_t *__d_in, basic::GlobalSingle_t *__d_out, uint64_t __num,
    unsigned __crystalNumPrefixSum) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __num)
    return;

  __d_out[tid].energy = __d_in[tid].energy;
  __d_out[tid].globalCrystalIndex = __crystalNumPrefixSum + __d_in[tid].crystalIndex;
  __d_out[tid].timeValue_pico = __d_in[tid].timevalue_pico;
}

template <typename OpGoodPair>
__global__ void kernel_countPromptCoinNum(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __totalSingleNum, uint16_t __timeWindowPico,
    OpGoodPair __goodPairOp, uint8_t *__d_promptCoinNum) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __totalSingleNum)
    return;

  unsigned promptCount = 0;
  const int64_t timeCurrent = __d_singlesIn[tid].timeValue_pico;
  for (unsigned i = tid + 1; i < __totalSingleNum && promptCount < 0xff; ++i) {
    const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
    if (timeNext - timeCurrent > __timeWindowPico)
      break;
    if (__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i])) {
      ++promptCount;
    }
  }
  __d_promptCoinNum[tid] = promptCount;
}

template <typename OpGoodPair>
__global__ void kernel_getPromptListmode(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __totalSingleNum, const uint64_t *__d_promptCoinPrefixNum,
    uint64_t __totalPromptNum, basic::Listmode_t *__d_promptCoinsOut, OpGoodPair __goodPairOp) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __totalSingleNum)
    return;

  const uint64_t promptIndexStart = __d_promptCoinPrefixNum[tid];
  const uint64_t promptIndexEnd = __d_promptCoinPrefixNum[tid + 1];
  const uint64_t promptNum = promptIndexEnd - promptIndexStart;
  if (promptNum == 0)
    return;
  if (promptIndexEnd > __totalPromptNum)
    return;

  const auto timeCurrent = __d_singlesIn[tid].timeValue_pico;
  uint64_t promptCurrent = promptIndexStart;
  for (uint64_t i = tid + 1; promptCurrent < promptIndexEnd; ++i) {
    const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
    if (__goodPairOp(__d_singlesIn[tid], __d_singlesIn[i])) {
      basic::Listmode_t coin;
      coin.time1_2pico = timeCurrent - timeNext;
      coin.globalCrystalIndex1 = __d_singlesIn[tid].globalCrystalIndex;
      coin.globalCrystalIndex2 = __d_singlesIn[i].globalCrystalIndex;
      __d_promptCoinsOut[promptCurrent] = coin;
      promptCurrent++;
    }
  }
}

template <typename OpGoodPair>
__global__ void kernel_countDelayCoinNum(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singleABNum, uint16_t __timeWindowPico,
    uint8_t *__d_delayCoinNum, OpGoodPair __goodPairOp) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singleABNum)
    return;
  if (__d_singlesIn[tid].globalCrystalIndex & 0x80000000) // if is b
  {
    __d_delayCoinNum[tid] = 0;
    return;
  }

  unsigned delayCount = 0;
  const int64_t timeCurrent = __d_singlesIn[tid].timeValue_pico;
  for (unsigned i = tid + 1; i < __singleABNum && delayCount < 0xff; ++i) {
    const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
    if (timeNext - timeCurrent > __timeWindowPico)
      break;
    if ((__d_singlesIn[i].globalCrystalIndex & 0x80000000) && __goodPairOp(__d_singlesIn[tid], __d_singlesIn[i])) {
      ++delayCount;
    }
  }
  __d_delayCoinNum[tid] = delayCount;
}

template <typename OpGoodPair>
__global__ void kernel_getDelayListmode(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singleABNum, const uint64_t *__d_delayCoinPrefixNum,
    uint64_t __realDelayCoinNum, basic::Listmode_t *__d_delayCoinsOut, OpGoodPair __goodPairOp) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singleABNum)
    return;

  const uint64_t delayIndexStart = __d_delayCoinPrefixNum[tid];
  const uint64_t delayIndexEnd = __d_delayCoinPrefixNum[tid + 1];
  const uint64_t delayNum = delayIndexEnd - delayIndexStart;
  if (delayNum == 0)
    return;
  if (delayIndexEnd > __realDelayCoinNum)
    return;

  const auto timeCurrent = __d_singlesIn[tid].timeValue_pico;
  uint64_t delayCurrent = delayIndexStart;
  for (uint64_t i = tid + 1; delayCurrent < delayIndexEnd && i < __singleABNum; ++i) {
    const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
    if ((__d_singlesIn[i].globalCrystalIndex & 0x80000000) && __goodPairOp(__d_singlesIn[tid], __d_singlesIn[i])) {
      basic::Listmode_t coin;
      coin.time1_2pico = timeCurrent - timeNext;
      coin.globalCrystalIndex1 = __d_singlesIn[tid].globalCrystalIndex & 0x7fffffff; // Clear the B mark
      coin.globalCrystalIndex2 = __d_singlesIn[i].globalCrystalIndex & 0x7fffffff;   // Clear the B mark
      __d_delayCoinsOut[delayCurrent] = coin;
      delayCurrent++;
    }
  }
}

template <typename OpGoodPair>
__global__ void kernel_countPromptAndDelayCoinNum(
    const basic::GlobalSingle_t *__d_singlesAB, uint64_t __singlesABNum, uint16_t __timeWindowPico,
    uint8_t *__d_promptCoinNum, uint8_t *__d_delayCoinNum, OpGoodPair __goodPairOp) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singlesABNum)
    return;

  // 如果 this element is marked B (high bit set) -> 不作为第一个事件 i 来计数
  if (__d_singlesAB[tid].globalCrystalIndex & 0x80000000) {
    __d_promptCoinNum[tid] = 0;
    __d_delayCoinNum[tid] = 0;
    return;
  }

  unsigned promptCount = 0;
  unsigned delayCount = 0;
  const int64_t timeCurrent = __d_singlesAB[tid].timeValue_pico;

  // 只向后遍历（j>i），直到时间差超过 timeWindow
  for (uint64_t j = tid + 1; j < __singlesABNum && (timeCurrent - __d_singlesAB[j].timeValue_pico) <= 0; ++j) {
    const int64_t timeNext = __d_singlesAB[j].timeValue_pico;
    if (timeNext - timeCurrent > __timeWindowPico)
      break;

    // 如果 j 是 B（被标记），则可能属于 delay 配对（i:A, j:B）
    bool jIsB = (__d_singlesAB[j].globalCrystalIndex & 0x80000000);
    bool jIsA = !jIsB;

    // PROMPT 条件： both A 且满足 good-pair
    if (jIsA && __goodPairOp(__d_singlesAB[tid], __d_singlesAB[j])) {
      if (promptCount < 0xff)
        ++promptCount;
    }

    // DELAY 条件： i is A (we already know) and j is B and goodPair (in your semantics)
    if (jIsB && __goodPairOp(__d_singlesAB[tid], __d_singlesAB[j])) {
      if (delayCount < 0xff)
        ++delayCount;
    }

    // 如果 promptCount 和 delayCount 都到达 0xff 可以提前 break
    if (promptCount >= 0xff && delayCount >= 0xff)
      break;
  }

  __d_promptCoinNum[tid] = static_cast<uint8_t>(promptCount);
  __d_delayCoinNum[tid] = static_cast<uint8_t>(delayCount);
}

template <typename OpGoodPair>
__global__ void kernel_getPromptAndDelayListmode(
    const basic::GlobalSingle_t *__d_singlesAB, uint64_t __singlesABNum, uint16_t __timeWindowPico,
    const uint64_t *__d_promptPrefix, const uint64_t *__d_delayPrefix, basic::Listmode_t *__d_promptOut,
    basic::Listmode_t *__d_delayOut, OpGoodPair __goodPairOp) {

  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singlesABNum)
    return;

  // 只有 A 类型会写出（作为第一个事件）
  if (__d_singlesAB[tid].globalCrystalIndex & 0x80000000)
    return;

  const int64_t timeCurrent = __d_singlesAB[tid].timeValue_pico;
  uint64_t promptWritePos = __d_promptPrefix[tid];
  uint64_t delayWritePos = __d_delayPrefix[tid];

  for (uint64_t j = tid + 1; j < __singlesABNum; ++j) {
    const int64_t timeNext = __d_singlesAB[j].timeValue_pico;
    if (timeNext - timeCurrent > __timeWindowPico)
      break;

    // j 是 A => prompt 候选
    if (!(__d_singlesAB[j].globalCrystalIndex & 0x80000000)) {
      if (__goodPairOp(__d_singlesAB[tid], __d_singlesAB[j])) {
        // write prompt
        basic::Listmode_t coin;
        coin.time1_2pico = timeCurrent - timeNext;
        coin.globalCrystalIndex1 = __d_singlesAB[tid].globalCrystalIndex; // already A
        coin.globalCrystalIndex2 = __d_singlesAB[j].globalCrystalIndex;
        __d_promptOut[promptWritePos++] = coin;
      }
    } else { // j is B => delay candidate; 清除高位标记后写出
      if (__goodPairOp(__d_singlesAB[tid], __d_singlesAB[j])) {
        basic::Listmode_t coin;
        coin.time1_2pico = timeCurrent - timeNext;
        coin.globalCrystalIndex1 = __d_singlesAB[tid].globalCrystalIndex & 0x7fffffff; // clear B bit for index1
        coin.globalCrystalIndex2 = __d_singlesAB[j].globalCrystalIndex & 0x7fffffff;   // clear B bit for index2
        __d_delayOut[delayWritePos++] = coin;
      }
    }
  }
}

struct OpSelectEnergyCoin {
  float energyWindow[2]; // {low, high}

  __host__ __device__ __forceinline__ OpSelectEnergyCoin(
      float __energyWindowLow, float __energyWindowHigh) {
    energyWindow[0] = __energyWindowLow;
    energyWindow[1] = __energyWindowHigh;
  }

  __host__ __device__ __forceinline__ bool operator()(
      const basic::GlobalSingle_t &a) const {
    return (energyWindow[0] <= a.energy && a.energy <= energyWindow[1]);
  }
};

struct OpSortSingles {
  __host__ __device__ bool operator()(
      const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) {
    return a.timeValue_pico < b.timeValue_pico;
  }
};

struct OpPairGoodPosition {
  __host__ __device__ __forceinline__ bool operator()(
      const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) const {
    return a.globalCrystalIndex != b.globalCrystalIndex;
  }
};

template <typename OpGoodPair>
uint64_t timeCoincidencePrompt(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singlesNum, basic::Listmode_t *__d_promptCoinsOut,
    uint64_t __maxPromptNum, uint16_t __timeWindowPico, size_t &__buffer_bytes, void *__d_buffer,
    const OpGoodPair &__goodPairOp) {
  if (__singlesNum == 0)
    return 0;
  auto d_promptNum = make_cuda_sync_ptr<uint8_t>(__singlesNum + 1);
  thrust::fill(thrust::device_pointer_cast(d_promptNum.begin()), thrust::device_pointer_cast(d_promptNum.end()),
               0); // Initialize to zero
  PRINT_CUDA_ERROR
  kernel_countPromptCoinNum<<<(__singlesNum + 255) / 256, 256>>>(__d_singlesIn, __singlesNum, __timeWindowPico,
                                                                 __goodPairOp, d_promptNum);
  PRINT_CUDA_ERROR
  auto d_promptNumPrefixSum = make_cuda_sync_ptr<uint64_t>(__singlesNum + 1);
  cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_promptNum.data(), d_promptNumPrefixSum.data(),
                                __singlesNum + 1); // Thus, d_promptNumPrefixSum[__singlesNum] == total prompt num.
  PRINT_CUDA_ERROR
  uint64_t promptNumTotal =
      make_vector_from_cuda_sync_ptr(
          d_promptNumPrefixSum, std::span<const uint64_t>(d_promptNumPrefixSum.end() - 1, d_promptNumPrefixSum.end()))
          .front();
  PRINT_CUDA_ERROR
  uint64_t realPromptNum = std::min<uint64_t>(promptNumTotal, __maxPromptNum);
  kernel_getPromptListmode<<<(__singlesNum + 255) / 256, 256>>>(__d_singlesIn, __singlesNum, d_promptNumPrefixSum,
                                                                realPromptNum, __d_promptCoinsOut, __goodPairOp);
  PRINT_CUDA_ERROR
  return realPromptNum;
}

__global__ void kernel_doubleSingles(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singlesNum, basic::GlobalSingle_t *__d_singlesOut,
    uint32_t __delayTimePico) {
  const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= __singlesNum)
    return;

  __d_singlesOut[tid * 2] = __d_singlesIn[tid];

  __d_singlesOut[tid * 2 + 1].energy = __d_singlesIn[tid].energy;
  __d_singlesOut[tid * 2 + 1].timeValue_pico = __d_singlesIn[tid].timeValue_pico - __delayTimePico;
  __d_singlesOut[tid * 2 + 1].globalCrystalIndex = 0x80000000 | __d_singlesIn[tid].globalCrystalIndex; // Mark as B
}

template <typename OpGoodPair>
CoincidenceCount timeCoincidenceCombined(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singlesNum, basic::Listmode_t *__d_promptCoinsOut,
    uint64_t __maxPromptNum, basic::Listmode_t *__d_delayCoinsOut, uint64_t __maxDelayNum, uint16_t __timeWindowPico,
    uint32_t __delayWindowPico, size_t &__buffer_bytes, void *__d_buffer, const OpGoodPair &__goodPairOp) {

  if (__singlesNum == 0)
    return {0, 0};

  // 1) allocate arrays: 2N singles (A + delayed B)
  auto d_singlesAB = make_cuda_sync_ptr<basic::GlobalSingle_t>(2 * __singlesNum);
  PRINT_CUDA_ERROR
  kernel_doubleSingles<<<(__singlesNum + 255) / 256, 256>>>(__d_singlesIn, __singlesNum, d_singlesAB.data(),
                                                            __delayWindowPico);
  PRINT_CUDA_ERROR

  // 2) sort the 2N array by time
  cub::DeviceMergeSort::SortKeys(__d_buffer, __buffer_bytes, d_singlesAB.data(), 2 * __singlesNum, OpSortSingles());
  PRINT_CUDA_ERROR

  // 3) count per-element prompt & delay numbers (uint8 each)
  auto d_promptNumPer = make_cuda_sync_ptr<uint8_t>(2 * __singlesNum + 1);
  auto d_delayNumPer = make_cuda_sync_ptr<uint8_t>(2 * __singlesNum + 1);
  thrust::fill(thrust::device_pointer_cast(d_promptNumPer.begin()), thrust::device_pointer_cast(d_promptNumPer.end()),
               0);
  thrust::fill(thrust::device_pointer_cast(d_delayNumPer.begin()), thrust::device_pointer_cast(d_delayNumPer.end()), 0);
  PRINT_CUDA_ERROR

  kernel_countPromptAndDelayCoinNum<<<(2 * __singlesNum + 255) / 256, 256>>>(d_singlesAB.data(), 2 * __singlesNum,
                                                                             __timeWindowPico, d_promptNumPer.data(),
                                                                             d_delayNumPer.data(), __goodPairOp);
  PRINT_CUDA_ERROR

  // 4) exclusive-scan to get prefix arrays (uint64_t)
  auto d_promptPrefix = make_cuda_sync_ptr<uint64_t>(2 * __singlesNum + 1);
  auto d_delayPrefix = make_cuda_sync_ptr<uint64_t>(2 * __singlesNum + 1);

  cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_promptNumPer.data(), d_promptPrefix.data(),
                                2 * __singlesNum + 1);
  PRINT_CUDA_ERROR
  cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_delayNumPer.data(), d_delayPrefix.data(),
                                2 * __singlesNum + 1);
  PRINT_CUDA_ERROR

  // 5) read totals (last element of prefix arrays)
  uint64_t totalPrompt = make_vector_from_cuda_sync_ptr(
                             d_promptPrefix, std::span<const uint64_t>(d_promptPrefix.end() - 1, d_promptPrefix.end()))
                             .front();
  uint64_t totalDelay = make_vector_from_cuda_sync_ptr(
                            d_delayPrefix, std::span<const uint64_t>(d_delayPrefix.end() - 1, d_delayPrefix.end()))
                            .front();

  uint64_t realPrompt = std::min<uint64_t>(totalPrompt, __maxPromptNum);
  uint64_t realDelay = std::min<uint64_t>(totalDelay, __maxDelayNum);

  // 6) finally write out prompt & delay pairs (the kernels assume outputs are large enough)
  kernel_getPromptAndDelayListmode<<<(2 * __singlesNum + 255) / 256, 256>>>(
      d_singlesAB.data(), 2 * __singlesNum, __timeWindowPico, d_promptPrefix.data(), d_delayPrefix.data(),
      __d_promptCoinsOut, __d_delayCoinsOut, __goodPairOp);
  PRINT_CUDA_ERROR

  CoincidenceCount result;
  result.promptCount = realPrompt;
  result.delayCount = realDelay;
  return result;
}

template <typename OpGoodPair>
uint64_t timeCoincidenceDelay(
    const basic::GlobalSingle_t *__d_singlesIn, uint64_t __singlesNum, basic::Listmode_t *__d_delayCoinsOut,
    uint64_t __maxDelayNum, uint16_t __timeWindowPico, uint32_t __delayWindowPico, size_t &__buffer_bytes,
    void *__d_buffer, const OpGoodPair &__goodPairOp) {
  if (__singlesNum == 0)
    return 0;

  auto d_delayNum = make_cuda_sync_ptr<uint8_t>(2 * __singlesNum + 1);
  auto d_singlesAB = make_cuda_sync_ptr<basic::GlobalSingle_t>(2 * __singlesNum);
  PRINT_CUDA_ERROR
  kernel_doubleSingles<<<(__singlesNum + 255) / 256, 256>>>(__d_singlesIn, __singlesNum, d_singlesAB,
                                                            __delayWindowPico);
  PRINT_CUDA_ERROR
  cub::DeviceMergeSort::SortKeys(__d_buffer, __buffer_bytes, d_singlesAB.data(), 2 * __singlesNum, OpSortSingles());
  PRINT_CUDA_ERROR
  kernel_countDelayCoinNum<<<(2 * __singlesNum + 255) / 256, 256>>>(d_singlesAB, 2 * __singlesNum, __timeWindowPico,
                                                                    d_delayNum, __goodPairOp);
  PRINT_CUDA_ERROR
  cuda_sync_ptr<uint64_t> d_delayNumPrefixSum = make_cuda_sync_ptr<uint64_t>(2 * __singlesNum + 1);
  cub::DeviceScan::ExclusiveSum(__d_buffer, __buffer_bytes, d_delayNum.data(), d_delayNumPrefixSum.data(),
                                2 * __singlesNum +
                                    1); // Thus, d_promptNumPrefixSum[2 *__singlesNum] == total prompt num.
  const uint64_t delayNumTotal =
      make_vector_from_cuda_sync_ptr(
          d_delayNumPrefixSum, std::span<const uint64_t>(d_delayNumPrefixSum.end() - 1, d_delayNumPrefixSum.end()))
          .front();
  auto realDelayNum = std::min<uint64_t>(delayNumTotal, __maxDelayNum);
  PRINT_CUDA_ERROR
  kernel_getDelayListmode<<<(2 * __singlesNum + 255) / 256, 256>>>(d_singlesAB, 2 * __singlesNum, d_delayNumPrefixSum,
                                                                   realDelayNum, __d_delayCoinsOut, __goodPairOp);
  PRINT_CUDA_ERROR
  return realDelayNum;
}

CoincidenceResult stoc(
    const LocalSinglesOfEachChannel &d_localSingles, device::DetectorBase *const *h_detectors,
    CoincidenceProtocol protocol) {
  // Join local single vectors into one global vector
  uint64_t totalSingleNum = std::accumulate(d_localSingles.begin(), d_localSingles.end(), 0ULL,
                                            [](uint64_t sum, const cuda_sync_ptr<basic::LocalSingle_t> &localSingle) {
                                              return sum + localSingle.elements();
                                            });
  unsigned totalCrystalNum = std::accumulate(
      h_detectors, h_detectors + d_localSingles.size(), 0u, [](unsigned sum, device::DetectorBase *detector) {
        return sum + detector->detectorUnchangable().geometry.getTotalCrystalNum();
      });
  CoincidenceResult result;
  result.d_countMap = make_cuda_sync_ptr<unsigned>(totalCrystalNum * 4);
  result.d_countMap.allocator().memset(0, result.d_countMap.span());
  auto d_globalSingleRaw = make_cuda_sync_ptr<basic::GlobalSingle_t>(totalSingleNum);
  auto d_tempBuffer = make_cuda_sync_ptr<char>(totalSingleNum * 2 * sizeof(basic::GlobalSingle_t));
  for (uint64_t convertedSingleNum{0}, crystalNumPrefixSum{0};
       const auto localSingleGroup : std::ranges::views::iota(0ull, d_localSingles.size())) {
    const auto &localSingle = d_localSingles[localSingleGroup];
    kernel_convertLocalToGlobal<<<(localSingle.elements() + 255) / 256, 256>>>(
        localSingle.data(), d_globalSingleRaw.data() + convertedSingleNum, localSingle.elements(), crystalNumPrefixSum);
    convertedSingleNum += localSingle.elements();
    crystalNumPrefixSum += h_detectors[localSingleGroup]->detectorUnchangable().geometry.getTotalCrystalNum();
  }

  // Count the number of singles in each channel [ Layer 1 ]
  kernel_countSingles<<<(totalSingleNum + 255) / 256, 256>>>(d_globalSingleRaw.data(), result.d_countMap.data(),
                                                             totalSingleNum);

  // Do energy coincidence
  auto bufferSize = d_tempBuffer.elements();

  auto energySelectedEnd = thrust::partition(thrust::device_pointer_cast(d_globalSingleRaw.begin()),
                                             thrust::device_pointer_cast(d_globalSingleRaw.end()),
                                             OpSelectEnergyCoin(protocol.energyLower_eV, protocol.energyUpper_eV));

  const auto selectedOutEnergyNum = energySelectedEnd - thrust::device_pointer_cast(d_globalSingleRaw.begin());
  // Count the number of singles in each channel [ Layer 2 ]
  kernel_countSingles<<<(selectedOutEnergyNum + 255) / 256, 256>>>(
      d_globalSingleRaw.data(), result.d_countMap.data() + totalCrystalNum, selectedOutEnergyNum);

  // Do time coincidence
  cub::DeviceMergeSort::SortKeys(d_tempBuffer.data(), bufferSize, d_globalSingleRaw.data(), selectedOutEnergyNum,
                                 OpSortSingles());
  // printf("selectedOutEnergyNum: %llu\n", selectedOutEnergyNum);
  result.d_promptListmode = make_cuda_sync_ptr<basic::Listmode_t>(selectedOutEnergyNum);
  result.d_delayListmode = make_cuda_sync_ptr<basic::Listmode_t>(selectedOutEnergyNum);
  const auto maxPromptNum = selectedOutEnergyNum;
  const auto maxDelayNum = selectedOutEnergyNum;
  PRINT_CUDA_ERROR
  result.promptCount = timeCoincidencePrompt(d_globalSingleRaw.data(), selectedOutEnergyNum,
                                             result.d_promptListmode.data(), maxPromptNum, protocol.timeWindow_ps,
                                             bufferSize, d_tempBuffer.data(), OpPairGoodPosition());
  PRINT_CUDA_ERROR
  result.delayCount = timeCoincidenceDelay(d_globalSingleRaw.data(), selectedOutEnergyNum,
                                           result.d_delayListmode.data(), maxDelayNum, protocol.timeWindow_ps,
                                           protocol.delayTime_ps, // 5 microseconds
                                           bufferSize, d_tempBuffer.data(), OpPairGoodPosition());
  PRINT_CUDA_ERROR
  // Count the number of singles in each channel [ Layer 3 ]
  if (result.promptCount)
    kernel_countCoincidence<<<(result.promptCount + 255) / 256, 256>>>(
        result.d_promptListmode.data(), result.d_countMap.data() + totalCrystalNum * 2, result.promptCount);
  PRINT_CUDA_ERROR
  // Count the number of singles in each channel [ Layer 4 ]
  if (result.delayCount)
    kernel_countCoincidence<<<(result.delayCount + 255) / 256, 256>>>(
        result.d_delayListmode.data(), result.d_countMap.data() + totalCrystalNum * 3, result.delayCount);
  PRINT_CUDA_ERROR
  return result;
}


} // namespace openpni::process

#endif
