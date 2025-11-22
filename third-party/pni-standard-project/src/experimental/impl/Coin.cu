#include <cub/device/device_merge_sort.cuh>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#include "Coin.h"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
__PNI_CUDA_MACRO__
void mark_as_delayed_coincidence(
    interface::LocalSingle &single) {
  single.channelIndex |= 0x8000; // Mark as delayed coincidence
}
void mark_as_prompt_coincidence(
    interface::LocalSingle &single) {
  single.channelIndex &= 0x7FFF; // Mark as prompt coincidence
}
__PNI_CUDA_MACRO__ bool is_delayed_coincidence(
    interface::LocalSingle const &single) {
  return (single.channelIndex & 0x8000) != 0;
}

void d_copy(
    interface::LocalSingle *d_dest, const interface::LocalSingle *d_src, size_t numElements) {
  cudaMemcpyAsync(d_dest, d_src, numElements * sizeof(interface::LocalSingle), cudaMemcpyDeviceToDevice,
                  basic::cuda_ptr::default_stream());
}
void d_countCrystal(
    unsigned *d_crystalCountMap, interface::LocalSingle const *d_singles, uint64_t singleCount,
    unsigned const *d_crystalNumInclusiveSum, unsigned channelCount) {
  tools::parallel_for_each_CUDA(singleCount, [=] __device__(uint64_t idx) {
    const auto single = d_singles[idx];
    const unsigned channelIndex = single.channelIndex;
    if (channelIndex >= channelCount)
      return;
    const unsigned crystalIndex = single.crystalIndex;
    const unsigned globalCrystalIndex =
        (channelIndex == 0 ? 0 : d_crystalNumInclusiveSum[channelIndex - 1]) + crystalIndex;
    atomicAdd(&d_crystalCountMap[globalCrystalIndex], 1);
  });
}
void d_countCrystal(
    unsigned *d_crystalCountMap, LocalListmode const *d_listmode, uint64_t listmodeCount,
    unsigned const *d_crystalNumInclusiveSum, unsigned channelCount) {
  tools::parallel_for_each_CUDA(listmodeCount, [=] __device__(uint64_t idx) {
    const auto listmode = d_listmode[idx];
    const unsigned channelIndex1 = listmode.channelIndex1;
    const unsigned channelIndex2 = listmode.channelIndex2;
    if (channelIndex1 >= channelCount || channelIndex2 >= channelCount)
      return;
    const unsigned crystalIndex1 = listmode.crystalIndex1;
    const unsigned crystalIndex2 = listmode.crystalIndex2;
    const unsigned globalCrystalIndex1 =
        (channelIndex1 == 0 ? 0 : d_crystalNumInclusiveSum[channelIndex1 - 1]) + crystalIndex1;
    const unsigned globalCrystalIndex2 =
        (channelIndex2 == 0 ? 0 : d_crystalNumInclusiveSum[channelIndex2 - 1]) + crystalIndex2;
    atomicAdd(&d_crystalCountMap[globalCrystalIndex1], 1);
    atomicAdd(&d_crystalCountMap[globalCrystalIndex2], 1);
  });
}
std::size_t d_energyFilter(
    interface::LocalSingle *d_singles, uint64_t singleCount, float energyLowerBound, float energyUpperBound) {
  if (energyLowerBound >= energyUpperBound)
    return 0;
  if (singleCount == 0)
    return 0;
  auto ptr_begin = thrust::device_pointer_cast(d_singles);
  auto ptr_end = ptr_begin + singleCount;
  auto ptr_mid = thrust::partition(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), ptr_begin, ptr_end,
                                   [=] __device__(interface::LocalSingle const &single) {
                                     return (single.energy >= energyLowerBound) && (single.energy <= energyUpperBound);
                                   });
  return ptr_mid - ptr_begin;
}
void d_doubleSinglesAndShiftTime(
    interface::LocalSingle *d_singles, uint64_t singleCount, uint64_t timeShift_pico) {
  tools::parallel_for_each_CUDA(singleCount, [=] __device__(uint64_t idx) {
    auto &single_old = d_singles[idx];
    auto &single_shifted = d_singles[idx + singleCount];
    single_shifted = single_old;
    single_shifted.timevalue_pico += timeShift_pico;
    mark_as_delayed_coincidence(single_shifted);
  });
}
#define USE_TRUST_SORT 0
#if !USE_TRUST_SORT
struct OpSortSingles {
  __host__ __device__ bool operator()(
      interface::LocalSingle const &a, interface::LocalSingle const &b) {
    return a.timevalue_pico < b.timevalue_pico;
  }
};
struct TimeBitExtractor {
  __host__ __device__ uint64_t operator()(
      interface::LocalSingle const &a) const {
    return a.timevalue_pico;
  }
};
#endif
void d_sortSinglesByTime(
    interface::LocalSingle *d_singles, uint64_t singleCount) {
  if (singleCount == 0)
    return;

#if USE_TRUST_SORT
  auto ptr_begin = thrust::device_pointer_cast(d_singles);
  auto ptr_end = ptr_begin + singleCount;
  thrust::sort(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), ptr_begin, ptr_end,
               [] __device__(interface::LocalSingle const &a, interface::LocalSingle const &b) {
                 return a.timevalue_pico < b.timevalue_pico;
               });
#else
  auto temp_keys = make_cuda_sync_ptr<uint64_t>(singleCount, "TempKeys");
  tools::parallel_for_each_CUDA(singleCount, [=, temp_keys = temp_keys.data()] __device__(uint64_t idx) {
    temp_keys[idx] = d_singles[idx].timevalue_pico;
  });

  cuda_sync_ptr<char> buffer;
  while (true) {
    std::size_t bufferSize = buffer.elements();
    cub::DeviceRadixSort::SortPairs(buffer.get(), bufferSize, temp_keys.get(), temp_keys.get(), d_singles, d_singles,
                                    singleCount, 0, sizeof(uint64_t) * 8, basic::cuda_ptr::default_stream());
    if (bufferSize == buffer.elements())
      return;
    buffer.reserve(bufferSize + 20'000);
  }
#endif
}
void d_countCoincidences(
    interface::LocalSingle const *d_singles, uint64_t singleCount, uint8_t *d_promptCount, uint8_t *d_delayCount,
    int timeWindow_pico) {
  tools::parallel_for_each_CUDA(singleCount, [=] __device__(uint64_t idx) {
    const auto &singleLeft = d_singles[idx];
    if (is_delayed_coincidence(singleLeft)) {
      d_promptCount[idx] = d_delayCount[idx] = 0;
      return;
    }
    uint8_t promptCount = 0;
    uint8_t delayCount = 0;
    for (std::size_t j = idx + 1; j < singleCount; ++j) {
      if (promptCount == 255 || delayCount == 255)
        break;

      const auto &singleRight = d_singles[j];
      if (singleRight.channelIndex == singleLeft.channelIndex)
        continue;

      int64_t timeDiff =
          static_cast<int64_t>(singleRight.timevalue_pico) - static_cast<int64_t>(singleLeft.timevalue_pico);
      if (timeDiff > timeWindow_pico)
        break;
      if (is_delayed_coincidence(singleRight)) {
        ++delayCount;
      } else {
        ++promptCount;
      }
    }
    d_promptCount[idx] = promptCount;
    d_delayCount[idx] = delayCount;
  });
}
void d_inclusiveSumUint8(
    uint8_t const *d_input, uint64_t elementCount, uint32_t *d_output) {
  auto begin = thrust::make_transform_iterator(thrust::device_pointer_cast(d_input),
                                               [] __host__ __device__(uint8_t i) -> uint32_t { return i; });
  auto end = begin + elementCount;
  auto out = thrust::device_pointer_cast(d_output);
  thrust::inclusive_scan(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), begin, end, out);
}
void d_extractCoincidenceListmode(
    interface::LocalSingle const *d_singles, uint64_t singleCount, uint32_t const *d_promptInclusiveSum,
    uint32_t const *d_delayInclusiveSum, LocalListmode *d_promptListmode, LocalListmode *d_delayListmode,
    uint64_t maxCount) {
  tools::parallel_for_each_CUDA(singleCount, [=] __device__(uint64_t idx) {
    const auto &singleLeft = d_singles[idx];
    const auto promptIndexBegin = (idx == 0) ? 0 : d_promptInclusiveSum[idx - 1];
    const auto delayIndexBegin = (idx == 0) ? 0 : d_delayInclusiveSum[idx - 1];
    auto promptIndexEnd = d_promptInclusiveSum[idx];
    auto delayIndexEnd = d_delayInclusiveSum[idx];
    if (promptIndexEnd > maxCount)
      promptIndexEnd = maxCount;
    if (delayIndexEnd > maxCount)
      delayIndexEnd = maxCount;

    auto promptIndex = promptIndexBegin;
    auto delayIndex = delayIndexBegin;
    for (std::size_t j = idx + 1; j < singleCount; ++j) {
      if (promptIndex >= promptIndexEnd && delayIndex >= delayIndexEnd)
        break;

      const auto &singleRight = d_singles[j];
      if (singleRight.channelIndex == singleLeft.channelIndex)
        continue;

      int64_t timeDiff = static_cast<int64_t>(singleRight.timevalue_pico) -
                         static_cast<int64_t>(singleLeft.timevalue_pico); // time 2-1
      if (is_delayed_coincidence(singleRight)) {
        auto &listModeDelay = d_delayListmode[delayIndex];
        listModeDelay.channelIndex1 = singleLeft.channelIndex & 0x7FFF;
        listModeDelay.crystalIndex1 = singleLeft.crystalIndex;
        listModeDelay.channelIndex2 = singleRight.channelIndex & 0x7FFF;
        listModeDelay.crystalIndex2 = singleRight.crystalIndex;
        listModeDelay.time1_2pico = -timeDiff; // time 1-2
        ++delayIndex;
      } else {
        auto &listModePrompt = d_promptListmode[promptIndex];
        listModePrompt.channelIndex1 = singleLeft.channelIndex;
        listModePrompt.crystalIndex1 = singleLeft.crystalIndex;
        listModePrompt.channelIndex2 = singleRight.channelIndex;
        listModePrompt.crystalIndex2 = singleRight.crystalIndex;
        listModePrompt.time1_2pico = -timeDiff; // time 1-2
        ++promptIndex;
      }
    }
  });
}

} // namespace openpni::experimental::node::impl
