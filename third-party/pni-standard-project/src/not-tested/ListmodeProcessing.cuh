#pragma once

namespace openpni::process {
inline cuda_sync_ptr<basic::GlobalSingle_t> rtos(
    // 这个函数消耗多少显存是不确定的，故很难做buffer进行缓存
    RawDataView d_rawdata, device::DetectorBase const *const *__detectors) {
  using PacketPositionInfo = openpni::device::PacketPositionInfo;
  const auto detectorNum = d_rawdata.channelNum;
  std::vector<std::size_t> packetNumOfEachChannel(detectorNum, 0);

  cuda_sync_ptr<PacketPositionInfo> d_packetPosition = make_cuda_sync_ptr<PacketPositionInfo>(d_rawdata.count);
  for_each_CUDA(d_packetPosition.elements(), [p = d_packetPosition.get(), d_rawdata] __device__(std::size_t idx) {
    p[idx].offset = d_rawdata.offset[idx];
    p[idx].length = d_rawdata.length[idx];
    p[idx].channel = d_rawdata.channel[idx];
  });

  for (PacketPositionInfo *begin = d_packetPosition.begin();
       const auto detectorIndex : std::views::iota(0u, detectorNum)) {
    const auto end = thrust::raw_pointer_cast(&*thrust::partition(
        thrust::device_pointer_cast(begin), thrust::device_pointer_cast(d_packetPosition.end()),
        [detectorIndex] __device__(const PacketPositionInfo &value) { return detectorIndex == value.channel; }));
    packetNumOfEachChannel[detectorIndex] = end - begin;
    begin = thrust::raw_pointer_cast(end);
  }
  std::size_t totalMaxSingleNum = 0;
  for (const auto detectorIndex : std::views::iota(0u, detectorNum))
    totalMaxSingleNum +=
        packetNumOfEachChannel[detectorIndex] * __detectors[detectorIndex]->detectorUnchangable().maxSingleNumPerPacket;
  cuda_sync_ptr<basic::LocalSingle_t> d_localSingles = make_cuda_sync_ptr<basic::LocalSingle_t>(totalMaxSingleNum);

  {
    PacketPositionInfo *begin = d_packetPosition.begin();
    basic::LocalSingle_t *localSingleBegin = d_localSingles.begin();
    for (const auto detectorIndex : std::views::iota(0u, detectorNum)) {
      auto end = begin + packetNumOfEachChannel[detectorIndex];
      auto localSingleEnd =
          localSingleBegin + (end - begin) * __detectors[detectorIndex]->detectorUnchangable().maxSingleNumPerPacket;
      if (begin == end)
        continue;
      __detectors[detectorIndex]->r2s_cuda(d_rawdata.data, begin, end - begin, localSingleBegin);
      for_each_CUDA(localSingleEnd - localSingleBegin, [p = localSingleBegin, detectorIndex] __device__(
                                                           std::size_t idx) { p[idx].bdmIndex = detectorIndex; });
      localSingleBegin = localSingleEnd;
    }
  }

  auto validLocalSingleEnd = thrust::raw_pointer_cast(&*thrust::partition(
      thrust::device_pointer_cast(d_localSingles.begin()), thrust::device_pointer_cast(d_localSingles.end()),
      [] __device__(const basic::LocalSingle_t &value) {
        return value.crystalIndex != static_cast<unsigned short>(-1);
      }));

  cuda_sync_ptr<basic::GlobalSingle_t> result =
      make_cuda_sync_ptr<basic::GlobalSingle_t>(validLocalSingleEnd - d_localSingles.begin());
  std::vector<std::size_t> crystalNumOfEachDetector(detectorNum, 0);
  for (const auto detectorIndex : std::views::iota(0u, detectorNum))
    crystalNumOfEachDetector[detectorIndex] =
        __detectors[detectorIndex]->detectorUnchangable().geometry.getTotalCrystalNum();
  std::vector<std::size_t> crystalNumPrefixSum(detectorNum, 0);
  std::exclusive_scan(std::begin(crystalNumOfEachDetector), std::end(crystalNumOfEachDetector),
                      std::begin(crystalNumPrefixSum), 0);
  auto d_prefixSum = make_cuda_sync_ptr_from_hcopy(crystalNumPrefixSum);
  for_each_CUDA(result.elements(), [d_localSingles = d_localSingles.get(), p = result.get(),
                                    prefixSum = d_prefixSum.get()] __device__(std::size_t idx) {
    p[idx].energy = d_localSingles[idx].energy;
    p[idx].timeValue_pico = d_localSingles[idx].timevalue_pico;
    p[idx].globalCrystalIndex = prefixSum[d_localSingles[idx].bdmIndex] + d_localSingles[idx].crystalIndex;
  });

  return result;
}

struct _StoC {
  static void countSingles(
      const basic::GlobalSingle_t *d_singles, unsigned *d_countMap, std::size_t singleNum) {
    if (singleNum == 0 || d_singles == nullptr || d_countMap == nullptr)
      return;
    for_each_CUDA(singleNum,
                  [=] __device__(std::size_t idx) { atomicAdd(&d_countMap[d_singles[idx].globalCrystalIndex], 1); });
  }
  static void countListmode(
      const basic::Listmode_t *d_coincidences, unsigned *d_countMap, std::size_t coinNum) {
    if (coinNum == 0 || d_coincidences == nullptr || d_countMap == nullptr)
      return;
    for_each_CUDA(coinNum, [=] __device__(std::size_t idx) {
      const auto &coin = d_coincidences[idx];
      atomicAdd(&d_countMap[coin.globalCrystalIndex1], 1);
      atomicAdd(&d_countMap[coin.globalCrystalIndex2], 1);
    });
  }
  struct _OpSortSingles {
    __host__ __device__ bool operator()(
        const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) {
      return a.timeValue_pico < b.timeValue_pico;
    }
  };
  struct _OpSelectEnergyCoin {
    float energyWindow[2]; // {low, high}

    __host__ __device__ __forceinline__ _OpSelectEnergyCoin(
        float __energyWindowLow, float __energyWindowHigh) {
      energyWindow[0] = __energyWindowLow;
      energyWindow[1] = __energyWindowHigh;
    }

    __host__ __device__ __forceinline__ bool operator()(
        const basic::GlobalSingle_t &a) const {
      return (energyWindow[0] <= a.energy && a.energy <= energyWindow[1]);
    }
  };
  struct OpPairGoodKeepAll {
    __host__ __device__ __forceinline__ bool operator()(
        const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) const {
      return true;
    }
  };
  struct _Protocol {
    float energyLowThreshold_eV = 350e3;
    float energyHighThreshold_eV = 650e3;
    int timeWindow_pico = 2000;          // 2ns
    int delayTimeWindow_pico = 20000000; // 20us
    unsigned totalCrystalNum{0};
  };
  auto protocol() const { return _Protocol{}; }

  struct _MemoryAccess {
    std::span<const basic::GlobalSingle_t> __d_globalSingles;
    const basic::IntegratedModel &scannerModel;
    void *__d_buffer;
    std::size_t &__d_bufferSize;
    _MemoryAccess(
        std::span<const basic::GlobalSingle_t> d_globalSingles, void *d_buffer, std::size_t &d_bufferSize)
        : __d_globalSingles(d_globalSingles)
        , scannerModel(scannerModel)
        , __d_buffer(d_buffer)
        , __d_bufferSize(d_bufferSize) {
      d_singleCopy = aligned_as<basic::GlobalSingle_t>(d_buffer);
      basic::GlobalSingle_t *d_bufferEnd = d_singleCopy + __d_globalSingles.size() * 2;
      m_bufferSizeNeeded = reinterpret_cast<uint8_t *>(d_bufferEnd) - reinterpret_cast<uint8_t *>(d_buffer);
    }
    [[nodiscard]] operator bool() {
      if (m_bufferSizeNeeded <= __d_bufferSize) {
        return true;
      }
      __d_bufferSize = m_bufferSizeNeeded;
      return false;
    }

    basic::GlobalSingle_t *singlesBegin() const { return d_singleCopy; }

  private:
    basic::GlobalSingle_t *d_singleCopy;
    std::size_t m_bufferSizeNeeded;
  };
  template <typename OpGoodPair>
  uint64_t time_coincidence_prompt(
      std::span<const basic::GlobalSingle_t> __d_singlesIn, basic::Listmode_t *__d_promptCoinsOut,
      uint64_t __maxPromptNum, uint16_t __timeWindowPico, cuda_sync_ptr<char> &__d_flexBuffer,
      OpGoodPair __goodPairOp) const {
    const auto singleNum = __d_singlesIn.size();
    auto d_promptNum = make_cuda_sync_ptr<uint8_t>(__d_singlesIn.size() + 1);
    d_promptNum.allocator().memset(0, d_promptNum.span());
    PRINT_CUDA_ERROR
    for_each_CUDA(__d_singlesIn.size(), [p = d_promptNum.get(), totalSingleNum = singleNum, __goodPairOp,
                                         __timeWindowPico, __d_singlesIn = __d_singlesIn.data(),
                                         __d_promptCoinNum = d_promptNum.data()] __device__(std::size_t idx) {
      unsigned promptCount = 0;
      const int64_t timeCurrent = __d_singlesIn[idx].timeValue_pico;
      for (unsigned i = idx + 1; i < totalSingleNum && promptCount < 0xff; ++i) {
        const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
        if (timeNext - timeCurrent > __timeWindowPico)
          break;
        if (__goodPairOp(__d_singlesIn[idx], __d_singlesIn[i])) {
          ++promptCount;
        }
      }
      __d_promptCoinNum[idx] = promptCount;
    });
    PRINT_CUDA_ERROR
    auto d_promptNumPrefixSum = make_cuda_sync_ptr<uint64_t>(__d_singlesIn.size() + 1);
    {
      std::size_t bufferSize = __d_flexBuffer.elements();
      for (int i = 0; i < 2; i++)
        if (cudaSuccess != cub::DeviceScan::ExclusiveSum(__d_flexBuffer.data(), bufferSize, d_promptNum.data(),
                                                         d_promptNumPrefixSum.data(), d_promptNumPrefixSum.elements());
            bufferSize > __d_flexBuffer.elements())
          __d_flexBuffer = make_cuda_sync_ptr<char>(bufferSize);
        else
          break; // Thus, d_promptNumPrefixSum[__singlesNum] == total prompt num.
    }
    PRINT_CUDA_ERROR
    uint64_t promptNumTotal =
        make_vector_from_cuda_sync_ptr(
            d_promptNumPrefixSum, std::span<const uint64_t>(d_promptNumPrefixSum.end() - 1, d_promptNumPrefixSum.end()))
            .front();
    PRINT_CUDA_ERROR
    uint64_t realPromptNum = std::min<uint64_t>(promptNumTotal, __maxPromptNum);
    for_each_CUDA(__d_singlesIn.size(),
                  [__maxPromptNum, promptCoinPrefixNum = d_promptNumPrefixSum.data(), singlesIn = __d_singlesIn.data(),
                   __goodPairOp, __d_promptCoinsOut] __device__(std::size_t idx) {
                    const uint64_t promptIndexStart = promptCoinPrefixNum[idx];
                    const uint64_t promptIndexEnd = promptCoinPrefixNum[idx + 1];
                    const uint64_t promptNum = promptIndexEnd - promptIndexStart;
                    if (promptNum == 0)
                      return;
                    if (promptIndexEnd > __maxPromptNum)
                      return;

                    const auto timeCurrent = singlesIn[idx].timeValue_pico;
                    uint64_t promptCurrent = promptIndexStart;
                    for (uint64_t i = idx + 1; promptCurrent < promptIndexEnd; ++i) {
                      const int64_t timeNext = singlesIn[i].timeValue_pico;
                      if (__goodPairOp(singlesIn[idx], singlesIn[i])) {
                        basic::Listmode_t coin;
                        coin.time1_2pico = timeCurrent - timeNext;
                        coin.globalCrystalIndex1 = singlesIn[idx].globalCrystalIndex;
                        coin.globalCrystalIndex2 = singlesIn[i].globalCrystalIndex;
                        __d_promptCoinsOut[promptCurrent] = coin;
                        promptCurrent++;
                      }
                    }
                  });
    PRINT_CUDA_ERROR
    return realPromptNum;
  }
  template <typename OpGoodPair>
  uint64_t time_coincidence_delay(
      std::span<const basic::GlobalSingle_t> __d_singlesIn, basic::Listmode_t *__d_delayCoinsOut,
      uint64_t __maxDelayNum, uint16_t __timeWindowPico, uint32_t __delayWindowPico,
      cuda_sync_ptr<char> &__d_flexBuffer, const OpGoodPair &__goodPairOp) const {
    const auto singleNum = __d_singlesIn.size();
    auto d_delayNum = make_cuda_sync_ptr<uint8_t>(2 * singleNum + 1);
    auto d_singlesAB = make_cuda_sync_ptr<basic::GlobalSingle_t>(2 * singleNum);
    PRINT_CUDA_ERROR
    for_each_CUDA(singleNum, [d_singlesAB = d_singlesAB.get(), d_singlesIn = __d_singlesIn.data(),
                              __delayWindowPico] __device__(std::size_t idx) {
      d_singlesAB[idx * 2] = d_singlesIn[idx];

      d_singlesAB[idx * 2 + 1].energy = d_singlesIn[idx].energy;
      d_singlesAB[idx * 2 + 1].timeValue_pico = d_singlesIn[idx].timeValue_pico - __delayWindowPico;
      d_singlesAB[idx * 2 + 1].globalCrystalIndex = 0x80000000 | d_singlesIn[idx].globalCrystalIndex; // Mark as B
    });
    PRINT_CUDA_ERROR {
      std::size_t bufferSize = __d_flexBuffer.elements();
      for (int i = 0; i < 2; i++)
        if (cudaSuccess != cub::DeviceMergeSort::SortKeys(__d_flexBuffer.get(), bufferSize, d_singlesAB.data(),
                                                          2 * singleNum, _OpSortSingles());
            bufferSize > __d_flexBuffer.elements())
          __d_flexBuffer = make_cuda_sync_ptr<char>(bufferSize);
        else
          break;
    }
    PRINT_CUDA_ERROR
    for_each_CUDA(2 * singleNum, [__d_singlesIn = __d_singlesIn.data(), singleABNum = d_singlesAB.elements(),
                                  __d_delayCoinNum = d_delayNum.data(), __timeWindowPico,
                                  __goodPairOp] __device__(std::size_t idx) {
      if (__d_singlesIn[idx].globalCrystalIndex & 0x80000000) // if is b
      {
        __d_delayCoinNum[idx] = 0;
        return;
      }

      unsigned delayCount = 0;
      const int64_t timeCurrent = __d_singlesIn[idx].timeValue_pico;
      for (std::size_t i = idx + 1; i < singleABNum && delayCount < 0xff; ++i) {
        const int64_t timeNext = __d_singlesIn[i].timeValue_pico;
        if (timeNext - timeCurrent > __timeWindowPico)
          break;
        if ((__d_singlesIn[i].globalCrystalIndex & 0x80000000) && __goodPairOp(__d_singlesIn[idx], __d_singlesIn[i])) {
          ++delayCount;
        }
      }
      __d_delayCoinNum[idx] = delayCount;
    });
    PRINT_CUDA_ERROR
    cuda_sync_ptr<uint64_t> d_delayNumPrefixSum = make_cuda_sync_ptr<uint64_t>(2 * singleNum + 1);
    {
      std::size_t bufferSize = __d_flexBuffer.elements();
      for (int i = 0; i < 2; i++)
        if (cudaSuccess != cub::DeviceScan::ExclusiveSum(__d_flexBuffer.data(), bufferSize, d_delayNum.data(),
                                                         d_delayNumPrefixSum.data(), d_delayNumPrefixSum.elements());
            bufferSize > __d_flexBuffer.elements())
          __d_flexBuffer = make_cuda_sync_ptr<char>(bufferSize);
        else
          break; // Thus, d_delayNumPrefixSum[__singlesNum] == total delay num.
    }
    PRINT_CUDA_ERROR
    const uint64_t delayNumTotal =
        make_vector_from_cuda_sync_ptr(
            d_delayNumPrefixSum, std::span<const uint64_t>(d_delayNumPrefixSum.end() - 1, d_delayNumPrefixSum.end()))
            .front();
    auto realDelayNum = std::min<uint64_t>(delayNumTotal, __maxDelayNum);
    for_each_CUDA(2 * singleNum, [delayCoinPrefixNum = d_delayNumPrefixSum.data(), singlesIn = __d_singlesIn.data(),
                                  singleABNum = d_singlesAB.elements(), __maxDelayNum, __goodPairOp,
                                  __d_delayCoinsOut] __device__(std::size_t idx) {
      const uint64_t delayIndexStart = delayCoinPrefixNum[idx];
      const uint64_t delayIndexEnd = delayCoinPrefixNum[idx + 1];
      const uint64_t delayNum = delayIndexEnd - delayIndexStart;
      if (delayNum == 0)
        return;
      if (delayIndexEnd > __maxDelayNum)
        return;

      const auto timeCurrent = singlesIn[idx].timeValue_pico;
      uint64_t delayCurrent = delayIndexStart;
      for (uint64_t i = idx + 1; delayCurrent < delayIndexEnd && i < singleABNum; ++i) {
        const int64_t timeNext = singlesIn[i].timeValue_pico;
        if ((singlesIn[i].globalCrystalIndex & 0x80000000) && __goodPairOp(singlesIn[idx], singlesIn[i])) {
          basic::Listmode_t coin;
          coin.time1_2pico = timeCurrent - timeNext;
          coin.globalCrystalIndex1 = singlesIn[idx].globalCrystalIndex & 0x7fffffff; // Clear the B mark
          coin.globalCrystalIndex2 = singlesIn[i].globalCrystalIndex & 0x7fffffff;   // Clear the B mark
          __d_delayCoinsOut[delayCurrent] = coin;
          delayCurrent++;
        }
      }
    });
    PRINT_CUDA_ERROR
    return realDelayNum;
  }
  struct _OpPairGoodKeepAll {
    __PNI_CUDA_MACRO__ __forceinline__ bool operator()(
        const basic::GlobalSingle_t &a, const basic::GlobalSingle_t &b) const {
      return true;
    }
  };
  template <typename OpGoodPair = _OpPairGoodKeepAll>
  [[nodiscard]] bool operator()(
      std::span<const basic::GlobalSingle_t> __d_globalSingles, std::span<unsigned> __d_countMap,
      std::span<basic::Listmode_t> &__d_prompt, std::span<basic::Listmode_t> &__d_delay, _Protocol __protocol,
      void *__d_fixedBuffer, std::size_t &__d_fixedBufferSize, cuda_sync_ptr<char> &__d_flexBuffer,
      OpGoodPair __goodPairOp = _OpPairGoodKeepAll()) const {
    auto access = _MemoryAccess(__d_globalSingles, __d_fixedBuffer, __d_fixedBufferSize);
    if (!access)
      return false;

    // Count the number of singles in each channel [ Layer 1 ]
    countSingles(__d_globalSingles.data(), __d_countMap.data(), __d_globalSingles.size());

    // Copy all singles to a continuous memory
    const auto begin = access.singlesBegin();
    for_each_CUDA(__d_globalSingles.size(), [p = begin, d_globalSingles = __d_globalSingles.data()] __device__(
                                                std::size_t idx) { p[idx] = d_globalSingles[idx]; });

    // Energy selection
    const auto end = thrust::raw_pointer_cast(&*thrust::partition(
        thrust::device_pointer_cast(begin), thrust::device_pointer_cast(begin + __d_globalSingles.size()),
        _OpSelectEnergyCoin(__protocol.energyLowThreshold_eV, __protocol.energyHighThreshold_eV)));

    const auto selectedOutEnergyNum = end - begin;
    // Count the number of singles in each channel [ Layer 2 ]
    countSingles(begin, __d_countMap.data() + __protocol.totalCrystalNum, selectedOutEnergyNum);

    // Do time coincidence
    {
      std::size_t flexBuffer = __d_flexBuffer.elements();
      for (int i = 0; i <= 2; i++)
        if (cub::DeviceMergeSort::SortKeys(__d_flexBuffer.get(), flexBuffer, begin, selectedOutEnergyNum,
                                           _OpSortSingles());
            flexBuffer > __d_flexBuffer.elements())
          __d_flexBuffer = make_cuda_sync_ptr<char>(flexBuffer);
        else
          break;
    }

    const auto realPromptNum = time_coincidence_prompt<OpGoodPair>(
        std::span<const basic::GlobalSingle_t>(begin, selectedOutEnergyNum), __d_prompt.data(),
        std::min<std::size_t>(selectedOutEnergyNum, __d_prompt.size()), __protocol.timeWindow_pico, __d_flexBuffer,
        __goodPairOp);
    const auto realDelayNum = time_coincidence_delay<OpGoodPair>(
        std::span<const basic::GlobalSingle_t>(begin, selectedOutEnergyNum), __d_delay.data(),
        std::min<std::size_t>(selectedOutEnergyNum, __d_delay.size()), __protocol.timeWindow_pico,
        __protocol.delayTimeWindow_pico, __d_flexBuffer, __goodPairOp);
    countListmode(__d_prompt.data(), __d_countMap.data() + 2 * __protocol.totalCrystalNum, realPromptNum);
    countListmode(__d_delay.data(), __d_countMap.data() + 3 * __protocol.totalCrystalNum, realDelayNum);
    __d_prompt = __d_prompt.subspan(0, realPromptNum);
    __d_delay = __d_delay.subspan(0, realDelayNum);
    return true;
  }
};
inline constexpr _StoC StoC{};
} // namespace openpni::process
