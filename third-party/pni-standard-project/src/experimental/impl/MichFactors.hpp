#pragma once
#include <future>
#include <ranges>
#include <vector>

#include "Projection.h"
#include "include/IO.hpp"
#include "include/experimental/algorithms/EasyMath.hpp"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/tools/Loop.hpp"
#include "include/experimental/tools/UtilFunctions.hpp"
#include "include/io/Decoding.hpp"
#include "include/misc/CycledBuffer.hpp"
#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::node::impl {} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node::impl {
inline std::pair<std::vector<float>, std::vector<float>> calBlockLORAndFanLOR(
    core::MichDefine __mich, const float *__michValue) {
  auto michInfo = core::MichInfoHub::create(__mich);
  auto locator = core::RangeGenerator::create(__mich);
  auto indexConverter = core::IndexConverter::create(__mich);

  auto blockPerRing = michInfo.getBlockNumOneRing();
  auto cryNumYInBlock = __mich.detector.crystalNumV;

  std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
  auto spanBlockLOR = core::MDSpan<2>::create(blockPerRing, blockPerRing);
  std::vector<float> fanLOR(locator.allFlatCrystals().size() * blockPerRing, 0);
  auto spanFanLOR = core::MDSpan<2>::create(blockPerRing, locator.allFlatCrystals().size());
  PNI_DEBUG("Start to do Fan sum.\n");
  struct reduce_result {
    std::vector<float> blockLOR;
    std::vector<float> fanLOR;
  };
  std::vector<std::future<reduce_result>> futuresReduce;
  for (const auto [begin, end] :
       tools::chunked_ranges_generator.by_group_count(0, michInfo.getLORNum(), tools::cpu_threads().threadNum())) {
    futuresReduce.emplace_back(std::async(std::launch::async, [&, begin, end]() {
      reduce_result result;
      result.blockLOR.resize(blockLOR.size(), 0);
      result.fanLOR.resize(fanLOR.size(), 0);
      for (std::size_t i = begin; i < end; ++i) {
        auto [cry1, cry2] = indexConverter.getCrystalIDFromLORID(i);
        auto value = __michValue[i];
        int blk1 = cry1.idInRing / cryNumYInBlock;
        int blk2 = cry2.idInRing / cryNumYInBlock;
        result.blockLOR[spanBlockLOR(blk1, blk2)] += value;
        result.blockLOR[spanBlockLOR(blk2, blk1)] += value;
        result.fanLOR[spanFanLOR(blk2, indexConverter.getFlatIdFromRectangleId(cry1))] += value;
        result.fanLOR[spanFanLOR(blk1, indexConverter.getFlatIdFromRectangleId(cry2))] += value;
      }
      return result;
    }));
  }
  for (auto &fut : futuresReduce) {
    auto res = fut.get();
    for (std::size_t i = 0; i < blockLOR.size(); ++i)
      blockLOR[i] += res.blockLOR[i];
    for (std::size_t i = 0; i < fanLOR.size(); ++i)
      fanLOR[i] += res.fanLOR[i];
  }

  PNI_DEBUG("Fan sum calculation done.\n");
  return {blockLOR, fanLOR};
}
inline std::pair<std::vector<float>, std::vector<float>> calBlockLORAndFanLOR(
    core::MichDefine __mich, std::span<basic::Listmode_t const> __listmodes) {
  auto michInfo = core::MichInfoHub::create(__mich);
  auto locator = core::RangeGenerator::create(__mich);
  auto indexConverter = core::IndexConverter::create(__mich);

  auto blockPerRing = michInfo.getBlockNumOneRing();
  auto cryNumYInBlock = __mich.detector.crystalNumV;

  std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
  auto spanBlockLOR = core::MDSpan<2>::create(blockPerRing, blockPerRing);
  std::vector<float> fanLOR(locator.allFlatCrystals().size() * blockPerRing, 0);
  auto spanFanLOR = core::MDSpan<2>::create(blockPerRing, locator.allFlatCrystals().size());
  PNI_DEBUG("Start to do Fan sum.\n");
  struct reduce_result {
    std::vector<float> blockLOR;
    std::vector<float> fanLOR;
  };
  std::vector<std::future<reduce_result>> futuresReduce;
  for (const auto [begin, end] :
       tools::chunked_ranges_generator.by_group_count(0, __listmodes.size(), tools::cpu_threads().threadNum()))
    futuresReduce.emplace_back(std::async(std::launch::async, [&, begin, end]() {
      reduce_result result;
      result.blockLOR.resize(blockLOR.size(), 0);
      result.fanLOR.resize(fanLOR.size(), 0);
      for (std::size_t i = begin; i < end; ++i) {
        auto &listmode = __listmodes[i];
        auto cry1 = indexConverter.getRectangleIDFromUniformID(
            indexConverter.getUniformIdFromFlatId(listmode.globalCrystalIndex1));
        auto cry2 = indexConverter.getRectangleIDFromUniformID(
            indexConverter.getUniformIdFromFlatId(listmode.globalCrystalIndex2));
        auto value = 1.f;
        int blk1 = cry1.idInRing / cryNumYInBlock;
        int blk2 = cry2.idInRing / cryNumYInBlock;
        result.blockLOR[spanBlockLOR(blk1, blk2)] += value;
        result.blockLOR[spanBlockLOR(blk2, blk1)] += value;
        result.fanLOR[spanFanLOR(blk2, indexConverter.getFlatIdFromRectangleId(cry1))] += value;
        result.fanLOR[spanFanLOR(blk1, indexConverter.getFlatIdFromRectangleId(cry2))] += value;
      }
      return result;
    }));
  for (auto &fut : futuresReduce) {
    auto res = fut.get();
    for (std::size_t i = 0; i < blockLOR.size(); ++i)
      blockLOR[i] += res.blockLOR[i];
    for (std::size_t i = 0; i < fanLOR.size(); ++i)
      fanLOR[i] += res.fanLOR[i];
  }

  PNI_DEBUG("Fan sum calculation done.\n");
  return {blockLOR, fanLOR};
}

inline std::vector<float> calCryFctByFanSum(
    core::MichDefine __mich, std::vector<float> const &__fanLOR, std::vector<float> const &__blockLOR,
    const unsigned __radialModuleNumS,
    const float BadChannelThreshold = 0) // with bad channel
{
  std::vector<float> fanSum(MichInfoHub(__mich).getTotalCrystalNum(), 0);

  auto michInfo = core::MichInfoHub::create(__mich);
  auto locator = core::RangeGenerator::create(__mich);
  auto indexConverter = core::IndexConverter::create(__mich);

  auto blockPerRing = michInfo.getBlockNumOneRing();
  auto cryNumYInBlock = __mich.detector.crystalNumV;

  std::vector<float> meanDecEff(blockPerRing, 0);
  auto blockLOR = __blockLOR;
  auto fanLOR = __fanLOR;

  tools::parallel_for_each(blockLOR.size(), [&](std::size_t index) {
    auto &blockLOREff = blockLOR[index];
    blockLOREff /= (locator.allRings().size() * cryNumYInBlock) * (locator.allRings().size() * cryNumYInBlock);
    blockLOREff = std::max(blockLOREff, BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  });
  tools::parallel_for_each(fanLOR.size(), [&](std::size_t index) {
    auto &fanLOREff = fanLOR[index];
    fanLOREff /= cryNumYInBlock * locator.allRings().size();
    fanLOREff = std::max(fanLOREff, BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  });

  tools::parallel_for_each(blockPerRing, [&](std::size_t groupA) {
    float meanProduct = 0;
    for (auto p : std::views::iota(0u, __radialModuleNumS)) {
      int anotherGroup = (groupA + blockPerRing / 2 + p + blockPerRing) % blockPerRing;
      float product1 = blockLOR[groupA + anotherGroup * blockPerRing];
      anotherGroup = (groupA + blockPerRing / 2 - p - 1 + blockPerRing) % blockPerRing;
      float product2 = blockLOR[groupA + anotherGroup * blockPerRing];
      for (auto k : std::views::iota(0u, blockPerRing / 2 + p)) {
        int groupA_k = (groupA + k + blockPerRing) % blockPerRing;
        int groupA_k_1 = (groupA + k + 1 + blockPerRing) % blockPerRing;
        anotherGroup = (groupA + k + blockPerRing / 2 - p + blockPerRing) % blockPerRing;
        product1 *=
            blockLOR[groupA_k + anotherGroup * blockPerRing] / blockLOR[groupA_k_1 + anotherGroup * blockPerRing];
      }
      for (auto k : std::views::iota(0u, blockPerRing / 2 - p - 1)) {
        int groupA_k = (groupA + k + blockPerRing) % blockPerRing;
        int groupA_k_1 = (groupA + k + 1 + blockPerRing) % blockPerRing;
        anotherGroup = (groupA + k + blockPerRing / 2 + p + 1 + blockPerRing) % blockPerRing;
        product2 *=
            blockLOR[groupA_k + anotherGroup * blockPerRing] / blockLOR[groupA_k_1 + anotherGroup * blockPerRing];
      }
      meanProduct += core::FMath<float>::fsqrt(core::FMath<float>::fsqrt(product1 * product2));
    }
    meanProduct /= __radialModuleNumS;
    meanDecEff[groupA] = meanProduct;
  });
  tools::parallel_for_each(michInfo.getTotalCrystalNum(), [&](std::size_t cry) {
    auto [ring, idInRing] = indexConverter.getRectangleIdFromFlatId(cry);
    float meanEff = 0;
    int groupA = idInRing / cryNumYInBlock;
    for (auto q : std::views::iota(-__radialModuleNumS, __radialModuleNumS + 1)) {
      int anotherGroup = (groupA + blockPerRing / 2 + q + blockPerRing) % blockPerRing;
      meanEff += fanLOR[cry * blockPerRing + anotherGroup] / meanDecEff[anotherGroup];
    }
    fanSum[cry] = meanEff / (2 * __radialModuleNumS + 1);
  });
  return fanSum;
}
inline core::Vector<int, 2> findGoodBinRange(
    // Return binStart, binEnd
    core::MichDefine __mich, float *normScan_mich, float *fwd_mich, float fActCorrCutLow, float fActCorrCutHigh) {
  const auto binNum = MichInfoHub(__mich).getBinNum();
  std::size_t availableView = 0;
  std::size_t totalGoodBin = 0;
  algorithms::set_max_value_to_1(std::span<float>(fwd_mich, MichInfoHub(__mich).getLORNum()));
  for (const auto [sl, ring1, ring2] : RangeGenerator(__mich).allSlices())
    for (const auto vi : RangeGenerator(__mich).allViews()) {
      const std::size_t goodBinCount = std::ranges::count_if(
          RangeGenerator(__mich).lorsForGivenViewSlice(vi, sl), [&](std::size_t LORIndex) noexcept -> bool {
            return fwd_mich[LORIndex] >= fActCorrCutLow && fwd_mich[LORIndex] <= fActCorrCutHigh;
          });
      if (goodBinCount > 0) {
        availableView += 1;
        totalGoodBin += goodBinCount;
      }
    }
  std::size_t goodBinPerView = totalGoodBin / availableView;
  goodBinPerView -= (1 - goodBinPerView % 2); // 若availableBinPerView为偶数则-1

  int binStart = (binNum - goodBinPerView) / 2;
  int binEnd = binStart + goodBinPerView;
  return core::Vector<int, 2>{binStart, binEnd};
}
inline std::vector<float> distributeMichToCrystalCounts(
    core::MichDefine __mich, const float *normScan_mich) {
  std::vector<algorithms::AverageHelper<double>> cryMichAverage(MichInfoHub(__mich).getTotalCrystalNum());
  for (const auto [lorId, crystal1, rid1, crystal2, rid2] : RangeGenerator(__mich).allLORAndRectangleCrystalsFlat()) {
    cryMichAverage[crystal1].add(normScan_mich[lorId]);
    cryMichAverage[crystal2].add(normScan_mich[lorId]);
  }

  const auto toAverage = cryMichAverage | std::views::transform([&](auto i) -> float { return i.get().value_or(0); });
  std::vector<float> result = std::vector<float>(toAverage.begin(), toAverage.end());
  return result;
}
inline std::pair<std::vector<float>, std::vector<float>> calBlockFct(
    // Return: [blockFctA, blockFctT]
    core::MichDefine __mich, float const *__cryCount) {
  // blockA
  std::vector<algorithms::AverageHelper<double>> blockAAverage(MichInfoHub(__mich).getRingNum());
  for (const auto [cry, ring, inRing] : RangeGenerator(__mich).allFlatAndRectangleCrystals())
    if (__cryCount[cry] > 0)
      blockAAverage[ring].add(__cryCount[cry]);
  std::vector<float> blockFctA = tools::to_vector(
      blockAAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); }));
  algorithms::set_max_value_to_1(std::span<float>(blockFctA));

  // blockT
  auto cryInPanelT =
      MichInfoHub(__mich).getCrystalNumYInPanel(); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的
  auto blockTNum = MichInfoHub(__mich).getRingNum() * cryInPanelT;
  std::vector<algorithms::AverageHelper<double>> blockTAverage(blockTNum);
  for (const auto [cry, ring, inRing] : RangeGenerator(__mich).allFlatAndRectangleCrystals()) {
    int bv = inRing % cryInPanelT;
    if (__cryCount[cry] > 0)
      blockTAverage[ring * cryInPanelT + bv].add(__cryCount[cry] / blockFctA[ring]);
  }
  std::vector<float> blockFctT = tools::to_vector(
      blockTAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); }));
  algorithms::set_max_value_to_1(std::span<float>(blockFctT));

  return {blockFctA, blockFctT};
}
inline std::vector<float> calPlaneFct(
    core::MichDefine __mich, float const *__fwd_mich, float const *__normScan_mich, float const *__cryCount) {
  std::vector<algorithms::AverageHelper<double>> everySliceAverage(MichInfoHub(__mich).getSliceNum());

  for (const auto [lor, b, v, s, cry1, cry2] : RangeGenerator(__mich).allLORAndBinViewSlicesAndFlatCrystals())
    if (__fwd_mich[lor] > 0 && __cryCount[cry1] > 0 && __cryCount[cry2] > 0)
      everySliceAverage[s].add(__normScan_mich[lor]);

  auto planeFct = tools::to_vector(everySliceAverage |
                                   std::views::transform([&](auto item) -> float { return item.get().value_or(0); }));

  double sliceAverageValue = algorithms::AverageHelper<double>::apply(
      planeFct | std::views::filter([](auto v) noexcept { return v > 0; }), 1.0);
  for (auto &value : planeFct)
    value /= sliceAverageValue;
  return planeFct;
}
inline std::vector<float> calRadialFct(
    core::MichDefine __mich, float const *__fwd_mich, float const *__normScan_mich, float const *__cryCount) {
  std::vector<algorithms::AverageHelper<double>> everyRadialAverage(MichInfoHub(__mich).getBinNum());
  for (const auto [lor, b, v, s, cry1, cry2] : RangeGenerator(__mich).allLORAndBinViewSlicesAndFlatCrystals())
    if (__fwd_mich[lor] > 0 && __cryCount[cry1] > 0 && __cryCount[cry2] > 0)
      everyRadialAverage[b].add(__normScan_mich[lor]);

  auto radialFct = tools::to_vector(everyRadialAverage |
                                    std::views::transform([&](auto item) -> float { return item.get().value_or(0); }));
  auto binSensMean = algorithms::AverageHelper<float>::apply(
      radialFct | std::views::filter([](auto v) noexcept { return v > 0; }), 1.0f);
  for (auto &value : radialFct)
    value /= binSensMean;
  return radialFct;
}
inline std::vector<float> calInterferenceFct(
    core::MichDefine __mich, float const *__fwd_mich, float const *__normScan_mich, float const *__cryCount) {
  auto cryInBlockT =
      MichInfoHub(__mich).getCrystalNumYInPanel(); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的

  std::vector<algorithms::AverageHelper<double>> everyInterferenceAverage(MichInfoHub(__mich).getBinNum() *
                                                                          cryInBlockT);
  for (auto [lor, b, v, s, cry1, cry2] : RangeGenerator(__mich).allLORAndBinViewSlicesAndFlatCrystals()) {
    int bv = v % cryInBlockT;
    if (__fwd_mich[lor] > 0 && __cryCount[cry1] > 0 && __cryCount[cry2] > 0)
      everyInterferenceAverage[bv * MichInfoHub(__mich).getBinNum() + b].add(__normScan_mich[lor]);
  }

  auto interferenceFct = tools::to_vector(
      everyInterferenceAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); }));

  auto meanInterferenceSens = algorithms::AverageHelper<float>::apply(
      interferenceFct | std::views::filter([](auto v) noexcept { return v > 0; }), 1.0f);
  for (auto &value : interferenceFct)
    value /= meanInterferenceSens;
  return interferenceFct;
}
inline bool binFctExtension(
    float *binFct, int binNum, int binStart, int binEnd, int binsToTrian) {

  const int lEdge = tools::find_first_if_or(
      std::ranges::views::iota(binStart, binEnd), [&](int i) noexcept { return binFct[i] > 0.8; }, binEnd - 1);
  const int rEdge = tools::find_first_if_or(
      std::ranges::views::iota(binStart, binEnd) | std::views::reverse, [&](int i) noexcept { return binFct[i] > 0.8; },
      binStart);
  if (lEdge >= rEdge)
    return false;

  auto lf = algorithms::LinearFittingHelper<float, algorithms::LinearFitting_WithBias>();
  auto rf = algorithms::LinearFittingHelper<float, algorithms::LinearFitting_WithBias>();
  for (int i = 0; i < binsToTrian; ++i) {
    lf.add(lEdge + i, binFct[lEdge + i]);
    rf.add(rEdge - i, binFct[rEdge - i]);
  }
  for (int x = 0; x < lEdge + binsToTrian; ++x)
    binFct[x] = std::max(.6f, lf.predict(x));
  for (int x = rEdge + 1 - binsToTrian; x < binNum; ++x)
    binFct[x] = std::max(.6f, rf.predict(x));
  return true;
}
inline std::vector<float> calExactCryFct(
    core::MichDefine __mich, float const *__normScan_mich, float const *__cryCount, const unsigned __radialModuleNumS,
    const float BadChannelThreshold) {
  auto cryNum = MichInfoHub(__mich).getTotalCrystalNum();

  auto [blockLOR, fanLOR] = calBlockLORAndFanLOR(__mich, __normScan_mich);
  auto cryFct = calCryFctByFanSum(__mich, fanLOR, blockLOR, __radialModuleNumS, BadChannelThreshold);

  algorithms::set_max_value_to_1(std::span<float>(cryFct));
  tools::parallel_for_each(cryNum, [&](std::size_t cry) {
    if (__cryCount[cry] < BadChannelThreshold)
      cryFct[cry] = 0;
  });
  return cryFct;
}
inline auto selfNormalization(
    core::MichDefine __mich, std::vector<float> const &__blockLOR, std::vector<float> const &__fanLOR,
    std::vector<float> const &__blockFctA, std::vector<float> const &__blockFctT, float __badChannelThreshold) {
  auto resultCryCount =
      calCryFctByFanSum(__mich, __fanLOR, __blockLOR, 6, __badChannelThreshold); // DelayedMichFactorize
  auto [resultBlockFctA, resultBlockFctT] = calBlockFct(__mich, resultCryCount.data());

  auto blockRingNum = MichInfoHub(__mich).getBlockRingNum();
  auto ringNum = MichInfoHub(__mich).getRingNum();
  int ringNumPerBlock = ringNum / blockRingNum;
  auto cryInPanelT = MichInfoHub(__mich).getCrystalNumYInPanel();
  std::vector<float> blockSumRaw(blockRingNum, 0);
  std::vector<float> blockSumSelf(blockRingNum, 0);

  for (auto i : std::views::iota(0ul, blockRingNum))
    for (auto k : std::views::iota(0, ringNumPerBlock)) {
      blockSumRaw[i] += __blockFctA[i * ringNumPerBlock + k];
      blockSumSelf[i] += resultBlockFctA[i * ringNumPerBlock + k];
    }

  for (auto i : std::views::iota(0ul, blockRingNum))
    for (auto k : std::views::iota(0, ringNumPerBlock))
      resultBlockFctA[i * ringNumPerBlock + k] *= blockSumRaw[i] / blockSumSelf[i];

  float blockTSumRaw = 0;
  float blockTSumSelf = 0;
  for (auto i : std::views::iota(0ul, MichInfoHub(__mich).getRingNum() * cryInPanelT)) {
    blockTSumRaw += __blockFctT[i];
    blockTSumSelf += resultBlockFctT[i];
  }
  for (auto i : std::views::iota(0ul, MichInfoHub(__mich).getRingNum() * cryInPanelT))
    resultBlockFctT[i] *= blockTSumRaw / blockTSumSelf;

  // update block profile
  return std::make_tuple(resultCryCount, resultBlockFctA, resultBlockFctT);
}

inline std::vector<float> distributeMichToBlockMich(
    core::MichDefine __mich, float const *__michValues) {
  auto blockRingNum = MichInfoHub(__mich).getBlockRingNum();
  auto dsSlice = blockRingNum * blockRingNum;
  std::vector<float> blockMich(dsSlice, 0);
  for (auto [lor, bi, vi, sl] : RangeGenerator(__mich).allLORAndBinViewSlices()) {
    auto [ring1, ring2] = IndexConverter(__mich).getRing1Ring2FromSlice(sl);
    int blockRing1 = ring1 / MichInfoHub(__mich).getCrystalNumZInBlock();
    int blockRing2 = ring2 / MichInfoHub(__mich).getCrystalNumZInBlock();
    int blockRingID = blockRing1 * blockRingNum + blockRing2;
    blockMich[blockRingID] += __michValues[lor];
  }
  return blockMich;
}

inline std::vector<float> calDTComponent(
    core::MichDefine __michDefine, DeadTimeTable __dtTable) {
  auto blockRingNum = MichInfoHub(__michDefine).getBlockRingNum();
  auto dsSlice = blockRingNum * blockRingNum;
  auto aquisitionNum = std::min(__dtTable.CTDTTable.size(), __dtTable.RTTable.size()) / dsSlice;
  std::unique_ptr<double[]> DTComponent = std::make_unique<double[]>(dsSlice);
  std::unique_ptr<double[]> Rr = std::make_unique<double[]>(aquisitionNum * dsSlice);
  std::unique_ptr<double[]> CFDT = std::make_unique<double[]>(aquisitionNum * dsSlice);
  std::unique_ptr<double[]> Rrandom = std::make_unique<double[]>(dsSlice);
  std::unique_ptr<double[]> RcaliMax = std::make_unique<double[]>(dsSlice);
  std::unique_ptr<double[]> RcaliMin = std::make_unique<double[]>(dsSlice);
  std::vector<double> tempRr(aquisitionNum, 0);

  double x1 = 0;
  double x2 = 0;
  double y1 = 0;
  double y2 = 0;
  // read table
  for (auto dssl : std::views::iota(0u, dsSlice))
    for (auto acq : std::views::iota(0ull, aquisitionNum)) {
      int index1 = dssl * aquisitionNum + acq;
      // int index2 = index1 + aquisitionNum * dsSlice;

      // Rr[index1] = dtProtocol.__out_CFDTTable[index1];
      // CFDT[index1] = dtProtocol.__out_CFDTTable[index2];
      Rr[index1] = __dtTable.RTTable[index1];
      CFDT[index1] = __dtTable.CTDTTable[index1];
    }

  // michDownSampling
  auto DSdelay_mich = distributeMichToBlockMich(__michDefine, __dtTable.delayMich);
  // Rrandom
  for (auto dssl : std::views::iota(0u, dsSlice)) {
    Rrandom[dssl] = DSdelay_mich[dssl] / __dtTable.scanTime;
    RcaliMax[dssl] = Rr[dssl * aquisitionNum];
    RcaliMin[dssl] = Rr[(dssl + 1) * aquisitionNum - 1];
  }

  // 插值
  for (int sl = 0; sl < dsSlice; sl++) // 开始计算插值
  {
    if (Rrandom[sl] < __dtTable.randomRateMin) // 若小于设定值randomRatesMin，则设为1，表示不进行死时间校正
      DTComponent[sl] = 1;

    // 一般校正表随机计数率最小值会小于randomRatesMin，这分支通常进不来，
    // 但考虑实验设置等原因，可能没有采集到很低的活度，此时在randomRatesMin和RcaliMin之间做外推
    else if (Rrandom[sl] <= RcaliMin[sl]) {
      x1 = Rr[(sl + 1) * aquisitionNum - 2];
      x2 = Rr[(sl + 1) * aquisitionNum - 1];
      y1 = CFDT[(sl + 1) * aquisitionNum - 2];
      y2 = CFDT[(sl + 1) * aquisitionNum - 1];
      DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
    } else if (Rrandom[sl] < RcaliMax[sl]) // 在校正表中随机计数率最大和最小的范围内
    {
      for (int acq = 0; acq < aquisitionNum; acq++)
        tempRr[acq] = Rr[(sl + 1) * aquisitionNum - acq - 1];

      auto it = lower_bound(tempRr.begin(), tempRr.end(), Rrandom[sl]);
      int index = distance(tempRr.begin(), it);

      x1 = tempRr[index];
      x2 = tempRr[index - 1];
      y1 = CFDT[(sl + 1) * aquisitionNum - index - 1];
      y2 = CFDT[(sl + 1) * aquisitionNum - index];
      DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
    } else if (Rrandom[sl] >= RcaliMax[sl]) // 超出校正表随机计数率的最大值
    {
      x1 = Rr[sl * aquisitionNum];
      x2 = Rr[sl * aquisitionNum + 1];
      y1 = CFDT[sl * aquisitionNum];
      y2 = CFDT[sl * aquisitionNum + 1];
      DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
    }

    if (DTComponent[sl] < 1) //?
    {
      DTComponent[sl] = 1;
    }
  }

  auto dtComponent =
      std::vector<float>(MichInfoHub(__michDefine).getBlockRingNum() * MichInfoHub(__michDefine).getBlockRingNum());
  for (auto [lor, bi, vi, sl] : RangeGenerator(__michDefine).allLORAndBinViewSlices()) {
    auto [ring1, ring2] = IndexConverter(__michDefine).getRing1Ring2FromSlice(sl);
    int blockRing1 = ring1 / MichInfoHub(__michDefine).getCrystalNumZInBlock();
    int blockRing2 = ring2 / MichInfoHub(__michDefine).getCrystalNumZInBlock();
    int blockRingId = blockRing1 * blockRingNum + blockRing2;
    dtComponent[blockRingId] = 1.0f / DTComponent[blockRingId];
  }
  return dtComponent;
}

} // namespace openpni::experimental::node::impl

namespace openpni::experimental::node::impl {
__PNI_CUDA_MACRO__ inline float calNormFactorsAll(
    core::MichDefine __mich, std::size_t __lorIndex, float const *__cryFct, float const *__blockFctA,
    float const *__blockFctT, float const *__planeFct, float const *__radialFct, float const *__interferenceFct,
    float const *__dtComponent, FactorBitMask __factorBitMask) {
  auto crystalPerRing = MichInfoHub(__mich).getCrystalNumOneRing();
  auto cryInPlanelT = MichInfoHub(__mich).getCrystalNumYInPanel();
  auto ringsPerBlock = MichInfoHub(__mich).getCrystalNumZInBlock();
  auto blockRingNum = MichInfoHub(__mich).getBlockRingNum();
  auto binNum = MichInfoHub(__mich).getBinNum();

  auto [bi, vi, sl] = IndexConverter(__mich).getBVSFromLOR(__lorIndex);
  auto [rid1, rid2] = IndexConverter(__mich).getCrystalIDFromLORID(__lorIndex);
  auto cry1 = IndexConverter(__mich).getFlatIdFromRectangleId(rid1);
  auto cry2 = IndexConverter(__mich).getFlatIdFromRectangleId(rid2);

  int ring1 = rid1.ringID;
  int ring2 = rid2.ringID;
  int bv1 = cry1 % crystalPerRing % cryInPlanelT;
  int bv2 = cry2 % crystalPerRing % cryInPlanelT;
  int bv = vi % cryInPlanelT;
  float cryFct = __cryFct[cry1] * __cryFct[cry2];
  float blockFct = __blockFctA[ring1] * __blockFctA[ring2] * __blockFctT[ring1 * cryInPlanelT + bv1] *
                   __blockFctT[ring2 * cryInPlanelT + bv2];
  float radialFct = __radialFct[bi];
  float planeFct = __planeFct[sl];
  float interFct = __interferenceFct[bv * binNum + bi];

  int blockRing1 = ring1 / ringsPerBlock;
  int blockRing2 = ring2 / ringsPerBlock;
  int blockRingID = blockRing1 * blockRingNum + blockRing2;
  float deadTimeFct = __dtComponent ? __dtComponent[blockRingID] : 1.0f;
  float result = 1.0f;
  if (__factorBitMask & CryFct)
    result *= cryFct;
  if (__factorBitMask & BlockFct)
    result *= blockFct;
  if (__factorBitMask & RadialFct)
    result *= radialFct;
  if (__factorBitMask & PlaneFct)
    result *= planeFct;
  if (__factorBitMask & InterferenceFct)
    result *= interFct;
  if (__factorBitMask & DTComponent)
    result *= deadTimeFct;
  return result;
}

void d_calNormFactorsAll(core::MichDefine __mich, std::span<core::MichStandardEvent const> events, float *__output,
                         float const *__cryFct, float const *__blockFctA, float const *__blockFctT,
                         float const *__planeFct, float const *__radialFct, float const *__interferenceFct,
                         float const *__dtComponent, FactorBitMask __factorBitMask);
void d_getDScatterFactorsBatch(float *__outFct, const float *__sssValue,
                               std::span<core::MichStandardEvent const> events, core::MichDefine __michDefine);
void d_getDScatterFactorsBatchTOF(float *__outFct, const float *__sssTOFTable,
                                  const core::CrystalGeom *__d_crystalGeometry,
                                  const core::CrystalGeom *__d_dsCrystalGeometry,
                                  std::span<core::MichStandardEvent const> __events, core::MichDefine __michDefine,
                                  core::MichDefine __dsmichDefine, float __tofBinWidth, int __tofBinNum);
} // namespace openpni::experimental::node::impl