#pragma once

#include <algorithm>
#include <optional>
#include <ranges>

#include "../example/PolygonalSystem.hpp"
#include "../process/Foreach.hpp"
#include "DeadTime.hpp"
namespace openpni::process {
template <typename T>
struct Average {
  std::size_t count = 0;
  T sum = 0;
  void add(
      T value) {
    sum += value;
    count++;
  }
  std::optional<T> get() const {
    if (count > 0)
      return sum / static_cast<T>(count);
    else
      return std::nullopt;
  }
};
inline void calCryFctByFanSum(
    std::vector<float> &fanSum, const float *mich, const unsigned __radialModuleNumS,
    const example::PolygonalSystem __polygon, const basic::DetectorGeometry __detectorGeo,
    const float BadChannelThreshold = 0) // with bad channel
{
  auto blockPerRing = example::polygon::getBlockNumOneRing(__polygon, __detectorGeo);
  auto cryNumYInBlock = __detectorGeo.crystalNumV;

  auto locator = openpni::example::polygon::Locator(__polygon, __detectorGeo);
  std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
  std::vector<float> fanLOR(locator.allCrystals().size() * blockPerRing, 0);
  std::vector<float> meanDecEff(blockPerRing, 0);
  for (const auto [lor, b, v, s, cry1, cry2] : locator.allLORAndBinViewSlicesAndCrystals()) {
    int blk1 = example::polygon::calBlockInRingFromCrystalId(__polygon, __detectorGeo, cry1);
    int blk2 = example::polygon::calBlockInRingFromCrystalId(__polygon, __detectorGeo, cry2);
    blockLOR[blk2 * blockPerRing + blk1] += mich[lor];
    blockLOR[blk1 * blockPerRing + blk2] += mich[lor];
    fanLOR[cry1 * blockPerRing + blk2] += mich[lor];
    fanLOR[cry2 * blockPerRing + blk1] += mich[lor];
  }

  for (auto &blockLOREff : blockLOR) {
    blockLOREff /= (locator.rings().size() * cryNumYInBlock) * (locator.rings().size() * cryNumYInBlock);
    blockLOREff = std::max(blockLOREff, BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  }
  for (auto &fanLOREff : fanLOR) {
    fanLOREff /= cryNumYInBlock * locator.rings().size();
    fanLOREff = std::max(fanLOREff, BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  }

  for (auto groupA : std::views::iota(0u, blockPerRing)) {
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
      meanProduct += basic::FMath<float>::fsqrt(basic::FMath<float>::fsqrt(product1 * product2));
    }
    meanProduct /= __radialModuleNumS;
    meanDecEff[groupA] = meanProduct;
  }
  for (auto cry : std::views::iota(0u, locator.allCrystals().size())) {
    float meanEff = 0;
    int groupA = example::polygon::calBlockInRingFromCrystalId(__polygon, __detectorGeo, cry);
    for (auto q : std::views::iota(-__radialModuleNumS, __radialModuleNumS + 1)) {
      int anotherGroup = (groupA + blockPerRing / 2 + q + blockPerRing) % blockPerRing;
      meanEff += fanLOR[cry * blockPerRing + anotherGroup] / meanDecEff[anotherGroup];
    }
    fanSum[cry] = meanEff / (2 * __radialModuleNumS + 1);
  }
}

struct normProtocol {
  example::PolygonalSystem polygon;
  basic::DetectorGeometry detectorGeo;
  float fActCorrCutLow;      /**< Low threshold for activity correction. */
  float fActCorrCutHigh;     /**< High threshold for activity correction. */
  float fCoffCutLow;         /**< Low threshold for normalization coefficients. */
  float fCoffCutHigh;        /**< High threshold for normalization coefficients. */
  float BadChannelThreshold; /**< Bad channel threshold. */
  int radialModuleNumS;

  static normProtocol defaultE180() {
    static auto E180Polygon = example::E180();
    static auto E180Detector = openpni::device::detectorUnchangable<device::bdm2::BDM2Runtime>();
    return {
        E180Polygon,
        E180Detector.geometry,
        0.05f,  // fActCorrCutLow
        0.22f,  // fActCorrCutHigh
        0.0f,   // fCoffCutLow
        100.0f, // fCoffCutHigh
        0.02f,  // BadChannelThreshold
        4       // radialModuleNumS
    };
  }
};

namespace norm {
struct Normalization {
  explicit Normalization(
      const normProtocol &__normPrtocal)
      : m_normProtocal(__normPrtocal)
      , m_locator(m_normProtocal.polygon, m_normProtocal.detectorGeo) {}
  ~Normalization() {};

public:
  bool cutBin(
      int &binStart, int &binEnd, float *normScan_mich, float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    std::size_t availableView = 0;
    std::size_t totalGoodBin = 0;
    set_max_value_to_1(m_locator.allLORs() | m_locator.setMich(fwd_mich));
    for (const auto sl : m_locator.slices())
      for (const auto vi : m_locator.views()) {
        const std::size_t goodBinCount =
            std::ranges::count_if(m_locator.lorsForGivenViewSlice(vi, sl), [&](std::size_t LORIndex) -> bool {
              return fwd_mich[LORIndex] >= m_normProtocal.fActCorrCutLow &&
                     fwd_mich[LORIndex] <= m_normProtocal.fActCorrCutHigh;
            });
        if (goodBinCount > 0) {
          availableView += 1;
          totalGoodBin += goodBinCount;
        }
      }
    std::size_t goodBinPerView = totalGoodBin / availableView;
    goodBinPerView -= (1 - goodBinPerView % 2); // 若availableBinPerView为偶数则-1

    binStart = (binNum - goodBinPerView) / 2;
    binEnd = binStart + goodBinPerView;
    for (const auto [lor, b, v, s] : m_locator.allLORAndBinViewSlices())
      if (b < binStart || b >= binEnd)
        normScan_mich[lor] = 0;

    return true;
  }
  bool ActivityCorr(
      float *normScan_mich,
      const float *fwd_mich) // 输入筛选过 binAvailable的triangle
  {
    for (const auto LORIndex : m_locator.allLORs())
      fwd_mich[LORIndex] > 0 ? normScan_mich[LORIndex] /= fwd_mich[LORIndex] : normScan_mich[LORIndex] = 0;
    return true;
  }
  bool calCryCount(
      const float *normScan_mich) {
    std::vector<Average<double>> cryMichAverage(m_locator.allCrystals().size());
    for (const auto [lorId, crystal1, crystal2] : m_locator.allLORAndCrystals()) {
      cryMichAverage[crystal1].add(normScan_mich[lorId]);
      cryMichAverage[crystal2].add(normScan_mich[lorId]);
    }

    const auto toAverage = m_locator.allCrystals() |
                           std::views::transform([&](auto i) -> float { return cryMichAverage[i].get().value_or(0); });
    m_cryCount = std::vector<float>(toAverage.begin(), toAverage.end());
    set_max_value_to_1(m_cryCount);
    for (auto &item : m_cryCount)
      if (item < m_normProtocal.BadChannelThreshold)
        item = 0;

    return true;
  }

  bool calBlockFct() {
    // blockA
    std::vector<Average<double>> blockAAverage(m_locator.rings().size());
    for (const auto [cry, ring, inRing] : m_locator.allCrystalRectanglesAndUV())
      if (m_cryCount[cry] > 0)
        blockAAverage[ring].add(m_cryCount[cry]);
    auto rangeA = blockAAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); });
    m_blockFctA = std::vector<float>(rangeA.begin(), rangeA.end());
    set_max_value_to_1(m_blockFctA);

    // blockT
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的
    auto blockTNum = m_locator.rings().size() * cryInPanelT;
    std::vector<Average<double>> blockTAverage(blockTNum);
    for (const auto [cry, ring, inRing] : m_locator.allCrystalRectanglesAndUV()) {
      int bv = inRing % cryInPanelT;
      if (m_cryCount[cry] > 0)
        blockTAverage[ring * cryInPanelT + bv].add(m_cryCount[cry] / m_blockFctA[ring]);
    }
    auto rangeT = blockTAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); });
    m_blockFctT = std::vector<float>(rangeT.begin(), rangeT.end());
    set_max_value_to_1(m_blockFctT);
    return true;
  }

  bool calPlaneFct(
      const float *normScan_mich, const float *fwd_mich) {
    std::vector<Average<double>> everySliceAverage(m_locator.slices().size());

    for (const auto [lor, b, v, s, cry1, cry2] : m_locator.allLORAndBinViewSlicesAndCrystals())
      if (fwd_mich[lor] > 0 && m_cryCount[cry1] > 0 && m_cryCount[cry2] > 0)
        everySliceAverage[s].add(normScan_mich[lor]);
    auto fctAverage =
        everySliceAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); });
    m_planeFct = decltype(m_planeFct)(fctAverage.begin(), fctAverage.end());

    Average<double> sliceAverage;
    for (auto &value : m_planeFct)
      if (value > 0)
        sliceAverage.add(value);
      else
        value = 0;

    float meanSliceSens = sliceAverage.get().value_or(1);
    for (auto &value : m_planeFct)
      value /= meanSliceSens;

    return true;
  }

  bool calRadialFct(
      const float *normScan_mich, const float *fwd_mich) {
    std::vector<Average<double>> everyRadialAverage(m_locator.bins().size());
    for (const auto [lor, b, v, s, cry1, cry2] : m_locator.allLORAndBinViewSlicesAndCrystals())
      if (fwd_mich[lor] > 0 && m_cryCount[cry1] > 0 && m_cryCount[cry2] > 0)
        everyRadialAverage[b].add(normScan_mich[lor]);
    auto fctAverage =
        everyRadialAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); });

    m_radialFct = decltype(m_radialFct)(fctAverage.begin(), fctAverage.end());
    Average<double> radialAverage;
    for (auto &value : m_radialFct)
      if (value > 0)
        radialAverage.add(value);
      else
        value = 0;

    float binSensMean = radialAverage.get().value_or(1);
    for (auto &value : m_radialFct)
      value /= binSensMean;

    return true;
  }

  bool calInterferenceFct(
      const float *normScan_mich, const float *fwd_mich) {
    auto cryInBlockT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的

    std::vector<Average<double>> everyInterferenceAverage(m_locator.bins().size() * cryInBlockT);
    for (auto [lor, b, v, s, cry1, cry2] : m_locator.allLORAndBinViewSlicesAndCrystals()) {
      int bv = v % cryInBlockT;
      if (fwd_mich[lor] > 0 && m_cryCount[cry1] > 0 && m_cryCount[cry2] > 0)
        everyInterferenceAverage[bv * m_locator.bins().size() + b].add(normScan_mich[lor]);
    }
    auto interAverage =
        everyInterferenceAverage | std::views::transform([&](auto item) -> float { return item.get().value_or(0); });
    m_interferenceFct = decltype(m_interferenceFct)(interAverage.begin(), interAverage.end());

    Average<double> interferenceAverage;
    for (auto &value : m_interferenceFct)
      if (value > 0)
        interferenceAverage.add(value);
      else
        value = 0;

    float meanInterferenceSens = interferenceAverage.get().value_or(1);
    for (auto &value : m_interferenceFct)
      value /= meanInterferenceSens;
    return true;
  }

  bool binFctExtension(
      float *binFct, int binNum, int binStart, int binEnd, int binsToTrian) {
    auto find_first_if_or = []<typename Range, typename Pred>(Range &&r, Pred &&p, auto &&default_value) {
      auto it = std::ranges::find_if(r, p);
      if (it != r.end())
        return *it;
      else
        return default_value;
    };

    const int lEdge = find_first_if_or(
        std::ranges::views::iota(binStart, binEnd), [&](int i) { return binFct[i] > 0.8; }, binEnd - 1);
    const int rEdge = find_first_if_or(
        std::ranges::views::iota(binStart, binEnd) | std::views::reverse, [&](int i) { return binFct[i] > 0.8; },
        binStart);
    if (lEdge >= rEdge)
      return false;

    auto lf = basic::LinearFitting.withBias<float>();
    auto rf = basic::LinearFitting.withBias<float>();
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

  bool calExactCryFct(
      const float *normScan_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNum = example::polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto blockPerRing = example::polygon::getBlockNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNumYInBlock = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的
    auto crystalRingNum = example::polygon::getRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);

    std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
    std::vector<float> fanLOR(cryNum * blockPerRing, 0);
    std::vector<float> meanDecEff(blockPerRing, 0);
    m_cryFct.resize(cryNum, 0);

    std::ofstream file("test/temp_normScan_mich.raw", std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open file for writing." << std::endl;
      return false;
    }
    file.write(reinterpret_cast<const char *>(normScan_mich), sizeof(float) * binNum * viewNum * sliceNum);

    calCryFctByFanSum(m_cryFct, normScan_mich, m_normProtocal.radialModuleNumS, m_normProtocal.polygon,
                      m_normProtocal.detectorGeo, m_normProtocal.BadChannelThreshold);

    set_max_value_to_1(m_cryFct);
    for (const auto cry : std::ranges::views::iota(0u, cryNum))
      if (m_cryCount[cry] < m_normProtocal.BadChannelThreshold)
        m_cryFct[cry] = 0;

    return true;
  }
  //=====================MAIN
public:
  bool ringScannerNormFctGenerate(
      float *normScan_mich, float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNum = example::polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInBlockT = example::polygon::getCrystalNumYInPanel(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的

    int binStart, binEnd;
    if (cutBin(binStart, binEnd, normScan_mich, fwd_mich) != true) {
      std::cout << "Failed to cut bins." << std::endl;
      return false;
    }
    if (ActivityCorr(normScan_mich, fwd_mich) != true) {
      std::cout << "Failed to perform activity correction." << std::endl;
      return false;
    }
    if (calCryCount(normScan_mich) != true) {
      std::cout << "Failed to count crystals." << std::endl;
      return false;
    }
    if (calBlockFct() != true) {
      std::cout << "Failed to calculate block factors." << std::endl;
      return false;
    }
    for (auto sl = 0; sl < sliceNum; sl++) {
      for (auto vi = 0; vi < viewNum; vi++) {
        for (auto bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
          auto cryPairs = example::polygon::calRectangleFlatCrystalIDFromLORID(m_normProtocal.polygon,
                                                                               m_normProtocal.detectorGeo, LORIndex);
          int ring1 = cryPairs.x / crystalPerRing;
          int ring2 = cryPairs.y / crystalPerRing;
          int bv1 = cryPairs.x % crystalPerRing % cryInPanelT;
          int bv2 = cryPairs.y % crystalPerRing % cryInPanelT;
          auto blockFctSum = m_blockFctA[ring1] * m_blockFctA[ring2] * m_blockFctT[ring1 * cryInPanelT + bv1] *
                             m_blockFctT[ring2 * cryInPanelT + bv2];
          (blockFctSum > 0) ? normScan_mich[LORIndex] /= blockFctSum : normScan_mich[LORIndex] = 0;
        }
      }
    }
    if (calPlaneFct(normScan_mich, fwd_mich) != true) {
      std::cout << "Failed to calculate plane factors." << std::endl;
      return false;
    }
    for (int sl = 0; sl < sliceNum; sl++) {
      for (int vi = 0; vi < viewNum; vi++) {
        for (int bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
          m_planeFct[sl] > 0 ? normScan_mich[LORIndex] /= m_planeFct[sl] : normScan_mich[LORIndex] = 0;
        }
      }
    }
    if (calRadialFct(normScan_mich, fwd_mich) != true) {
      std::cerr << "Failed to calculate radial factors." << std::endl;
      return false;
    }
    if (binFctExtension(m_radialFct.data(), binNum, binStart, binEnd, 10) != true) {
      std::cerr << "Failed to extend radial factors." << std::endl;
      return false;
    }
    for (auto sl = 0; sl < sliceNum; sl++) {
      for (auto vi = 0; vi < viewNum; vi++) {
        for (auto bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
          m_radialFct[bi] > 0 ? normScan_mich[LORIndex] /= m_radialFct[bi] : normScan_mich[LORIndex] = 0;
        }
      }
    }
    if (calInterferenceFct(normScan_mich, fwd_mich) != true) {
      std::cerr << "Failed to calculate interference factors." << std::endl;
      return false;
    }
    for (int bv = 0; bv < cryInBlockT; ++bv) {
      if (binFctExtension(m_interferenceFct.data() + bv * binNum, binNum, binStart, binEnd, 10) != true) {
        std::cerr << "Failed to extend interference factors." << std::endl;
        return false;
      }
    }
    for (auto sl = 0; sl < sliceNum; sl++) {
      for (auto vi = 0; vi < viewNum; vi++) {
        int bv = vi % cryInBlockT;
        for (auto bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
          m_interferenceFct[bv * binNum + bi] > 0 ? normScan_mich[LORIndex] /= m_interferenceFct[bv * binNum + bi]
                                                  : normScan_mich[LORIndex] = 0;
        }
      }
    }
    if (calExactCryFct(normScan_mich) != true) {
      std::cerr << "Failed to calculate fanSum." << std::endl;
      return false;
    }
    return true;
  }

  bool selfNormalization(
      const float *delay_mich) {
    Normalization delayComponent(m_normProtocal);
    delayComponent.m_cryCount = decltype(m_cryCount)(m_cryCount.begin(), m_cryCount.end());
    calCryFctByFanSum(delayComponent.m_cryCount, delay_mich, 6, m_normProtocal.polygon,
                      m_normProtocal.detectorGeo); // DelayedMichFactorize
    if (delayComponent.calBlockFct() != true)
      return false;

    auto blockRingNum = example::polygon::getBlockRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto ringNum = example::polygon::getRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    int ringNumPerBlock = ringNum / blockRingNum;
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    std::vector<float> blockSumRaw(blockRingNum, 0);
    std::vector<float> blockSumSelf(blockRingNum, 0);

    for (auto i : std::views::iota(0ul, blockRingNum))
      for (auto k : std::views::iota(0, ringNumPerBlock)) {
        blockSumRaw[i] += m_blockFctA[i * ringNumPerBlock + k];
        blockSumSelf[i] += delayComponent.m_blockFctA[i * ringNumPerBlock + k];
      }

    for (auto i : std::views::iota(0ul, blockRingNum))
      for (auto k : std::views::iota(0, ringNumPerBlock))
        delayComponent.m_blockFctA[i * ringNumPerBlock + k] *= blockSumRaw[i] / blockSumSelf[i];

    float blockTSumRaw = 0;
    float blockTSumSelf = 0;
    for (auto i : std::views::iota(0ul, m_locator.rings().size() * cryInPanelT)) {
      blockTSumRaw += m_blockFctT[i];
      blockTSumSelf += delayComponent.m_blockFctT[i];
    }
    for (auto i : std::views::iota(0ul, m_locator.rings().size() * cryInPanelT))
      delayComponent.m_blockFctT[i] *= blockTSumRaw / blockTSumSelf;

    // update block profile
    m_blockFctA = delayComponent.m_blockFctA;
    m_blockFctT = delayComponent.m_blockFctT;

    return true;
  }

  bool reNormalizedByDT(
      float *normFctMich,      // DT是由table计算出来的所以类型相同
      const float *delay_mich, // randMich
      const _deadTimeDataView<double> dtProtocol) {
    auto blockRingNum = example::polygon::getBlockRingNum(dtProtocol.__polygon, dtProtocol.__detectorGeo);
    auto dsSlice = blockRingNum * blockRingNum;
    std::unique_ptr<double[]> DTComponent = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<float[]> DSdelay_mich = std::make_unique<float[]>(dsSlice);
    std::unique_ptr<double[]> Rr = std::make_unique<double[]>(dtProtocol.__acquisitionNum * dsSlice);
    std::unique_ptr<double[]> CFDT = std::make_unique<double[]>(dtProtocol.__acquisitionNum * dsSlice);
    std::unique_ptr<double[]> Rrandom = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> RcaliMax = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> RcaliMin = std::make_unique<double[]>(dsSlice);
    std::vector<double> tempRr(dtProtocol.__acquisitionNum, 0);

    double x1 = 0;
    double x2 = 0;
    double y1 = 0;
    double y2 = 0;
    // read table
    for (auto dssl : std::views::iota(0u, dsSlice))
      for (auto acq : std::views::iota(0, dtProtocol.__acquisitionNum)) {
        int index1 = dssl * dtProtocol.__acquisitionNum + acq;
        int index2 = index1 + dtProtocol.__acquisitionNum * dsSlice;

        Rr[index1] = dtProtocol.__out_CFDTTable[index1];
        CFDT[index1] = dtProtocol.__out_CFDTTable[index2];
      }

    // michDownSampling
    deadTime::_MichDownSampling::_downSamplingByBlock{DSdelay_mich.get(), delay_mich, dtProtocol.__polygon,
                                                      dtProtocol.__detectorGeo}();
    // Rrandom
    for (auto dssl : std::views::iota(0u, dsSlice)) {
      Rrandom[dssl] = DSdelay_mich[dssl] / dtProtocol.__scanTime_;
      RcaliMax[dssl] = Rr[dssl * dtProtocol.__acquisitionNum];
      RcaliMin[dssl] = Rr[(dssl + 1) * dtProtocol.__acquisitionNum - 1];
    }

    // 插值
    for (int sl = 0; sl < dsSlice; sl++) // 开始计算插值
    {
      if (Rrandom[sl] < dtProtocol.__randomRateMin) // 若小于设定值randomRatesMin，则设为1，表示不进行死时间校正
        DTComponent[sl] = 1;

      // 一般校正表随机计数率最小值会小于randomRatesMin，这分支通常进不来，
      // 但考虑实验设置等原因，可能没有采集到很低的活度，此时在randomRatesMin和RcaliMin之间做外推
      else if (Rrandom[sl] <= RcaliMin[sl]) {
        x1 = Rr[(sl + 1) * dtProtocol.__acquisitionNum - 2];
        x2 = Rr[(sl + 1) * dtProtocol.__acquisitionNum - 1];
        y1 = CFDT[(sl + 1) * dtProtocol.__acquisitionNum - 2];
        y2 = CFDT[(sl + 1) * dtProtocol.__acquisitionNum - 1];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      } else if (Rrandom[sl] < RcaliMax[sl]) // 在校正表中随机计数率最大和最小的范围内
      {
        for (int acq = 0; acq < dtProtocol.__acquisitionNum; acq++)
          tempRr[acq] = Rr[(sl + 1) * dtProtocol.__acquisitionNum - acq - 1];

        auto it = lower_bound(tempRr.begin(), tempRr.end(), Rrandom[sl]);
        int index = distance(tempRr.begin(), it);

        x1 = tempRr[index];
        x2 = tempRr[index - 1];
        y1 = CFDT[(sl + 1) * dtProtocol.__acquisitionNum - index - 1];
        y2 = CFDT[(sl + 1) * dtProtocol.__acquisitionNum - index];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      } else if (Rrandom[sl] >= RcaliMax[sl]) // 超出校正表随机计数率的最大值
      {
        x1 = Rr[sl * dtProtocol.__acquisitionNum];
        x2 = Rr[sl * dtProtocol.__acquisitionNum + 1];
        y1 = CFDT[sl * dtProtocol.__acquisitionNum];
        y2 = CFDT[sl * dtProtocol.__acquisitionNum + 1];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      }

      if (DTComponent[sl] < 1) //?
      {
        DTComponent[sl] = 1;
      }
    }

    for (auto [lor, bi, vi, sl] : m_locator.allLORAndBinViewSlices()) {
      auto [ring1, ring2] =
          example::polygon::calRing1Ring2FromSlice(dtProtocol.__polygon, dtProtocol.__detectorGeo, sl);
      int blockRing1 = ring1 / dtProtocol.__detectorGeo.crystalNumU;
      int blockRing2 = ring2 / dtProtocol.__detectorGeo.crystalNumU;
      int blockRingId = blockRing1 * blockRingNum + blockRing2;
      normFctMich[lor] /= DTComponent[blockRingId];
    }
    return true;
  }

public:
  normProtocol m_normProtocal;
  example::polygon::Locator m_locator;
  std::vector<float> m_cryCount;
  std::vector<float> m_blockFctA; // axial block profile,size recommended to be crystalRingNum
  std::vector<float> m_blockFctT; // transaxial block profile,size recommended to be crystalRingNum * cryInPanelT
  std::vector<float> m_planeFct;
  std::vector<float> m_radialFct;
  std::vector<float> m_interferenceFct;
  std::vector<float> m_cryFct; // fansum
};

} // namespace norm
} // namespace openpni::process
