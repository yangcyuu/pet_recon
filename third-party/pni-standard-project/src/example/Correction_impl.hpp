#pragma once
#include "include/basic/DataView.hpp"
#include "include/example/PolygonalSystem.hpp"
#include "include/math/EMStep.hpp"
namespace openpni::example::polygon::corrections // common functions
{
inline void cryFctByfanSum(
    std::vector<float> &fanSum, const float *mich, const unsigned radialModuleNumS,
    const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo,
    const float BadChannelThreshold = 0) // with bad channel
{
  auto binNum = example::polygon::getBinNum(polygon, detectorGeo);
  auto viewNum = example::polygon::getViewNum(polygon, detectorGeo);
  auto sliceNum = example::polygon::getSliceNum(polygon, detectorGeo);
  auto cryNum = polygon::getTotalCrystalNum(polygon, detectorGeo);
  auto crystalPerRing = example::polygon::getCrystalNumOneRing(polygon, detectorGeo);
  auto blockPerRing = example::polygon::getBlockNumOneRing(polygon, detectorGeo);
  auto cryNumYInBlock = detectorGeo.crystalNumV;
  auto crystalRingNum = example::polygon::getRingNum(polygon, detectorGeo);
  std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
  std::vector<float> fanLOR(cryNum * blockPerRing, 0);
  std::vector<float> meanDecEff(blockPerRing, 0);

  for (auto sl = 0; sl < sliceNum; sl++) {
    for (auto vi = 0; vi < viewNum; vi++) {
      for (auto bi = 0; bi < binNum; bi++) {
        size_t LORIndex = size_t(bi + vi * binNum) + size_t(sl) * size_t(binNum * viewNum);
        auto cryPairs = example::polygon::calRectangleFlatCrystalIDFromLORID(polygon, detectorGeo, LORIndex);
        int blk1 = example::polygon::calBlockInRingFromCrystalId(polygon, detectorGeo, cryPairs.x);
        int blk2 = example::polygon::calBlockInRingFromCrystalId(polygon, detectorGeo, cryPairs.y);
        blockLOR[blk2 * blockPerRing + blk1] += mich[LORIndex];
        blockLOR[blk1 * blockPerRing + blk2] += mich[LORIndex];
        fanLOR[cryPairs.x * blockPerRing + blk2] += mich[LORIndex];
        fanLOR[cryPairs.y * blockPerRing + blk1] += mich[LORIndex];
      }
    }
  }
  for (auto &blockLOREff : blockLOR) {
    blockLOREff /= (crystalRingNum * cryNumYInBlock) * (crystalRingNum * cryNumYInBlock);
    blockLOREff = std::max(blockLOREff,
                           BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  }
  for (auto &fanLOREff : fanLOR) {
    fanLOREff /= cryNumYInBlock * crystalRingNum;
    fanLOREff = std::max(fanLOREff,
                         BadChannelThreshold); // 这里怎么再阈值问题上不等于0 而是取BadChannelThreshold？
  }

  int S = radialModuleNumS;

  for (int groupA = 0; groupA < blockPerRing; groupA++) {
    float meanProduct = 0;
    for (int p = 0; p <= S - 1; ++p) {
      int anotherGroup = (groupA + blockPerRing / 2 + p + blockPerRing) % blockPerRing;
      float product1 = blockLOR[groupA + anotherGroup * blockPerRing];

      anotherGroup = (groupA + blockPerRing / 2 - p - 1 + blockPerRing) % blockPerRing;
      float product2 = blockLOR[groupA + anotherGroup * blockPerRing];

      for (int k = 0; k <= blockPerRing / 2 + p - 1; ++k) {
        int groupA_k = (groupA + k + blockPerRing) % blockPerRing;
        int groupA_k_1 = (groupA + k + 1 + blockPerRing) % blockPerRing;
        anotherGroup = (groupA + k + blockPerRing / 2 - p + blockPerRing) % blockPerRing;
        product1 *=
            blockLOR[groupA_k + anotherGroup * blockPerRing] / blockLOR[groupA_k_1 + anotherGroup * blockPerRing];
      }
      for (int k = 0; k <= blockPerRing / 2 - p - 2; ++k) {
        int groupA_k = (groupA + k + blockPerRing) % blockPerRing;
        int groupA_k_1 = (groupA + k + 1 + blockPerRing) % blockPerRing;
        anotherGroup = (groupA + k + blockPerRing / 2 + p + 1 + blockPerRing) % blockPerRing;
        product2 *=
            blockLOR[groupA_k + anotherGroup * blockPerRing] / blockLOR[groupA_k_1 + anotherGroup * blockPerRing];
      }
      meanProduct += fsqrt(fsqrt(product1 * product2));
    }
    meanProduct /= S;
    meanDecEff[groupA] = meanProduct;
  }
  for (int crystalId = 0; crystalId < cryNum; crystalId++) {
    float meanEff = 0;
    int groupA = example::polygon::calBlockInRingFromCrystalId(polygon, detectorGeo, crystalId);
    for (int q = -S; q <= S; q++) {
      int anotherGroup = (groupA + blockPerRing / 2 + q + blockPerRing) % blockPerRing;
      meanEff += fanLOR[crystalId * blockPerRing + anotherGroup] / meanDecEff[anotherGroup];
    }
    fanSum[crystalId] = meanEff / (2 * S + 1);
  }
}
} // namespace openpni::example::polygon::corrections

namespace openpni::example::polygon::corrections // correction functions
{
class AttnCorrection::AttnCorrection_impl {
public:
  bool AttnMapToAttnCoffAt511keV(
      float *attnCoff, const float *attnMap,
      const openpni::basic::Image3DGeometry &attnMap3dSize, // same as petImg
      const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
    const auto totalDetectorNum = polygon.totalDetectorNum();
    const auto crystalNumInDetector = detectorGeo.getTotalCrystalNum();
    auto LORNum = example::polygon::getLORNum(polygon, detectorGeo);
    std::unique_ptr<float[]> attn_fwdMich = std::make_unique<float[]>(LORNum);

    std::vector<openpni::basic::Vec3<float>> cryPos;
    std::vector<basic::Coordinate3D<float>> detectorCoordinatesWithDirection;
    for (const auto detectorIndex : std::views::iota(0u, totalDetectorNum)) {
      detectorCoordinatesWithDirection.push_back(
          openpni::example::coordinateFromPolygon(polygon, detectorIndex / (polygon.detectorPerEdge * polygon.edges),
                                                  detectorIndex % (polygon.detectorPerEdge * polygon.edges)));
      for (const auto crystalIndex : std::views::iota(0u, crystalNumInDetector)) {
        const auto coord = openpni::basic::calculateCrystalGeometry(detectorCoordinatesWithDirection.back(),
                                                                    detectorGeo, crystalIndex);
        cryPos.push_back(coord.position);
      }
    }

    // EM dataset
    openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float, float> dataSetForEMSum;
    dataSetForEMSum.qtyValue = nullptr;
    dataSetForEMSum.crystalPosition = cryPos.data();
    dataSetForEMSum.indexer.scanner = polygon;
    dataSetForEMSum.indexer.detector = detectorGeo;
    dataSetForEMSum.indexer.subsetId = 0;
    dataSetForEMSum.indexer.subsetNum = 1;
    dataSetForEMSum.indexer.binCut = 0; // no bin cut
    process::EMSum(Image3DInputSpan{attnMap3dSize, attnMap}, attnMap3dSize.roi(), attn_fwdMich.get(), dataSetForEMSum,
                   openpni::math::ProjectionMethodSiddon(), basic::CpuMultiThread::callWithAllThreads());
    //

    for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
      attnCoff[LORIndex] = exp(-attn_fwdMich[LORIndex]);
    }
    return true;
  }
};

// norm

class NormCorrection::NormCorrection_impl {
public:
  explicit NormCorrection_impl(const NormCorrection::normCorrProtocal &__normPrtocal)
      : m_normProtocal(__normPrtocal) {};

public:
  bool ringScannerNormFctGenerate(
      float *normScan_mich, float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNumPerSlice = binNum * viewNum;
    auto cryNum = polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInBlockT = example::polygon::getCrystalNumYInPanel(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的

    int binStart, binEnd;
    bool status;

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
          auto cryPairs =
              example::polygon::calCrystalIDFromLORID(m_normProtocal.polygon, m_normProtocal.detectorGeo, LORIndex);
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
    auto ringNum = example::polygon::getRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNum = example::polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto blockRingNum = example::polygon::getBlockRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(m_normProtocal.polygon, m_normProtocal.detectorGeo);

    NormCorrection delayComponent(m_normProtocal);

    auto &deylayCryCount = delayComponent.getCryCount();
    auto &delayBlockFctA = delayComponent.getBlockFctA();
    auto &delayBlockFctT = delayComponent.getBlockFctT();
    deylayCryCount.resize(cryNum, 0);
    cryFctByfanSum(deylayCryCount, delay_mich, 6, m_normProtocal.polygon,
                   m_normProtocal.detectorGeo); // DelayedMichFactorize
    if (delayComponent.BlockFct() != true) {
      std::cout << "Failed to calculate delayed block factors." << std::endl;
      return false;
    } // DelayedBlockFactorize
    std::cout << "regenerate block fct..." << std::endl;
    std::vector<float> blockSumRaw(blockRingNum, 0);
    std::vector<float> blockSumSelf(blockRingNum, 0);
    for (auto i = 0; i < blockRingNum; i++) {
      for (auto k = 0; k < cryInPanelT; k++) {
        blockSumRaw[i] += m_blockFctA[i * cryInPanelT + k];
        blockSumSelf[i] += delayBlockFctA[i * cryInPanelT + k];
      }
    }
    for (auto i = 0; i < blockRingNum; i++) {
      for (auto k = 0; k < cryInPanelT; k++) {
        delayBlockFctA[i * cryInPanelT + k] *= blockSumRaw[i] / blockSumSelf[i];
      }
    }

    float blockTSumRaw = 0, blockTSumSelf = 0;
    for (auto i = 0; i < ringNum * cryInPanelT; i++) {
      blockTSumRaw += m_blockFctT[i];
      blockTSumSelf += delayBlockFctT[i];
    }
    for (auto i = 0; i < ringNum * cryInPanelT; i++) {
      delayBlockFctT[i] *= blockTSumRaw / blockTSumSelf;
    }

    // update block profile
    m_blockFctA = delayBlockFctA;
    m_blockFctT = delayBlockFctT;

    return true;
  }

public:
  bool cutBin(
      int &binStart, int &binEnd, float *normScan_mich, float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    int availableBinPerView = 0;
    int availableView = 0;
    int totalAvailableBin = 0;
    float maxFwdProj = 0;
    for (int ii = 0; ii < binNum * viewNum * sliceNum; ++ii) {
      if (maxFwdProj < fwd_mich[ii]) {
        maxFwdProj = fwd_mich[ii];
      }
    }
    for (int sl = 0; sl < sliceNum; sl++) {
      for (int vi = 0; vi < viewNum; vi++) {
        int availableBinCount = 0;
        for (int bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(bi) + size_t(vi * binNum) + size_t(sl * binNum * viewNum);
          fwd_mich[LORIndex] /= maxFwdProj;
          if (fwd_mich[LORIndex] >= m_normProtocal.fActCorrCutLow &&
              fwd_mich[LORIndex] <= m_normProtocal.fActCorrCutHigh) {
            availableBinCount++;
          }
        }
        if (availableBinCount > 0) {
          availableView += 1;
          totalAvailableBin += availableBinCount;
        }
      }
    }
    availableBinPerView = totalAvailableBin / availableView;
    availableBinPerView -= (1 - availableBinPerView % 2); // 若availableBinPerView为偶数则-1
    // bincut
    binStart = (binNum - availableBinPerView) / 2;
    binEnd = binStart + availableBinPerView;
    for (int slvi = 0; slvi < sliceNum * viewNum; slvi++) {
      for (int bi = 0; bi < binNum; bi++) {
        size_t LORIndex = size_t(slvi) * size_t(binNum) + size_t(bi);
        if (bi < binStart || bi >= binEnd) {
          normScan_mich[LORIndex] = 0;
        }
      }
    }
    return true;
  }

  bool ActivityCorr(
      float *normScan_mich,
      const float *fwd_mich) // 输入筛选过 binAvailable的triangle
  {
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);

    for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
      fwd_mich[LORIndex] > 0 ? normScan_mich[LORIndex] /= fwd_mich[LORIndex] : normScan_mich[LORIndex] = 0;
    }

    return true;
  }

  bool calCryCount(
      const float *normScan_mich) {
    auto cryNum = polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    std::vector<uint64_t> cryCount(cryNum, 0);
    float cryEffmax = 0;
    m_cryCount.resize(cryNum, 0);
    for (size_t LORIndex = 0; LORIndex < LORNum; LORIndex++) {
      const auto cryPairs =
          example::polygon::calCrystalIDFromLORID(m_normProtocal.polygon, m_normProtocal.detectorGeo, LORIndex);
      cryCount[cryPairs.x]++;
      cryCount[cryPairs.y]++;
      m_cryCount[cryPairs.x] += normScan_mich[LORIndex];
      m_cryCount[cryPairs.y] += normScan_mich[LORIndex];
    }

    for (int i = 0; i < cryNum; i++) {
      cryCount[i] > 0 ? m_cryCount[i] = m_cryCount[i] / cryCount[i] : m_cryCount[i] = 0;
      cryEffmax = (m_cryCount[i] > 0 && m_cryCount[i] > cryEffmax) ? m_cryCount[i] : cryEffmax;
    }

    for (auto cryIndex = 0; cryIndex < cryNum; cryIndex++) {
      m_cryCount[cryIndex] < m_normProtocal.BadChannelThreshold ? m_cryCount[cryIndex] = 0
                                                                : m_cryCount[cryIndex] /= cryEffmax;
    }

    return true;
  }

  bool calBlockFct() {
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto crystalRingNum = example::polygon::getRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInPanelT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的
    auto blockTNum = crystalRingNum * cryInPanelT;
    std::vector<float> positiveT(blockTNum, 0); // 经轴方向block profile
    float maxA = 0;                             // 最大值归一化到1
    float maxT = 0;                             // 最大值归一化到1
    m_blockFctA.resize(crystalRingNum, 0);
    m_blockFctT.resize(blockTNum, 0);
    // blockA
    std::cout << " cal blockA..." << std::endl;
    for (auto ring = 0; ring < crystalRingNum; ring++) {
      size_t positiveCry = 0;
      for (auto cry = 0; cry < crystalPerRing; cry++) {
        if (m_cryCount[ring * crystalPerRing + cry] > 0) {
          m_blockFctA[ring] += m_cryCount[ring * crystalPerRing + cry];
          positiveCry++;
        }
      }
      if (positiveCry > 0) {
        m_blockFctA[ring] /= positiveCry;
      }
      if (m_blockFctA[ring] > maxA) {
        maxA = m_blockFctA[ring];
      }
    }
    for (auto ring = 0; ring < crystalRingNum; ++ring) {
      m_blockFctA[ring] /= maxA;
    }
    std::cout << " cal blockT..." << std::endl;

    // blockT
    for (auto ring = 0; ring < crystalRingNum; ++ring) {
      for (auto cry = 0; cry < crystalPerRing; ++cry) {
        if (m_cryCount[ring * crystalPerRing + cry] > 0) {
          int bv = cry % cryInPanelT;
          m_blockFctT[ring * cryInPanelT + bv] += m_cryCount[ring * crystalPerRing + cry] / m_blockFctA[ring];
          positiveT[ring * cryInPanelT + bv]++; // 计算非坏道数
        }
      }
    }
    for (auto ringbv = 0; ringbv < crystalRingNum * cryInPanelT; ++ringbv) {
      if (positiveT[ringbv] > 0) {
        m_blockFctT[ringbv] /= positiveT[ringbv]; // 排除坏道影响
      }
      if (m_blockFctT[ringbv] > maxT) {
        maxT = m_blockFctT[ringbv]; // 最大值归一化到1
      }
    }
    for (auto ringbv = 0; ringbv < crystalRingNum * cryInPanelT; ++ringbv) {
      m_blockFctT[ringbv] /= maxT;
    }

    return true;
  }

  bool calPlaneFct(
      const float *normScan_mich, const float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNumPerSlice = binNum * viewNum;

    float sliceSensAll = 0; // sens means sensitivity
    int positiveSlice = 0;
    std::vector<uint64_t> planeCount(sliceNum, 0);

    m_planeFct.resize(sliceNum, 0);

    for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
      const auto cryPairs =
          example::polygon::calCrystalIDFromLORID(m_normProtocal.polygon, m_normProtocal.detectorGeo, LORIndex);
      if (fwd_mich[LORIndex] > 0 && m_cryCount[cryPairs.x] > 0 && m_cryCount[cryPairs.y] > 0) {
        m_planeFct[LORIndex / LORNumPerSlice] += normScan_mich[LORIndex];
        planeCount[LORIndex / LORNumPerSlice]++;
      }
    }

    for (auto sl = 0; sl < sliceNum; sl++) {
      if (planeCount[sl] > 0) {
        m_planeFct[sl] /= planeCount[sl];
        sliceSensAll += m_planeFct[sl];
        positiveSlice++;
      } else {
        m_planeFct[sl] = 0;
      }
    }

    float meanSliceSens = sliceSensAll / positiveSlice;

    for (auto sl = 0; sl < sliceNum; sl++) {
      m_planeFct[sl] /= meanSliceSens;
    }

    return true;
  }

  bool calRadialFct(
      const float *normScan_mich, const float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);

    float binSensAll = 0;
    int positiveBin = 0;

    std::vector<float> radialCount(binNum, 0);
    m_radialFct.resize(binNum, 0);

    for (auto sl = 0; sl < sliceNum; sl++) {
      for (auto vi = 0; vi < viewNum; vi++) {
        for (auto bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl * binNum * viewNum) + size_t(bi + vi * binNum);
          const auto cryPairs =
              example::polygon::calCrystalIDFromLORID(m_normProtocal.polygon, m_normProtocal.detectorGeo, LORIndex);
          if (fwd_mich[LORIndex] > 0 && m_cryCount[cryPairs.x] > 0 && m_cryCount[cryPairs.y] > 0) {
            m_radialFct[bi] += normScan_mich[LORIndex];
            radialCount[bi]++;
          }
        }
      }
    }

    for (int bi = 0; bi < binNum; bi++) {
      if (radialCount[bi] > 0) {
        m_radialFct[bi] /= radialCount[bi];
        binSensAll += m_radialFct[bi];
        positiveBin++;
      }
    }
    float binSensMean = binSensAll / positiveBin;
    for (auto binIndex = 0; binIndex < binNum; binIndex++) {
      m_radialFct[binIndex] /= binSensMean;
    }

    return true;
  }

  bool calInterferenceFct(
      const float *normScan_mich, const float *fwd_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryInBlockT = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的

    float InterSensAll = 0;
    int positiveInter = 0;

    std::vector<float> interferenceCount(binNum * cryInBlockT, 0);
    m_interferenceFct.resize(binNum * cryInBlockT, 0);

    for (auto sl = 0; sl < sliceNum; sl++) {
      for (auto vi = 0; vi < viewNum; vi++) {
        int bv = vi % cryInBlockT;
        for (auto bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * (binNum * viewNum) + size_t(vi * binNum + bi);
          const auto cryPairs =
              example::polygon::calCrystalIDFromLORID(m_normProtocal.polygon, m_normProtocal.detectorGeo, LORIndex);
          if (fwd_mich[LORIndex] > 0 && m_cryCount[cryPairs.x] > 0 && m_cryCount[cryPairs.y] > 0) {
            m_interferenceFct[bv * binNum + bi] += normScan_mich[LORIndex];
            interferenceCount[bv * binNum + bi]++;
          }
        }
      }
    }

    for (auto InterIndex = 0; InterIndex < binNum * cryInBlockT; InterIndex++) {
      if (interferenceCount[InterIndex] > 0) {
        m_interferenceFct[InterIndex] /= interferenceCount[InterIndex];
        InterSensAll += m_interferenceFct[InterIndex];
        positiveInter++;
      }
    }

    auto meanInter = InterSensAll / positiveInter;
    for (auto InterIndex = 0; InterIndex < binNum * cryInBlockT; InterIndex++) {
      m_interferenceFct[InterIndex] /= meanInter;
    }

    return true;
  }

  bool binFctExtension(
      float *binFct, const int &binNum, const int &binStart, const int &binEnd, const int &binsToTrian) {
    std::unique_ptr<float[]> tmp_xR = std::make_unique<float[]>(binsToTrian);
    std::unique_ptr<float[]> tmp_xL = std::make_unique<float[]>(binsToTrian);
    std::unique_ptr<float[]> tmp_yR = std::make_unique<float[]>(binsToTrian);
    std::unique_ptr<float[]> tmp_yL = std::make_unique<float[]>(binsToTrian);
    int lEdge = binStart;
    int rEdge = binEnd - 1;
    while (binFct[lEdge] <= 0.8 && lEdge < binEnd) // 0.8 E180经验参数
    {
      lEdge++;
    }
    while (binFct[rEdge] <= 0.8 && rEdge >= binStart) {
      rEdge--;
    }
    for (int i = 0; i < binsToTrian; ++i) {
      tmp_xL[i] = lEdge + i;
      tmp_yL[i] = binFct[lEdge + i];
      tmp_xR[i] = rEdge - i;
      tmp_yR[i] = binFct[rEdge - i]; // binEnd 不包含在有效的bin内
    }
    float ar, br, al, bl; // a：斜率；b：截距；r：右侧；l：左侧。
    openpni::basic::LineFitLeastSquares<float>(tmp_xL.get(), tmp_yL.get(), al, bl, binsToTrian);
    openpni::basic::LineFitLeastSquares<float>(tmp_xR.get(), tmp_yR.get(), ar, br, binsToTrian);
    for (int x = 0; x < lEdge + binsToTrian; ++x) {
      binFct[x] = al * x + bl;
      if (binFct[x] < 0.6)
        binFct[x] = 0.6;
    }
    for (int x = rEdge + 1 - binsToTrian; x < binNum; ++x) {
      binFct[x] = ar * x + br;
      if (binFct[x] < 0.6)
        binFct[x] = 0.6;
    }
    return true;
  }

  bool calExactCryFct(
      const float *normScan_mich) {
    auto binNum = example::polygon::getBinNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNum = polygon::getTotalCrystalNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto blockPerRing = example::polygon::getBlockNumOneRing(m_normProtocal.polygon, m_normProtocal.detectorGeo);
    auto cryNumYInBlock = example::polygon::getCrystalNumYInPanel(
        m_normProtocal.polygon,
        m_normProtocal.detectorGeo); // block在Y方向上包含的晶体数；以整个PDM内的晶体数量来计算是有必要的
    auto crystalRingNum = example::polygon::getRingNum(m_normProtocal.polygon, m_normProtocal.detectorGeo);

    std::vector<float> blockLOR(blockPerRing * blockPerRing, 0);
    std::vector<float> fanLOR(cryNum * blockPerRing, 0);
    std::vector<float> meanDecEff(blockPerRing, 0);
    float LOREffmax = 0;
    float LOREffmin = 0;
    float LOREffmean = 0;
    m_cryFct.resize(cryNum, 0);

    cryFctByfanSum(m_cryFct, normScan_mich, m_normProtocal.radialModuleNumS, m_normProtocal.polygon,
                   m_normProtocal.detectorGeo, m_normProtocal.BadChannelThreshold);

    for (auto cryIndex = 0; cryIndex < cryNum; cryIndex++) {
      LOREffmean += m_cryFct[cryIndex];
      LOREffmax = std::max(LOREffmax, m_cryFct[cryIndex]);
      LOREffmin = std::min(LOREffmin, m_cryFct[cryIndex]);
    }

    LOREffmean /= cryNum;
    for (auto cryIndex = 0; cryIndex < cryNum; cryIndex++) {
      m_cryFct[cryIndex] /= LOREffmax;
      if (m_cryFct[cryIndex] <= m_normProtocal.BadChannelThreshold)
        m_cryFct[cryIndex] = 0; // bad channel threshold
    }

    return true;
  }

public:
  std::vector<float> m_cryCount;
  std::vector<float> m_blockFctA; // axial block profile,size recommended to be crystalRingNum
  std::vector<float> m_blockFctT; // transaxial block profile,size recommended to be
                                  // crystalRingNum * cryInPanelT
  std::vector<float> m_planeFct;
  std::vector<float> m_radialFct;
  std::vector<float> m_interferenceFct;
  std::vector<float> m_cryFct; // fansum
private:
  NormCorrection::normCorrProtocal m_normProtocal;
};

// deadTime
class DeadTimeCorrection::DeadTimeCorrection_impl {
public:
  explicit DeadTimeCorrection_impl(const DeadTimeCorrection::deadTimeProtocal &_dtProtocal)
      : m_dtProtocal(_dtProtocal) {};

  bool modelBasedDTComponent(
      double *dt_mich,                      // DT是由table计算出来的所以类型相同
      const float *delay_mich,              // randMich
      const double *calibrationtable) const // size:2 * m_acquisitionNum * m_dsSlice
  {
    auto binNum = example::polygon::getBinNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto blockRingNum = example::polygon::getBlockRingNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto dsSlice = blockRingNum * blockRingNum;
    std::unique_ptr<double[]> DTComponent = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> DSdelay_mich = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> Rr = std::make_unique<double[]>(m_dtProtocal.acquisitionNum * dsSlice);
    std::unique_ptr<double[]> CFDT = std::make_unique<double[]>(m_dtProtocal.acquisitionNum * dsSlice);
    std::unique_ptr<double[]> Rrandom = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> RcaliMax = std::make_unique<double[]>(dsSlice);
    std::unique_ptr<double[]> RcaliMin = std::make_unique<double[]>(dsSlice);
    std::vector<double> tempRr(m_dtProtocal.acquisitionNum, 0);

    double x1 = 0;
    double x2 = 0;
    double y1 = 0;
    double y2 = 0;
    // read table
    for (auto sl = 0; sl < dsSlice; sl++) {
      for (auto acq = 0; acq < m_dtProtocal.acquisitionNum; acq++) {
        int index1 = sl * m_dtProtocal.acquisitionNum + acq;
        int index2 = index1 + m_dtProtocal.acquisitionNum * dsSlice;

        Rr[index1] = calibrationtable[index1];
        CFDT[index1] = calibrationtable[index2];
      }
    }
    // michDownSampling
    if (michDownSampling(DSdelay_mich.get(), delay_mich) != true)
      return false;

    // Rrandom
    for (auto dssl = 0; dssl < dsSlice; dssl++) {
      Rrandom[dssl] = DSdelay_mich[dssl] / m_dtProtocal.scanTime;
      RcaliMax[dssl] = Rr[dssl * m_dtProtocal.acquisitionNum];
      RcaliMin[dssl] = Rr[(dssl + 1) * m_dtProtocal.acquisitionNum - 1];
    }

    // 插值
    for (int sl = 0; sl < dsSlice; sl++) // 开始计算插值
    {
      if (Rrandom[sl] < m_dtProtocal.randomRateMin) // 若小于设定值randomRatesMin，则设为1，表示不进行死时间校正
      {
        DTComponent[sl] = 1;
      }
      // 一般校正表随机计数率最小值会小于randomRatesMin，这分支通常进不来，
      // 但考虑实验设置等原因，可能没有采集到很低的活度，此时在randomRatesMin和RcaliMin之间做外推
      else if (Rrandom[sl] <= RcaliMin[sl]) {
        x1 = Rr[(sl + 1) * m_dtProtocal.acquisitionNum - 2];
        x2 = Rr[(sl + 1) * m_dtProtocal.acquisitionNum - 1];
        y1 = CFDT[(sl + 1) * m_dtProtocal.acquisitionNum - 2];
        y2 = CFDT[(sl + 1) * m_dtProtocal.acquisitionNum - 1];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      } else if (Rrandom[sl] < RcaliMax[sl]) // 在校正表中随机计数率最大和最小的范围内
      {
        for (int acq = 0; acq < m_dtProtocal.acquisitionNum; acq++) {
          tempRr[acq] = Rr[(sl + 1) * m_dtProtocal.acquisitionNum - acq - 1];
        }

        auto it = lower_bound(tempRr.begin(), tempRr.end(), Rrandom[sl]);
        int index = distance(tempRr.begin(), it);

        x1 = tempRr[index];
        x2 = tempRr[index - 1];
        y1 = CFDT[(sl + 1) * m_dtProtocal.acquisitionNum - index - 1];
        y2 = CFDT[(sl + 1) * m_dtProtocal.acquisitionNum - index];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      } else if (Rrandom[sl] >= RcaliMax[sl]) // 超出校正表随机计数率的最大值
      {
        x1 = Rr[sl * m_dtProtocal.acquisitionNum];
        x2 = Rr[sl * m_dtProtocal.acquisitionNum + 1];
        y1 = CFDT[sl * m_dtProtocal.acquisitionNum];
        y2 = CFDT[sl * m_dtProtocal.acquisitionNum + 1];
        DTComponent[sl] = y1 + (y2 - y1) * (Rrandom[sl] - x1) / (x2 - x1);
      }

      if (DTComponent[sl] < 1) //?
      {
        DTComponent[sl] = 1;
      }
    }

    for (size_t sl = 0; sl < sliceNum; sl++) {
      int ring1, ring2;
      example::polygon::calRing1Ring2FromSlice(m_dtProtocal.polygon, m_dtProtocal.detectorGeo, sl, ring1, ring2);
      int blockRing1 = ring1 / m_dtProtocal.detectorGeo.crystalNumU;
      int blockRing2 = ring2 / m_dtProtocal.detectorGeo.crystalNumU;
      int blockRingId = blockRing1 * blockRingNum + blockRing2;
      for (size_t vi = 0; vi < viewNum; vi++) {
        for (size_t bi = 0; bi < binNum; bi++) {
          size_t lorId = bi + vi * binNum + sl * binNum * viewNum;
          dt_mich[lorId] /= DTComponent[blockRingId];
        }
      }
    }
    return true;
  }

private:
  bool michDownSampling(
      double *downSample_triangle, const float *delay_mich) const {
    int ring1, ring2;
    int blockRing1, blockRing2;
    int blockRingID;
    auto binNum = example::polygon::getBinNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto viewNum = example::polygon::getViewNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto sliceNum = example::polygon::getSliceNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto LORNum = example::polygon::getLORNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);
    auto blockRingNum = example::polygon::getBlockRingNum(m_dtProtocal.polygon, m_dtProtocal.detectorGeo);

    for (size_t sl = 0; sl < sliceNum; sl++) {
      example::polygon::calRing1Ring2FromSlice(m_dtProtocal.polygon, m_dtProtocal.detectorGeo, sl, ring1, ring2);
      blockRing1 = ring1 / m_dtProtocal.detectorGeo.crystalNumU;
      blockRing2 = ring2 / m_dtProtocal.detectorGeo.crystalNumU;
      blockRingID = blockRing1 * blockRingNum + blockRing2;
      for (size_t vi = 0; vi < viewNum; vi++) {
        for (size_t bi = 0; bi < binNum; bi++) {
          size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
          downSample_triangle[blockRingID] += delay_mich[LORIndex];
        }
      }
    }

    return true;
  }

private:
  DeadTimeCorrection::deadTimeProtocal m_dtProtocal; // default is E180
};

// rand
class RandCorrection::RandCorrection_impl {
public:
  bool smoothMichByNiu(
      float *randmich_triangle, const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo,
      const unsigned minSectorDifference = 4, const unsigned radialModuleNumS = 6) {
    auto cryNum = polygon::getTotalCrystalNum(polygon, detectorGeo);
    std::vector<float> cryFct(cryNum, 0);
    auto crystalPerRing = example::polygon::getCrystalNumOneRing(polygon, detectorGeo);
    auto cryNumYInPanel = example::polygon::getCrystalNumYInPanel(polygon, detectorGeo);
    cryFctByfanSum(cryFct, randmich_triangle, radialModuleNumS, polygon, detectorGeo);

    for (auto LORIndex = 0; LORIndex < example::polygon::getLORNum(polygon, detectorGeo); LORIndex++) {
      const auto cryPairs = example::polygon::calCrystalIDFromLORID(polygon, detectorGeo, LORIndex);
      int panel1 = cryPairs.x % crystalPerRing / cryNumYInPanel;
      int panel2 = cryPairs.y % crystalPerRing / cryNumYInPanel;
      example::polygon::isPairFar(polygon, panel1, panel2, minSectorDifference)
          ? randmich_triangle[LORIndex] = cryFct[cryPairs.x] * cryFct[cryPairs.y]
          : randmich_triangle[LORIndex] = 0;
    }
    return false;
  }
};

// scat
//     class ScatterCorrection::ScatterCorrection_impl
//     {
//     public:
//         explicit ScatterCorrection_impl(const ScatterCorrection::scatterProtocal
//         &_scatterProtocal)
//             : m_scatterProtocal(_scatterProtocal) {};

//     private: // math
//         double
//         calTotalComptonCrossSection(double energy) const
//         {
//             const double Re = 2.818E-13; // cm
//             double k = energy / 511.f;
//             double a = log(1 + 2 * k);
//             double prefactor = 2 * M_PI * Re * Re;
//             return prefactor * ((1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 *
//             k) - a / k) +
//                                 a / (2.0 * k) - (1.0 + 3.0 * k) / (1.0 + 2.0 * k) /
//                                 (1.0 + 2.0 * k));
//         }

//         double calScannerEffwithScatter(double energy) const
//         {
//             // The cumulative distribution function (CDF) of the normal, or Gaussian,
//             // distribution with standard deviation sigma and mean mu is:
//             // F(x) = 0.5 * ( 1 + erf((x-mu)/sqrt(2)*sigma) )
//             // 2.35482=2 * sqrt(2 * ln(2)); sqrt(2) = 1.41421
//             double sigmaTimesSqrt2 = m_scatterProtocal.scatterEnergyWindow.otherInfo *
//             511 / 2.35482 * 1.41421; return 0.5 *
//             (erf((m_scatterProtocal.scatterEnergyWindow.high - energy) /
//             sigmaTimesSqrt2) -
//                           erf((m_scatterProtocal.scatterEnergyWindow.low - energy) /
//                           sigmaTimesSqrt2));
//         }

//         double calTotalComptonCrossSectionRelativeTo511keV(double scatterEnergy) const
//         {
//             const double a = scatterEnergy / 511.0;
//             // Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) /
//             9 staCorrection_impl.cuhtic const double prefactor = 9.0 / (-40 + 27 * log(3.));

//             return // checked this in Mathematica
//                 prefactor *
//                 (((-4 - a * (16 + a * (18 + 2 * a))) / ((1 + 2 * a) * (1 + 2 * a)) +
//                   ((2 + (2 - a) * a) * log(1 + 2 * a)) / a) /
//                  (a * a));
//         }
//         double calTotalAttenInScatterEnergy(double totalAtten511, double scatterEnergy)
//         const
//         {
//             return pow(totalAtten511,
//             calTotalComptonCrossSectionRelativeTo511keV(scatterEnergy));
//         }

//         double calScatCosTheta(int cryIndex1, int scatIndex, int cryIndex2, const
//         std::vector<basic::Point3D<float>> crystalPosition) const
//         {
//             auto _as = crystalPosition[cryIndex1] - m_scatterPoint[scatIndex].loc; //
//             pointA to sctpoint double d_as = _as.l2() * 0.1; // cm auto _sb =
//             crystalPosition[cryIndex2] - m_scatterPoint[scatIndex].loc; // sctpoint to
//             pointB double d_sb = _sb.l2() * 0.1; // cm auto _ab =
//             crystalPosition[cryIndex1] - crystalPosition[cryIndex2];    // pointA to
//             pointB double d_ab = _ab.l2() * 0.1; // cm return -(d_as * d_as + d_sb *
//             d_sb - d_ab * d_ab) / (2 * d_as * d_sb);
//         }

//         double calScatCosTheta_new(basic::Point3D<float> cryPosA, basic::Point3D<float>
//         cryPosB, basic::Point3D<float> sctPos) const
//         {
//             auto _as = cryPosA - sctPos;  // pointA to sctpoint
//             double d_as = _as.l2() * 0.1; // cm
//             auto _sb = cryPosB - sctPos;  // sctpoint to pointB
//             double d_sb = _sb.l2() * 0.1; // cm
//             auto _ab = cryPosA - cryPosB; // pointA to pointB
//             double d_ab = _ab.l2() * 0.1; // cm
//             return -(d_as * d_as + d_sb * d_sb - d_ab * d_ab) / (2 * d_as * d_sb);
//         }

//         double calDiffCrossSection(double scatCosTheta) const
//         {
//             // Kelin-Nishina formula. re is classical electron radius
//             const double Re = 2.818E-13;               // cm
//             double waveRatio = 1 / (2 - scatCosTheta); //  lamda/lamda'
//             return 0.5 * Re * Re * waveRatio * waveRatio *
//                    (waveRatio + 1 / waveRatio + scatCosTheta * scatCosTheta - 1);
//         }

//         bool averageFilterApply(
//             std::vector<double> &array,
//             int filterLength = 0) const
//         {
//             int arrayLength = static_cast<int>(array.size()) / 2;
//             std::vector<double> array_temp(array);
//             for (int index = 0; index < arrayLength; index++)
//             {
//                 int lowerLimit = std::max(0, index - filterLength);
//                 int upperLimit = std::min(arrayLength - 1, index + filterLength);
//                 double sum_a = 0;
//                 double sum_b = 0;
//                 for (int i = lowerLimit; i <= upperLimit; i++)
//                 {
//                     sum_a += array_temp[2 * i];
//                     sum_b += array_temp[2 * i + 1];
//                 }
//                 array[2 * index] = sum_a / (upperLimit - lowerLimit + 1);
//                 array[2 * index + 1] = sum_b / (upperLimit - lowerLimit + 1);
//             }
//             return true;
//         }
//         //==========================================================initialtor
bool initScatterPoint(
    float *attnMap, const openpni::basic::Image3DGeometry &attnMapSize) {
  openpni::basic::Point3D<float> gridNum;
  gridNum = attnMapSize.voxelNum * attnMapSize.voxelSize / m_scatterProtocal.scatterGrid;
  int totalNum = (int)ceil(gridNum.x) * (int)ceil(gridNum.y) * (int)ceil(gridNum.z);
  m_scatterPoint.reserve(totalNum);
  std::cout << m_scatterPoint.size() << " scatter points reserved." << std::endl;
  if (m_scatterProtocal.randomSelect) {
    std::cout << "Notice: We will select scatter point randomly. " << std::endl;
    srand(0);
  }
  // current grid minimum border
  openpni::basic::Point3D<float> Lim = {0.0, 0.0, 0.0};
  openpni::basic::Point3D<float> upperLim;
  openpni::basic::Point3D<float> pointRand;
  openpni::basic::Point3D<int> voxelIndex;
  scatterPoint3D scatterPointTemp;
  while (Lim.x < attnMapSize.voxelSize.x * attnMapSize.voxelNum.x) {
    upperLim.x =
        std::min(Lim.x + m_scatterProtocal.scatterGrid.x, (float)(attnMapSize.voxelSize.x * attnMapSize.voxelNum.x));
    Lim.y = 0;
    while (Lim.y < attnMapSize.voxelSize.y * attnMapSize.voxelNum.y) {
      upperLim.y =
          std::min(Lim.y + m_scatterProtocal.scatterGrid.y, (float)(attnMapSize.voxelSize.y * attnMapSize.voxelNum.y));
      Lim.z = 0;
      while (Lim.z < attnMapSize.voxelSize.z * attnMapSize.voxelNum.z) {
        upperLim.z = std::min(Lim.z + m_scatterProtocal.scatterGrid.z,
                              (float)(attnMapSize.voxelSize.z * attnMapSize.voxelNum.z));
        pointRand = (upperLim - Lim) * static_cast<float>(m_scatterProtocal.randomSelect) * rand() / (float)RAND_MAX;
        pointRand = pointRand + Lim;
        voxelIndex = basic::make_point3d<int>(pointRand / attnMapSize.voxelSize);
        if (attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
                    voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] >= m_scatterProtocal.scatterPointThreshold) {
          scatterPointTemp.loc = pointRand - attnMapSize.voxelSize * attnMapSize.voxelNum / 2 + attnMapSize.centre();
          // cm-1
          scatterPointTemp.mu = attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
                                        voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] *
                                10;
          m_scatterPoint.push_back(scatterPointTemp);
        }
        Lim.z += m_scatterProtocal.scatterGrid.z;
      }
      Lim.y += m_scatterProtocal.scatterGrid.y;
    }
    Lim.x += m_scatterProtocal.scatterGrid.x;
  }
  m_scatterPoint.shrink_to_fit();
  if (m_scatterPoint.size() <= 0) {
    std::cout << "no good scatter point. At initScatterPoint." << std::endl;
    return false;
  }
  return true;
}
//         bool initScannerEffTable()
//         {
//             int energyBinNum = int((m_scatterProtocal.scannerEffTableEnergy.high -
//             m_scatterProtocal.scannerEffTableEnergy.low) /
//                                    m_scatterProtocal.scannerEffTableEnergy.otherInfo) +
//                                1;
//             m_scannerEff.resize(energyBinNum, 0);
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//             for (int i = 0; i < energyBinNum; i++)
//                 m_scannerEff[i] = static_cast<float>(calScannerEffwithScatter(
//                     m_scatterProtocal.scannerEffTableEnergy.low + i *
//                     m_scatterProtocal.scannerEffTableEnergy.otherInfo));
//             return true;
//         }

//         bool initAttnCutBedCoff(
//             float *attnImg,
//             const openpni::basic::Image3DGeometry &attnMap3dSize,
//             const std::vector<openpni::basic::Point3D<float>> &crystalPosition,
//             const example::RingPETScanner &scanner)
//         {
//             m_attnCutBedCoff.resize(scanner.LORNum(), 0);
//             // cutBed
//             for (int i = 0; i < attnMap3dSize.totalVoxelNum(); i++)
//             {
//                 if (attnImg[i] < 0.0096 / 3.0)
//                 {
//                     attnImg[i] = 0.0;
//                 }
//             }
//             // fwd
//             std::vector<float> attn_fwdTriangle(std::size_t(scanner.totalCrystalNum())
//             * (std::size_t(scanner.totalCrystalNum()) - 1) / 2, 0);

//             const auto cryPos =
//             openpni::example::getCrystalPositionAllHost<float>(scanner); const auto
//             rearranger = openpni::example::Rearranger(scanner); const auto subsets =
//             openpni::example::getSubset(scanner, 1); auto Datasets =
//             openpni::basic::get_DataSetsViewMich(scanner.totalCrystalNum(),
//                                                                  subsets,
//                                                                  attn_fwdTriangle.data(),
//                                                                  cryPos.data());

//             for (const auto &dataset : Datasets)
//             {
//                 openpni::process::forwardProjection<openpni::process::ProjectionMethodSiddon>(
//                     attnImg,
//                     attnMap3dSize,
//                     attn_fwdTriangle.data(),
//                     dataset);
//             }

//             openpni::process::rearrageTriagle2Mich(
//                 attn_fwdTriangle.data(),
//                 m_attnCutBedCoff.data(),
//                 rearranger);

//             for (auto LORIndex = 0; LORIndex < scanner.LORNum(); LORIndex++)
//             {
//                 m_attnCutBedCoff[LORIndex] = exp(-m_attnCutBedCoff[LORIndex]);
//             }
//             // lgxtest save m_attnCutBedCoff to file
//             std::ofstream
//             attnCutBedCoffFile("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/lgxtest/sss_attnCutBedCoff0624.img3d",
//             std::ios::binary); if (attnCutBedCoffFile.is_open())
//             {
//                 attnCutBedCoffFile.write(reinterpret_cast<char
//                 *>(m_attnCutBedCoff.data()), sizeof(float) * m_attnCutBedCoff.size());
//                 attnCutBedCoffFile.close();
//             }
//             return true;
//         }

//         template <std::floating_point attnValueType>
//         bool initTotalAttn(
//             const std::vector<basic::Point3D<float>> &crystalPosition,
//             const attnValueType *attnImg,
//             const basic::Image3DGeometry &image3dSize)
//         {
//             m_totalAttn.resize(crystalPosition.size() * m_scatterPoint.size(), 0);
//             std::cout << "initTotalAttn: crystalPosition.size() = " <<
//             crystalPosition.size() << ", m_scatterPoint.size() = " <<
//             m_scatterPoint.size() << std::endl;
//             // 对于每个model_crytal，计算他与scatterPoint的Siddon raytracying
//             for (int posIndex = 0; posIndex < crystalPosition.size(); posIndex++)
//             {
//                 for (int scatIndex = 0; scatIndex < m_scatterPoint.size(); scatIndex++)
//                 {
//                     int indexNow = posIndex * m_scatterPoint.size() + scatIndex;
//                     auto rayTracingValue =
//                     ProjectionMethodSiddon::forward(&crystalPosition[posIndex],
//                                                                            &m_scatterPoint[scatIndex].loc,
//                                                                            attnImg,
//                                                                            &image3dSize);
//                     m_totalAttn[indexNow] = exp(-rayTracingValue);
//                 }
//             }

//             return true;
//         }

//         bool initTotalEmission(
//             const std::vector<openpni::basic::Point3D<float>> &crystalPosition,
//             const float *reconImg,
//             const basic::Image3DGeometry &image3dSize)
//         {
//             m_totalEmission = std::vector<float>(crystalPosition.size() *
//             m_scatterPoint.size(), 0); for (int modelIndex = 0; modelIndex <
//             crystalPosition.size(); modelIndex++)
//             {
//                 for (int scatIndex = 0; scatIndex < m_scatterPoint.size(); scatIndex++)
//                 {
//                     int indexNow = modelIndex * m_scatterPoint.size() + scatIndex;
//                     auto rayTracingValue = process::ProjectionMethodSiddon::forward(
//                         &crystalPosition[modelIndex],
//                         &m_scatterPoint[scatIndex].loc,
//                         reconImg,
//                         &image3dSize);
//                     m_totalEmission[indexNow] = rayTracingValue;
//                 }
//             }
//             return true;
//         }

//         bool InitProjectArea(
//             const openpni::example::RingPETScanner &scanner,
//             const std::vector<openpni::basic::Point3D<float>> &crystalPosition)
//         {
//             int cryNumOneRIng = scanner.crystalNumOneRing();
//             double crystalArea = scanner.crystalSize.y * scanner.crystalSize.z * 0.01;
//             // cm2 int crystalNum = crystalPosition.size(); int scatterPointNum =
//             m_scatterPoint.size(); m_projectArea.resize(crystalNum * scatterPointNum,
//             0);
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//             for (int cryIndex = 0; cryIndex < crystalNum; cryIndex++)
//             {
//                 int crystalInRing = cryIndex % cryNumOneRIng;
//                 int panel = crystalInRing / scanner.crystalNumYInPanel();
//                 double sita = double(panel) * 2 * M_PI / scanner.panelNum;
//                 auto crystalPosNow = crystalPosition[cryIndex];
//                 for (int scatIndex = 0; scatIndex < scatterPointNum; scatIndex++)
//                 {
//                     auto c_s = crystalPosition[cryIndex] -
//                     m_scatterPoint[scatIndex].loc; double c2sDistance = c_s.l2() * 0.1;
//                     // cm auto scatterPosNow = m_scatterPoint[scatIndex].loc; double
//                     cosTheta = (crystalPosNow.x - scatterPosNow.x) *
//                                           cos(sita) +
//                                       (crystalPosNow.y - scatterPosNow.y) *
//                                           sin(sita) /
//                                           c2sDistance * 0.1; // cm
//                     m_projectArea[cryIndex * scatterPointNum + scatIndex] =
//                     static_cast<float>(crystalArea * cosTheta);
//                 }
//             }
//             return true;
//         }

//         //====================================================================Function
bool getScaleBySlice(
    const float *scatNoTailFitting, int ring1, int ring2, double &a, double &b, const float *prompt,
    const float *norm, // default : 0
    const float *rand, // default : 1
    const openpni::example::RingPETScanner &scanner) {
  auto binNum = scanner.binNum();
  auto LORNumOneSlice = scanner.viewNum() * scanner.binNum();
  auto binNumOutFOVOneSide = openpni::example::getBinNumOutFOVOneSide(scanner, m_scatterProtocal.minSectorDifference);
  size_t sliceIndexNow = ring1 * scanner.ringNum() + ring2;
  size_t indexStart = sliceIndexNow * LORNumOneSlice;

  int num = 0;      // n
  double sumXY = 0; // sum(x*y)
  double sumX = 0;  // sum(x)
  double sumY = 0;  // sum(y)
  double sumXX = 0; // sum(x*x)

  for (int LORIndex = 0; LORIndex < LORNumOneSlice; LORIndex++) {
    int binIndex = LORIndex % binNum;
    if (m_attnCutBedCoff[LORIndex + indexStart] >= m_scatterProtocal.scatterTailFittingThreshold &&
        binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
      if (norm[LORIndex + indexStart] == 0)
        continue;
      num++;
      sumX += scatNoTailFitting[LORIndex + indexStart];
      sumY += prompt[LORIndex + indexStart] - rand[LORIndex + indexStart];
      sumXY += scatNoTailFitting[LORIndex + indexStart] * (prompt[LORIndex + indexStart] - rand[LORIndex + indexStart]);
      sumXX += scatNoTailFitting[LORIndex + indexStart] * scatNoTailFitting[LORIndex + indexStart];
    }
  }

  if (m_scatterProtocal.with_bias) {
    a = (num * sumXY - sumX * sumY) / (num * sumXX - sumX * sumX);
    b = (sumXX * sumY - sumX * sumXY) / (num * sumXX - sumX * sumX);
  } else {
    a = fabs(sumXY / sumXX);
    b = 0;
  }
  return true;
}

bool doScaleBySlice(
    float *scatter, const float *scat_noTailFitting, int ring1, int ring2, double a, double b, const float *norm,
    const openpni::example::RingPETScanner &scanner) {
  auto binNum = scanner.binNum();
  auto LORNumOneSlice = scanner.viewNum() * scanner.binNum();
  size_t sliceIndexNow = ring1 * scanner.ringNum() + ring2;
  size_t indexStart = sliceIndexNow * LORNumOneSlice;
  size_t binNumOutFOVOneSide = openpni::example::getBinNumOutFOVOneSide(scanner, m_scatterProtocal.minSectorDifference);

  for (size_t LORIndex = 0; LORIndex < LORNumOneSlice; LORIndex++) {
    int binIndex = LORIndex % binNum;
    if (binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
      // exclude the bad channel
      if (norm[LORIndex + indexStart] == 0) {
        scatter[LORIndex + indexStart] = 0;
        continue;
      }
      scatter[LORIndex + indexStart] = static_cast<float>(a * scat_noTailFitting[LORIndex + indexStart] + b);
    }

    else
      scatter[LORIndex + indexStart] = 0;
  }
  return true;
}
//         bool singleScatterSimulation(
//             float *scatterTriangle_noTailFitting,
//             const std::vector<openpni::basic::Point3D<float>> &crystalPosition,
//             const openpni::example::RingPETScanner &scanner,
//             const float *reconImg,
//             const basic::Image3DGeometry &image3dSize) // 最好不要用cpu版本
//         {
//             auto LORNum = scanner.LORNum();
//             int binNumOutFOVOneSide = openpni::example::getBinNumOutFOVOneSide(scanner,
//             m_scatterProtocal.minSectorDifference); int binNunm = scanner.binNum();
//             openpni::basic::Point3D<float> scatterGridNum;
//             scatterGridNum = image3dSize.voxelSize * image3dSize.voxelNum /
//             m_scatterProtocal.scatterGrid; int totalNum = (int)ceil(scatterGridNum.x) *
//             (int)ceil(scatterGridNum.y) * (int)ceil(scatterGridNum.z); const double
//             scatter_volume = image3dSize.voxelSize.x * image3dSize.voxelNum.x *
//                                           image3dSize.voxelSize.y *
//                                           image3dSize.voxelNum.y *
//                                           image3dSize.voxelSize.z *
//                                           image3dSize.voxelNum.z / totalNum * 1e3;
//             const double totalComptonCrossSection511keV =
//             calTotalComptonCrossSection(511.f); const double ScannerEff511keV =
//             calScannerEffwithScatter(511.f); double common_factor = 0.25 / M_PI *
//             scatter_volume *
//                                    ScannerEff511keV / totalComptonCrossSection511keV;

//             int scatterPointNum = static_cast<int>(m_scatterPoint.size());
//             int binNum = scanner.binNum();
//             int sliceNum = scanner.sliceNum();
//             int LORNumOneSlice = scanner.viewNum() * scanner.binNum();
//             std::cout << "here" << std::endl;
//             for (int sliceIndex = 0; sliceIndex < sliceNum; sliceIndex++)
//             {
//                 std::cout << "sliceIndex = " << sliceIndex << std::endl;
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//                 for (int indexOneSlice = 0; indexOneSlice < LORNumOneSlice;
//                 indexOneSlice++)
//                 {
//                     int LORIndex = sliceIndex * LORNumOneSlice + indexOneSlice;
//                     int binIndex = LORIndex % binNum;
//                     if (binIndex < binNumOutFOVOneSide || binIndex >= binNum -
//                     binNumOutFOVOneSide)
//                         continue;
//                     const auto [cryID1, cryID2] =
//                     openpni::example::getCrystalIDFromLORID(scanner, LORIndex); for
//                     (int scatPointIndex = 0; scatPointIndex < scatterPointNum;
//                     scatPointIndex++)
//                     {
//                         // the cosine of scatter angle
//                         int index1 = cryID1 * scatterPointNum + scatPointIndex;
//                         int index2 = cryID2 * scatterPointNum + scatPointIndex;
//                         double scatCosTheta = calScatCosTheta(cryID1, scatPointIndex,
//                         cryID2, crystalPosition); double scatterEnergy = 511 / (2 -
//                         scatCosTheta); // the energy of scatteted photon if
//                         (scatterEnergy < m_scatterProtocal.scannerEffTableEnergy.low ||
//                             scatterEnergy >
//                             m_scatterProtocal.scannerEffTableEnergy.high) continue;
//                         double scannerEffNow =
//                             m_scannerEff[int((scatterEnergy -
//                             m_scatterProtocal.scannerEffTableEnergy.low) /
//                                              m_scatterProtocal.scannerEffTableEnergy.otherInfo)];

//                         double Ia = m_totalEmission[index1] * m_totalAttn[index1] *
//                                     calTotalAttenInScatterEnergy(m_totalAttn[index2],
//                                     scatterEnergy);
//                         double Ib = m_totalEmission[index2] * m_totalAttn[index2] *
//                                     calTotalAttenInScatterEnergy(m_totalAttn[index1],
//                                     scatterEnergy);

//                         auto p_rAS = crystalPosition[cryID1] -
//                         m_scatterPoint[scatPointIndex].loc; double rAS = p_rAS.l2() *
//                         0.1; // cm auto p_rBS = crystalPosition[cryID2] -
//                         m_scatterPoint[scatPointIndex].loc; double rBS = p_rBS.l2() *
//                         0.1; // cm

//                         scatterTriangle_noTailFitting[LORIndex] +=
//                         static_cast<float>(m_scatterPoint[scatPointIndex].mu *
//                                                                                       m_projectArea[index1] *
//                                                                                       m_projectArea[index2] *
//                                                                                       calDiffCrossSection(scatCosTheta)
//                                                                                       * (Ia + Ib) * scannerEffNow /
//                                                                                       (rAS * rAS * rBS * rBS));
//                     }
//                     scatterTriangle_noTailFitting[LORIndex] *= common_factor;
//                 }
//             }
//             return true;
//         }

//         bool singleScatterSimulation_CUDA(
//             float *scatterMich_noTailFitting,
//             const std::vector<openpni::basic::Point3D<float>> &crystalPosition,
//             const openpni::example::RingPETScanner &scanner,
//             const float *reconImg,
//             const basic::Image3DGeometry &image3dSize)
//         {
//             auto LORNum = scanner.LORNum();
//             int binNumOutFOVOneSide = openpni::example::getBinNumOutFOVOneSide(scanner,
//             m_scatterProtocal.minSectorDifference); int binNum = scanner.binNum(); int
//             sliceNum = scanner.sliceNum(); auto crystalNumOneRing =
//             scanner.crystalNumOneRing(); int ringNum = scanner.ringNum(); int
//             LORNumOneSlice = scanner.viewNum() * scanner.binNum();
//             openpni::basic::Point3D<float> scatterGridNum;
//             scatterGridNum = image3dSize.voxelSize * image3dSize.voxelNum /
//             m_scatterProtocal.scatterGrid; int totalNum = (int)ceil(scatterGridNum.x) *
//             (int)ceil(scatterGridNum.y) * (int)ceil(scatterGridNum.z); const double
//             scatter_volume = image3dSize.voxelSize.x * image3dSize.voxelNum.x *
//                                           image3dSize.voxelSize.y *
//                                           image3dSize.voxelNum.y *
//                                           image3dSize.voxelSize.z *
//                                           image3dSize.voxelNum.z / totalNum * 1e3;
//             const double totalComptonCrossSection511keV =
//             calTotalComptonCrossSection(511.f); const double ScannerEff511keV =
//             calScannerEffwithScatter(511.f); double common_factor = 0.25 / M_PI *
//             scatter_volume *
//                                    ScannerEff511keV / totalComptonCrossSection511keV;
//             int scatterPointNum = static_cast<int>(m_scatterPoint.size());

//             // prepare device data
//             std::cout << "start allocate d_crystalPos" << std::endl;
//             auto d_crystalPos =
//             openpni::basic::make_cuda_sync_ptr<openpni::basic::Point3D<float>>(crystalPosition.data(),
//                                                                                               crystalPosition.size(),
//                                                                                               cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_scatterPoint" << std::endl;
//             auto d_scatterPoint =
//             openpni::basic::make_cuda_sync_ptr<scatterPoint3D>(m_scatterPoint.data(),
//                                                                                 m_scatterPoint.size(),
//                                                                                 cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_projectArea" << std::endl;
//             auto d_projectArea =
//             openpni::basic::make_cuda_sync_ptr<float>(m_projectArea.data(),
//                                                                       m_projectArea.size(),
//                                                                       cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_totalEmission" << std::endl;
//             auto d_totalEmission =
//             openpni::basic::make_cuda_sync_ptr<float>(m_totalEmission.data(),
//                                                                         m_totalEmission.size(),
//                                                                         cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_totalAttn" << std::endl;
//             auto d_totalAttn =
//             openpni::basic::make_cuda_sync_ptr<float>(m_totalAttn.data(),
//                                                                     m_totalAttn.size(),
//                                                                     cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_scannerEff" << std::endl;
//             auto d_scannerEff =
//             openpni::basic::make_cuda_sync_ptr<float>(m_scannerEff.data(),
//                                                                      m_scannerEff.size(),
//                                                                      cudaMemcpyHostToDevice);
//             std::cout << "start allocate d_scatterTriangle_noTailFitting" << std::endl;
//             auto d_scatMich_noTailFitting =
//             openpni::basic::make_cuda_sync_ptr<float>(scatterMich_noTailFitting,
//                                                                                  LORNum,
//                                                                                  cudaMemcpyHostToDevice);
//             // batch = 4096
//             int batch = 4096;
//             std::cout << "start doing kernel" << std::endl;
//             for (size_t lorIndex = 0; lorIndex < LORNum; lorIndex += batch)
//             {
//                 singleScatterSimulation_kernel_impl(
//                     d_scatMich_noTailFitting,
//                     d_crystalPos,
//                     d_scatterPoint,
//                     d_totalEmission,
//                     d_totalAttn,
//                     d_scannerEff,
//                     d_projectArea,
//                     lorIndex,
//                     scatterPointNum,
//                     binNum,
//                     sliceNum,
//                     crystalNumOneRing,
//                     ringNum,
//                     binNumOutFOVOneSide,
//                     m_scatterProtocal.scannerEffTableEnergy.low,
//                     m_scatterProtocal.scannerEffTableEnergy.high,
//                     m_scatterProtocal.scannerEffTableEnergy.otherInfo,
//                     LORNum - lorIndex,
//                     batch);
//                 cudaDeviceSynchronize();
//             }
//             cudaDeviceSynchronize();

//             std::cout << "kernel done" << std::endl;
//             d_scatMich_noTailFitting.copyToHost(scatterMich_noTailFitting);
//             std::cout << "copy to host done" << std::endl;
//             for (int i = 0; i < LORNum; i++)
//             {
//                 scatterMich_noTailFitting[i] *= common_factor;
//             }

//             return true;
//         }

bool tailFittingLeastSquareBySlice(
    float *scatterTriangle, const float *scatterTriangle_noTailFitting, const float *norm, const float *rand,
    const float *prompt, const openpni::example::RingPETScanner &scanner) {
  auto LORNum = scanner.LORNum();
  auto ringNum = scanner.ringNum();
  std::unique_ptr<float[]> scatterTriangle_noTailFittingTemp = std::make_unique<float[]>(LORNum);
  std::copy(&scatterTriangle_noTailFitting[0], &scatterTriangle_noTailFitting[0] + LORNum,
            &scatterTriangle_noTailFittingTemp[0]);
  double oldSum = 0;
  double newSum = 0;
  for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
    oldSum += scatterTriangle_noTailFittingTemp[LORIndex];
    scatterTriangle_noTailFittingTemp[LORIndex] *= norm[LORIndex];
    newSum += scatterTriangle_noTailFittingTemp[LORIndex];
  }
  double scale = oldSum / newSum;
  for (size_t LORIndex = 0; LORIndex < LORNum; LORIndex++)
    scatterTriangle_noTailFittingTemp[LORIndex] /= static_cast<float>(scale);

  // -ringNum + 1 < ringdiff < 0
#pragma omp parallel for num_threads(m_limitedThreadNum)
  for (int ringdiff = -ringNum + 1; ringdiff < 0; ringdiff++) {
    int ring1Start = 0;
    int ring2Start = ring1Start - ringdiff;
    // store scale value a and b
    std::vector<double> scale;
    for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring2Temp < ringNum; ring1Temp++, ring2Temp++) {
      double a = 1;
      double b = 0;
      getScaleBySlice(scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b, prompt, norm, rand, scanner);
      scale.push_back(a);
      scale.push_back(b);
    }
    averageFilterApply(scale);
    for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring2Temp < ringNum;
         ring1Temp++, ring2Temp++) { // ring1Temp will always start from 0
      double a = scale[2 * ring1Temp];
      double b = scale[2 * ring1Temp + 1];
      doScaleBySlice(scatterTriangle, scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b, norm,
                     scanner);
    }
  }
  // 0 <= ringdiff < ringNum
#pragma omp parallel for num_threads(m_limitedThreadNum)
  for (int ringdiff = 0; ringdiff < ringNum; ringdiff++) {
    int ring2Start = 0;
    int ring1Start = ringdiff - ring2Start;
    // store scale value a and b
    std::vector<double> scale;
    for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring1Temp < ringNum; ring1Temp++, ring2Temp++) {
      double a = 1;
      double b = 0;
      getScaleBySlice(scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b, prompt, norm, rand, scanner);
      scale.push_back(a);
      scale.push_back(b);
    }
    averageFilterApply(scale);
    for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring1Temp < ringNum;
         ring1Temp++, ring2Temp++) { // ring2Temp will always start from 0
      double a = scale[2 * ring2Temp];
      double b = scale[2 * ring2Temp + 1];
      doScaleBySlice(scatterTriangle, scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b, norm,
                     scanner);
    }
  }

  return true;
}

//     public: // main
//         template <typename ProjectionMethod = openpni::process::ProjectionMethodSiddon,
//                   typename ConvKernel,
//                   std::floating_point ImageValueType>
//         bool scatterCorrection_CUDA(
//             float *prompt_mich,  // reconMich
//             float *scatter_mich, // saving final result
//             float *attnImg,      // must have
//             float *norm,         // default : 1，only use norm mich with: float*
//             blockProfA, float* blockProfT, float*
//             crystalFct，它们是生成归一化mich是伴随生成的部分组件 float *rand, //
//             default : 0,must have in v3,需要smooth预处理 const
//             openpni::example::RingPETScanner &scanner,
//             basic::DataSetFromMich_CUDA<float> &dataSets,
//             const basic::Image3DGeometry &attnMap3dSize,
//             openpni::example::OSEMParam<ImageValueType> &OSEMparam,
//             const ConvKernel &kernel)
//         {
//             const auto LORNum = scanner.LORNum();
//             basic::TriangleIndex ti(scanner.totalCrystalNum());
//             auto cryPos = example::getCrystalPositionAllHost<float>(scanner);
//             const auto rearranger = openpni::example::Rearranger(scanner);
//             std::vector<float> scatterTmpMich(scanner.LORNum(), 0);
//             // 0.get a copy of michAdd
//             std::vector<float> originMichAdd(ti.fullLength(), 0);
//             dataSets.d_michAdd.copyToHost(originMichAdd.data());
//             std::cout << "initScatterPoint" << std::endl;
//             if (initScatterPoint(attnImg, attnMap3dSize) != true)
//             {
//                 std::cout << "error at initScatterPoint" << std::endl;
//                 return false;
//             }
//             std::cout << "initScannerEffTable" << std::endl;
//             if (initScannerEffTable() != true)
//             {
//                 std::cout << "error at initScannerEfftable" << std::endl;
//                 return false;
//             }
//             // std::cout << "initAttnCutBedCoff_CUDA" << std::endl;
//             // if (initAttnCutBedCoff_CUDA(attnImg, attnMap3dSize, cryPos, scanner) !=
//             true)
//             // {
//             //     std::cout << "error at initAttnCutBedCoff" << std::endl;
//             //     return false;
//             // }
//             if (initAttnCutBedCoff(attnImg, attnMap3dSize, cryPos, scanner) != true)
//             {
//                 std::cout << "error at initAttnCutBedCoff" << std::endl;
//                 return false;
//             }
//             std::cout << "initTotalAttn" << std::endl;
//             if (initTotalAttn(cryPos, attnImg, attnMap3dSize) != true)
//             {
//                 std::cout << "error at initTotalAttn" << std::endl;
//                 return false;
//             }
//             std::cout << "InitProjectArea" << std::endl;
//             if (InitProjectArea(scanner, cryPos) != true) // these are not depended on
//             reconImg
//             {
//                 std::cout << "error at InitProjectArea" << std::endl;
//                 return false;
//             }
//             std::vector<ImageValueType> imgOSEM(attnMap3dSize.totalVoxelNum(), 0);
//             // 2.do OSEM
//             for (int iteration = 0; iteration < m_scatterProtocal.iterationNum;
//             iteration++)
//             {
//                 std::cout << "doing scatterEsitimate[" << iteration << "]" <<
//                 std::endl; bool hasSenmap = (iteration != 0);
//                 // osem only calSenmap at first time
//                 openpni::example::OSEM_CUDA<ProjectionMethod>(
//                     dataSets,
//                     attnMap3dSize,
//                     imgOSEM,
//                     kernel,
//                     OSEMparam,
//                     hasSenmap);
//                 std::cout << "doing totalEmission" << std::endl;
//                 // test out OSEM
//                 std::ofstream
//                 midOSEM("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/lgxtest/sss_midOSEM_CUDA0701.img3d",
//                 std::ios::binary); if (midOSEM.is_open())
//                 {
//                     midOSEM.write(reinterpret_cast<char *>(imgOSEM.data()),
//                     sizeof(float) * imgOSEM.size()); midOSEM.close();
//                 }
//                 if (initTotalEmission(cryPos, imgOSEM.data(), attnMap3dSize) != true)
//                 // depended on reconImg
//                 {
//                     std::cout << "error at initTotalEmission" << std::endl;
//                     return false;
//                 }
//                 std::vector<float> scattermich_noTailFitting(LORNum, 0);
//                 std::cout << "doing singleScatterSimulation" << std::endl;
//                 if (singleScatterSimulation_CUDA(
//                         scattermich_noTailFitting.data(),
//                         cryPos,
//                         scanner,
//                         imgOSEM.data(),
//                         attnMap3dSize) != true)
//                 {
//                     std::cout << "error at SingleScatterSimulation_CUDA" << std::endl;
//                     return false;
//                 }
//                 // test singleSCat
//                 std::cout << "singleScatterSimulation finished" << std::endl;
//                 std::ofstream
//                 mideScat("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/lgxtest/sss_midSctNoFit0624.img3d",
//                 std::ios::binary); if (mideScat.is_open())
//                 {
//                     mideScat.write(reinterpret_cast<char
//                     *>(scattermich_noTailFitting.data()), sizeof(float) *
//                     scattermich_noTailFitting.size()); mideScat.close();
//                 }
//                 // 3. do tail fittings
//                 std::cout << "doing tailFittingLeastSquareBySlice" << std::endl;
//                 std::fill(scatterTmpMich.begin(), scatterTmpMich.end(), 0.0f);
//                 if (tailFittingLeastSquareBySlice(
//                         scatterTmpMich.data(),
//                         scattermich_noTailFitting.data(),
//                         norm,
//                         rand,
//                         prompt_mich,
//                         scanner) != true)
//                 {
//                     std::cout << "error at tailFittingLeastSquareBySlice" << std::endl;
//                     return false;
//                 }
//                 std::cout << "finish tailFittingLeastSquareBySlice" << std::endl;
//                 // 4. update michAdd
//                 std::vector<float> scatterTmpTriangle(ti.fullLength(), 0);
//                 process::rearrageMich2Triangle(
//                     scatterTmpMich.data(),
//                     scatterTmpTriangle.data(),
//                     rearranger);
//                 std::vector<float> tempTriangleAdd(ti.fullLength(), 0);
//                 for (int i = 0; i < ti.fullLength(); i++)
//                 {
//                     tempTriangleAdd[i] = originMichAdd[i] + scatterTmpTriangle[i];
//                 }
//                 auto d_tempTriangleAdd =
//                 openpni::basic::make_cuda_sync_ptr<float>(tempTriangleAdd.data(),
//                                                                               tempTriangleAdd.size(),
//                                                                               cudaMemcpyHostToDevice);
//                 dataSets.d_michAdd.swap(d_tempTriangleAdd);
//             }
//             for (int i = 0; i < LORNum; i++)
//             {
//                 scatter_mich[i] = scatterTmpMich[i];
//             }
//             return true;
//         }

//     public:
//         int m_limitedThreadNum = 8;

//     private:
//         ScatterCorrection::scatterProtocal m_scatterProtocal;
//         std::vector<float> m_scannerEff;            // energyEffTable,which size is
//         int((upperEnergy - lowerEnergy) / energyInterval) + 1)
//         std::vector<scatterPoint3D> m_scatterPoint; // 散射点location + mu
//         std::vector<float> m_totalAttn;             // size = model中cry数目 *
//         scatterPoint中location的数目 std::vector<float> m_attnCutBedCoff;        // fwd
//         of umapCutBed std::vector<float> m_totalEmission; std::vector<float>
//         m_projectArea;
//     };
}; // namespace openpni::example::polygon::corrections