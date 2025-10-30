#pragma once
#include <mutex>

#include "Copy.h"
#include "MichCrystal.h"
#include "MichFactors.hpp"
#include "Projection.h"
#include "Share.hpp"
#include "ShellMichHelper.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/file/NormFile.hpp"
#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::node {

class MichNormalization_impl {
public:
  explicit MichNormalization_impl(
      core::MichDefine __mich)
      : m_michDefine(__mich)
  // , m_shellMichHelper(__mich)
  {}
  ~MichNormalization_impl() {};
  std::unique_ptr<MichNormalization_impl> copy();

public:
  void setShell(                    // 设置壳体的尺寸
      float innerRadius,            // 壳体内半径，单位mm
      float outerRadius,            // 壳体外半径，单位mm
      float axialLength,            // 壳体轴向长度，单位mm
      float parallaxScannerRadial,  // 视差效应下的扫描仪半径
      core::Grids<3, float> grids); // 视差效应下的图像的网格
  void setFActCorrCutLow(float v);
  void setFActCorrCutHigh(float v);
  void setFCoffCutLow(float v);
  void setFCoffCutHigh(float v);
  void setBadChannelThreshold(float v);
  void setRadialModuleNumS(int v);
  void setDeadTimeTable(DeadTimeTable dtTable);
  void bindShellScanMich(float *promptMich);
  void bindShellFwdMich(float *fwdMich);
  void bindSelfNormMich(float *delayMich);
  void addSelfNormListmodes(std::span<basic::Listmode_t const> listmodes);
  std::unique_ptr<float[]> getActivityMich();
  std::unique_ptr<float[]> dumpNormalizationMich();
  float const *getHNormFactorsBatch(std::span<core::MichStandardEvent const> events, FactorBitMask im);
  float const *getDNormFactorsBatch(std::span<core::MichStandardEvent const> events, FactorBitMask im);
  float const *getHNormFactorsBatch(std::span<std::size_t const> lorIndices, FactorBitMask im);
  float const *getDNormFactorsBatch(std::span<std::size_t const> lorIndices, FactorBitMask im);
  void saveToFile(std::string path);
  void recoverFromFile(std::string path);

public:
  std::unique_ptr<float[]> dumpCryFctMich();
  std::unique_ptr<float[]> dumpBlockFctMich();
  std::unique_ptr<float[]> dumpRadialFctMich();
  std::unique_ptr<float[]> dumpPlaneFctMich();
  std::unique_ptr<float[]> dumpInterferenceFctMich();
  std::unique_ptr<float[]> dumpDTComponentMich();
  std::unique_ptr<float[]> dumpFactors(FactorBitMask im);

private:
  bool cutBin(int &binStart, int &binEnd, float *normScan_mich, float *fwd_mich);
  bool ActivityCorr(float *normScan_mich, const float *fwd_mich);
  bool calCryCount(const float *normScan_mich);
  bool calBlockFct();
  bool calPlaneFct(const float *normScan_mich, const float *fwd_mich);
  bool calRadialFct(const float *normScan_mich, const float *fwd_mich);
  bool calInterferenceFct(const float *normScan_mich, const float *fwd_mich);
  bool binFctExtension(float *binFct, int binNum, int binStart, int binEnd, int binsToTrian);
  bool calExactCryFct(const float *normScan_mich);

private:
  void generateNormalization();
  void checkOrThrowGenerateFlags();
  bool ringScannerNormFctGenerate(float *normScan_mich, float *fwd_mich);
  bool selfNormalization();
  bool generateDeadTimeComponent();
  void resetGeneratedFlags();
  void checkDeviceFactors();

private:
  core::MichDefine m_michDefine;
  // temp
  std::vector<float> m_fwdInside;
  //
  std::vector<float> m_cryCount;
  std::vector<float> m_blockFctA; // axial block profile,size recommended to be crystalRingNum
  std::vector<float> m_blockFctT; // transaxial block profile,size recommended to be crystalRingNum * cryInPanelT
  std::vector<float> m_planeFct;
  std::vector<float> m_radialFct;
  std::vector<float> m_interferenceFct;
  std::vector<float> m_cryFct; // fansum
  std::vector<float> m_selfBlockFctA;
  std::vector<float> m_selfBlockFctT;
  std::vector<float> m_dtComponent;

  float m_fActCorrCutLow;      /**< Low threshold for activity correction. */
  float m_fActCorrCutHigh;     /**< High threshold for activity correction. */
  float m_fCoffCutLow;         /**< Low threshold for normalization coefficients. */
  float m_fCoffCutHigh;        /**< High threshold for normalization coefficients. */
  float m_BadChannelThreshold; /**< Bad channel threshold. */
  int m_radialModuleNumS;

  bool m_normFactorGenerated = false;
  bool m_selfNormGenerated = false;
  bool m_deadTimeGenerated = false;

  std::variant<std::monostate, float *> m_componentScanNormSource;
  std::variant<std::monostate, float *> m_componentFwdSource;
  std::vector<float> m_selfBlockLOR;
  std::vector<float> m_selfFanLOR;

  std::vector<float> mh_tempNormFactors;
  cuda_sync_ptr<float> md_tempNormFactors{"MichNorm_tempNormFactors"};
  std::vector<core::MichStandardEvent> mh_tempEvents;

  std::recursive_mutex m_mutex;

  // impl::ShellMichHelper m_shellMichHelper;

  std::optional<DeadTimeTable> m_deadTimeTable;

  cuda_sync_ptr<float> md_cryCount{"MichNorm_CryCount"};
  cuda_sync_ptr<float> md_blockFctA{"MichNorm_BlockFctA"};
  cuda_sync_ptr<float> md_blockFctT{"MichNorm_BlockFctT"};
  cuda_sync_ptr<float> md_planeFct{"MichNorm_PlaneFct"};
  cuda_sync_ptr<float> md_radialFct{"MichNorm_RadialFct"};
  cuda_sync_ptr<float> md_interferenceFct{"MichNorm_InterferenceFct"};
  cuda_sync_ptr<float> md_cryFct{"MichNorm_CryFct"};
  cuda_sync_ptr<float> md_selfBlockFctA{"MichNorm_SelfBlockFctA"};
  cuda_sync_ptr<float> md_selfBlockFctT{"MichNorm_SelfBlockFctT"};
  cuda_sync_ptr<float> md_dtComponent{"MichNorm_DTComponent"};
  bool m_normFactorInDevice = false;
};
} // namespace openpni::experimental::node

namespace openpni::experimental::node {
inline void MichNormalization_impl::bindShellScanMich(
    float *promptMich) {
  m_componentScanNormSource = promptMich;
  m_normFactorGenerated = false;
  m_normFactorInDevice = false;
}
inline void MichNormalization_impl::bindShellFwdMich(
    float *fwdMich) {
  m_componentFwdSource = fwdMich;
  m_normFactorGenerated = false;
  m_normFactorInDevice = false;
}
inline void MichNormalization_impl::resetGeneratedFlags() {
  m_normFactorGenerated = false;
  m_selfNormGenerated = false;
  m_deadTimeGenerated = false;
  m_normFactorInDevice = false;
}
inline bool MichNormalization_impl::cutBin(
    int &binStart, int &binEnd, float *normScan_mich, float *fwd_mich) {
  auto goodBinRange =
      impl::findGoodBinRange(m_michDefine, normScan_mich, fwd_mich, m_fActCorrCutLow, m_fActCorrCutHigh);
  binStart = goodBinRange[0];
  binEnd = goodBinRange[1];
  for (const auto [lor, b, v, s] : RangeGenerator(m_michDefine).allLORAndBinViewSlices())
    if (b < binStart || b >= binEnd)
      normScan_mich[lor] = 0;
  return true;
}
inline bool MichNormalization_impl::ActivityCorr(
    float *normScan_mich,
    const float *fwd_mich) // 输入筛选过 binAvailable的triangle
{
  for (const auto LORIndex : RangeGenerator(m_michDefine).allLORs())
    fwd_mich[LORIndex] > 0 ? normScan_mich[LORIndex] /= fwd_mich[LORIndex] : normScan_mich[LORIndex] = 0;
  return true;
}
inline bool MichNormalization_impl::calCryCount(
    const float *normScan_mich) {

  m_cryCount = impl::distributeMichToCrystalCounts(m_michDefine, normScan_mich);
  algorithms::set_max_value_to_1(std::span<float>(m_cryCount));
  for (auto &item : m_cryCount)
    if (item < m_BadChannelThreshold)
      item = 0;

  return true;
}
inline bool MichNormalization_impl::calBlockFct() {
  auto [blockFctA, blockFctT] = impl::calBlockFct(m_michDefine, m_cryCount.data());
  m_blockFctA = std::move(blockFctA);
  m_blockFctT = std::move(blockFctT);
  return true;
}
inline bool MichNormalization_impl::calPlaneFct(
    const float *normScan_mich, const float *fwd_mich) {
  m_planeFct = impl::calPlaneFct(m_michDefine, fwd_mich, normScan_mich, m_cryCount.data());
  return true;
}
inline bool MichNormalization_impl::calRadialFct(
    const float *normScan_mich, const float *fwd_mich) {
  m_radialFct = impl::calRadialFct(m_michDefine, fwd_mich, normScan_mich, m_cryCount.data());
  return true;
}
inline bool MichNormalization_impl::calInterferenceFct(
    const float *normScan_mich, const float *fwd_mich) {
  m_interferenceFct = impl::calInterferenceFct(m_michDefine, fwd_mich, normScan_mich, m_cryCount.data());
  return true;
}
inline bool MichNormalization_impl::binFctExtension(
    float *binFct, int binNum, int binStart, int binEnd, int binsToTrian) {
  return impl::binFctExtension(binFct, binNum, binStart, binEnd, binsToTrian);
}
inline bool MichNormalization_impl::calExactCryFct(
    const float *normScan_mich) {
  m_cryFct =
      impl::calExactCryFct(m_michDefine, normScan_mich, m_cryCount.data(), m_radialModuleNumS, m_BadChannelThreshold);

  return true;
}
inline void MichNormalization_impl::generateNormalization() {
  std::lock_guard __lock(m_mutex);
  if (!m_normFactorGenerated) {
    if (std::holds_alternative<float *>(m_componentScanNormSource) &&
        std::holds_alternative<float *>(m_componentFwdSource)) {
      float *normScan_mich = std::get<float *>(m_componentScanNormSource);
      float *fwd_mich = std::get<float *>(m_componentFwdSource);
      if (ringScannerNormFctGenerate(normScan_mich, fwd_mich)) {
        m_normFactorGenerated = true;
        m_normFactorInDevice = false;
      } else {
        throw exceptions::algorithm_unexpected_condition("Failed to generate normalization factors.");
      }
    } else {
      throw exceptions::algorithm_unexpected_condition("Component normalization source not bound.");
    }
  }
  if (!m_selfNormGenerated) {
    if (m_selfBlockLOR.size() && m_selfFanLOR.size())
      if (selfNormalization()) {
        m_selfNormGenerated = true;
        m_normFactorInDevice = false;
      } else {
        throw exceptions::algorithm_unexpected_condition("Failed to generate self-normalization factors.");
      }
  }
  if (!m_deadTimeGenerated) {
    if (m_deadTimeTable.has_value()) {
      if (generateDeadTimeComponent()) {
        m_deadTimeGenerated = true;
        m_normFactorInDevice = false;
      } else {
        throw exceptions::algorithm_unexpected_condition("Failed to generate dead time component.");
      }
    } else {
      // Do nothing, dead time correction is optional
    }
  }
}
inline void MichNormalization_impl::checkOrThrowGenerateFlags() {
  if (!m_normFactorGenerated) {
    throw exceptions::algorithm_unexpected_condition("Normalization factors not generated.");
  }
}
inline bool MichNormalization_impl::ringScannerNormFctGenerate(
    float *normScan_mich, float *fwd_mich) {
  auto binNum = MichInfoHub(m_michDefine).getBinNum();
  auto viewNum = MichInfoHub(m_michDefine).getViewNum();
  auto sliceNum = MichInfoHub(m_michDefine).getSliceNum();
  auto LORNum = MichInfoHub(m_michDefine).getLORNum();
  auto cryNum = MichInfoHub(m_michDefine).getTotalCrystalNum();
  auto crystalPerRing = MichInfoHub(m_michDefine).getCrystalNumOneRing();
  auto cryInBlockT = MichInfoHub(m_michDefine).getCrystalNumYInPanel();
  auto cryInPanelT = MichInfoHub(m_michDefine).getCrystalNumYInPanel();

  int binStart, binEnd;
  PNI_DEBUG("Doing cut bin...\n")
  if (cutBin(binStart, binEnd, normScan_mich, fwd_mich) != true) {
    PNI_DEBUG("Failed to cut bins.")
    return false;
  }
  PNI_DEBUG("Doing activity correction...\n")
  if (ActivityCorr(normScan_mich, fwd_mich) != true) {
    PNI_DEBUG("Failed to perform activity correction.")
    return false;
  }
  PNI_DEBUG("Calculating crystal counts...\n")
  if (calCryCount(normScan_mich) != true) {
    PNI_DEBUG("Failed to count crystals.\n")
    return false;
  }
  PNI_DEBUG("Calculating block factors...\n")
  if (calBlockFct() != true) {
    PNI_DEBUG("Failed to calculate block factors.\n")
    return false;
  }
  PNI_DEBUG("Applying block factors...\n")
  for (auto sl = 0; sl < sliceNum; sl++) {
    for (auto vi = 0; vi < viewNum; vi++) {
      for (auto bi = 0; bi < binNum; bi++) {
        size_t LORIndex = size_t(sl) * size_t(binNum * viewNum) + size_t(vi * binNum + bi);
        auto [rid1, rid2] = IndexConverter(m_michDefine).getCrystalIDFromLORID(LORIndex);
        int ring1 = rid1.ringID;
        int ring2 = rid2.ringID;
        int bv1 = rid1.idInRing % crystalPerRing % cryInPanelT;
        int bv2 = rid2.idInRing % crystalPerRing % cryInPanelT;
        auto blockFctSum = m_blockFctA[ring1] * m_blockFctA[ring2] * m_blockFctT[ring1 * cryInPanelT + bv1] *
                           m_blockFctT[ring2 * cryInPanelT + bv2];
        (blockFctSum > 0) ? normScan_mich[LORIndex] /= blockFctSum : normScan_mich[LORIndex] = 0;
      }
    }
  }
  if (calPlaneFct(normScan_mich, fwd_mich) != true) {
    PNI_DEBUG("Failed to calculate plane factors.\n")
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

  PNI_DEBUG("Calculating radial factors...\n")
  if (calRadialFct(normScan_mich, fwd_mich) != true) {
    PNI_DEBUG("Failed to calculate radial factors.")
    return false;
  }
  PNI_DEBUG("Extending bin factor extension...\n")
  if (binFctExtension(m_radialFct.data(), binNum, binStart, binEnd, 10) != true) {
    PNI_DEBUG("Failed to extend radial factors.\n")
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
  PNI_DEBUG("Calculating interference factors...\n")
  if (calInterferenceFct(normScan_mich, fwd_mich) != true) {
    PNI_DEBUG("Failed to calculate interference factors.\n")
    return false;
  }
  PNI_DEBUG("Extending interference bin factor extension...\n")
  for (int bv = 0; bv < cryInBlockT; ++bv) {
    if (binFctExtension(m_interferenceFct.data() + bv * binNum, binNum, binStart, binEnd, 10) != true) {
      PNI_DEBUG("Failed to extend interference factors.\n")
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
  PNI_DEBUG("Calculating exact crystal factors...\n")
  if (calExactCryFct(normScan_mich) != true) {
    PNI_DEBUG("Failed to calculate fanSum.\n")
    return false;
  }
  return true;
}
inline bool MichNormalization_impl::selfNormalization() {
  PNI_DEBUG("Starting self-normalization...\n")
  auto [delayCryCount, blockFctA, blockFctT] = impl::selfNormalization(m_michDefine, m_selfBlockLOR, m_selfFanLOR,
                                                                       m_blockFctA, m_blockFctT, m_BadChannelThreshold);
  m_cryCount = std::move(delayCryCount);
  m_selfBlockFctA = std::move(blockFctA);
  m_selfBlockFctT = std::move(blockFctT);
  PNI_DEBUG("Self-normalization done.\n")
  return true;
}
inline bool MichNormalization_impl::generateDeadTimeComponent() {
  m_dtComponent = impl::calDTComponent(m_michDefine, *m_deadTimeTable);
  m_deadTimeGenerated = true;
  return true;
}
inline std::unique_ptr<float[]> MichNormalization_impl::getActivityMich() {
  if (std::holds_alternative<float *>(m_componentFwdSource))
    return host_deep_copy(std::get<float *>(m_componentFwdSource), MichInfoHub(m_michDefine).getLORNum());
  else
    throw exceptions::algorithm_unexpected_condition("no Activity mich source not bound.");
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpNormalizationMich() {
  return dumpFactors(FactorBitMask::All);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpCryFctMich() {
  return dumpFactors(FactorBitMask::CryFct);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpBlockFctMich() {
  return dumpFactors(FactorBitMask::BlockFct);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpRadialFctMich() {
  return dumpFactors(FactorBitMask::RadialFct);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpPlaneFctMich() {
  return dumpFactors(FactorBitMask::PlaneFct);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpInterferenceFctMich() {
  return dumpFactors(FactorBitMask::InterferenceFct);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpDTComponentMich() {
  return dumpFactors(FactorBitMask::DTComponent);
}
inline std::unique_ptr<float[]> MichNormalization_impl::dumpFactors(
    FactorBitMask im) {
  generateNormalization();
  checkOrThrowGenerateFlags();

  std::unique_ptr<float[]> out_normCoff =
      std::make_unique_for_overwrite<float[]>(MichInfoHub(m_michDefine).getLORNum());
  tools::parallel_for_each(MichInfoHub(m_michDefine).getLORNum(), [&](std::size_t lorIndex) {
    auto blockFctA = m_selfNormGenerated ? m_selfBlockFctA.data() : m_blockFctA.data();
    auto blockFctT = m_selfNormGenerated ? m_selfBlockFctT.data() : m_blockFctT.data();
    auto dtComponent = m_deadTimeGenerated ? m_dtComponent.data() : nullptr;
    out_normCoff[lorIndex] =
        impl::calNormFactorsAll(m_michDefine, lorIndex, m_cryFct.data(), blockFctA, blockFctT, m_planeFct.data(),
                                m_radialFct.data(), m_interferenceFct.data(), dtComponent, im);
  });
  return out_normCoff;
}
inline void MichNormalization_impl::setShell(
    float innerRadius, float outerRadius, float axialLength, float parallaxScannerRadial,
    core::Grids<3, float> parallaxGrids) {
  auto shellPolygonMich = m_michDefine;
  shellPolygonMich.polygon.radius = parallaxScannerRadial;
  // create shell image
  std::vector<float> shellImage(parallaxGrids.totalSize());
  for (const auto &index : parallaxGrids.index_span()) {
    const auto point = parallaxGrids.voxel_center(index);
    auto [x, y, z] = point;
    if (algorithms::l2(point.right_shrink<1>()) >= innerRadius &&
        algorithms::l2(point.right_shrink<1>()) <= outerRadius && std::abs(point[2]) <= axialLength / 2.f) {
      shellImage[parallaxGrids.size[index]] = 1.f;
    } else {
      shellImage[parallaxGrids.size[index]] = 0.f;
    }
  }
  std::ofstream outFile("source.Img", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(shellImage.data()), shellImage.size() * sizeof(float));
  outFile.close();
  std::cout << "Shell image created, total voxel num: " << shellImage.size() << std::endl;
  // do fwd projection to get fwd mich
  m_fwdInside.resize(MichInfoHub(shellPolygonMich).getLORNum(), 0.f);
  auto michInfo = MichInfoHub(shellPolygonMich);
  auto michCrystal = MichCrystal(shellPolygonMich);
  for (const auto [begin, end] :
       tools::chunked_ranges_generator.by_max_size(0, michInfo.getMichSize(), 5 * 1024 * 1024)) {
    auto lors = tools::fill_vector(begin, end);
    auto *crystalGeoms = michCrystal.getHCrystalsBatch(std::span<std::size_t const>(lors));
    tools::parallel_for_each(lors.size(), [&](std::size_t index) {
      auto lor = lors[index];
      const auto cry1 = crystalGeoms[index * 2].O;
      const auto cry2 = crystalGeoms[index * 2 + 1].O;
      m_fwdInside[lor] =
          impl::instant_path_integral(core::instant_random_float(index),
                                      core::TensorDataInput<float, 3>{parallaxGrids, shellImage.data()}, cry1, cry2);
    });
  }
  m_componentFwdSource = m_fwdInside.data();
  m_normFactorGenerated = false;
  m_normFactorInDevice = false;
}

inline void MichNormalization_impl::setFActCorrCutLow(
    float v) {
  m_fActCorrCutLow = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::setFActCorrCutHigh(
    float v) {
  m_fActCorrCutHigh = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::setFCoffCutLow(
    float v) {
  m_fCoffCutLow = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::setFCoffCutHigh(
    float v) {
  m_fCoffCutHigh = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::setBadChannelThreshold(
    float v) {
  m_BadChannelThreshold = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::setRadialModuleNumS(
    int v) {
  m_radialModuleNumS = v;
  resetGeneratedFlags();
}
inline void MichNormalization_impl::bindSelfNormMich(
    float *delayMich) {
  auto [block, fan] = impl::calBlockLORAndFanLOR(m_michDefine, delayMich);
  m_selfBlockLOR = std::move(block);
  m_selfFanLOR = std::move(fan);
  m_selfNormGenerated = false;
}
inline void MichNormalization_impl::addSelfNormListmodes(
    std::span<basic::Listmode_t const> listmodes) {
  auto [block, fan] = impl::calBlockLORAndFanLOR(m_michDefine, listmodes);
  if (m_selfBlockLOR.size() != block.size() || m_selfFanLOR.size() != fan.size()) {
    m_selfBlockLOR = std::move(block);
    m_selfFanLOR = std::move(fan);
  } else {
    for (size_t i = 0; i < block.size(); ++i) {
      m_selfBlockLOR[i] += block[i];
    }
    for (size_t i = 0; i < fan.size(); ++i) {
      m_selfFanLOR[i] += fan[i];
    }
  }
  m_selfNormGenerated = false;
}
inline void MichNormalization_impl::setDeadTimeTable(
    DeadTimeTable dtTable) {
  m_deadTimeTable = dtTable;
  m_deadTimeGenerated = false;
}
inline float const *MichNormalization_impl::getHNormFactorsBatch(
    std::span<core::MichStandardEvent const> events, FactorBitMask im) {
  generateNormalization();
  checkOrThrowGenerateFlags();
  if (mh_tempNormFactors.size() < events.size())
    mh_tempNormFactors.resize(events.size());

  tools::parallel_for_each(events.size(), [&](std::size_t index) {
    auto lorIndex =
        IndexConverter(m_michDefine).getLORIDFromRectangleID(events[index].crystal1, events[index].crystal2);
    float *blockFctA = m_selfNormGenerated ? m_selfBlockFctA.data() : m_blockFctA.data();
    float *blockFctT = m_selfNormGenerated ? m_selfBlockFctT.data() : m_blockFctT.data();
    float *dtComponent = m_deadTimeGenerated ? m_dtComponent.data() : nullptr;
    mh_tempNormFactors[index] =
        impl::calNormFactorsAll(m_michDefine, lorIndex, m_cryFct.data(), blockFctA, blockFctT, m_planeFct.data(),
                                m_radialFct.data(), m_interferenceFct.data(), dtComponent, im);
  });
  return mh_tempNormFactors.data();
}
inline float const *MichNormalization_impl::getDNormFactorsBatch(
    std::span<core::MichStandardEvent const> events, FactorBitMask im) {
  generateNormalization();
  checkOrThrowGenerateFlags();
  checkDeviceFactors();

  md_tempNormFactors.reserve(events.size());

  float *blockFctA = m_selfNormGenerated ? md_selfBlockFctA.data() : md_blockFctA.data();
  float *blockFctT = m_selfNormGenerated ? md_selfBlockFctT.data() : md_blockFctT.data();
  float *dtComponent = m_deadTimeGenerated ? md_dtComponent.data() : nullptr;
  float *cryFct = md_cryFct.data();
  float *planeFct = md_planeFct.data();
  float *radialFct = md_radialFct.data();
  float *interFct = md_interferenceFct.data();
  impl::d_calNormFactorsAll(m_michDefine, events, md_tempNormFactors.get(), cryFct, blockFctA, blockFctT, planeFct,
                            radialFct, interFct, dtComponent, im);

  return md_tempNormFactors;
}
inline float const *MichNormalization_impl::getHNormFactorsBatch(
    std::span<std::size_t const> lorIndices, FactorBitMask im) {
  if (mh_tempEvents.size() < lorIndices.size())
    mh_tempEvents.resize(lorIndices.size());
  impl::h_fill_crystal_ids(mh_tempEvents.data(), lorIndices.data(), lorIndices.size(), m_michDefine);
  return getHNormFactorsBatch(std::span<core::MichStandardEvent const>(mh_tempEvents.data(), lorIndices.size()), im);
}
inline float const *MichNormalization_impl::getDNormFactorsBatch(
    std::span<std::size_t const> lorIndices, FactorBitMask im) {
  tl_mich_standard_events().reserve(lorIndices.size());
  impl::d_fill_crystal_ids(tl_mich_standard_events().get(), lorIndices.data(), lorIndices.size(), m_michDefine);
  return getDNormFactorsBatch(tl_mich_standard_events().cspan(lorIndices.size()), im);
}
inline void MichNormalization_impl::saveToFile(
    std::string path) {
  std::lock_guard __lock(m_mutex);
  generateNormalization();
  checkOrThrowGenerateFlags();

  file::MichNormalizationFile file(file::MichNormalizationFile::Write);
  file.setCryCount(m_cryCount);
  file.setBlockFctA(m_blockFctA);
  file.setBlockFctT(m_blockFctT);
  file.setPlaneFct(m_planeFct);
  file.setRadialFct(m_radialFct);
  file.setInterferenceFct(m_interferenceFct);
  file.setCryFct(m_cryFct);

  file.open(path);
}
inline void MichNormalization_impl::recoverFromFile(
    std::string path) {
  std::lock_guard __lock(m_mutex);
  file::MichNormalizationFile file(file::MichNormalizationFile::Read);
  file.open(path);

#define ASSIGN_IF_SIZE_CORRECT(member, getter, expectedSize)                                                           \
  if (file.getter().size() == expectedSize) {                                                                          \
    m_##member = file.getter();                                                                                        \
  } else {                                                                                                             \
    throw exceptions::algorithm_unexpected_condition(std::format(                                                      \
        "MichNormalizationFile " #member " size mismatch, expected {}, got {}", expectedSize, file.getter().size()));  \
  }

  ASSIGN_IF_SIZE_CORRECT(cryCount, getCryCount, MichInfoHub(m_michDefine).getTotalCrystalNum());
  ASSIGN_IF_SIZE_CORRECT(blockFctA, getBlockFctA, MichInfoHub(m_michDefine).getRingNum());
  ASSIGN_IF_SIZE_CORRECT(blockFctT, getBlockFctT,
                         MichInfoHub(m_michDefine).getRingNum() * MichInfoHub(m_michDefine).getCrystalNumYInPanel());
  ASSIGN_IF_SIZE_CORRECT(planeFct, getPlaneFct, MichInfoHub(m_michDefine).getSliceNum());
  ASSIGN_IF_SIZE_CORRECT(radialFct, getRadialFct, MichInfoHub(m_michDefine).getBinNum());
  ASSIGN_IF_SIZE_CORRECT(interferenceFct, getInterferenceFct,
                         MichInfoHub(m_michDefine).getCrystalNumYInPanel() * MichInfoHub(m_michDefine).getBinNum());
  ASSIGN_IF_SIZE_CORRECT(cryFct, getCryFct, MichInfoHub(m_michDefine).getTotalCrystalNum());
  m_normFactorGenerated = true;
  m_normFactorInDevice = false;
  PNI_DEBUG(std::format("MichNormalization recovered from file: {}\n", path));
}

inline void MichNormalization_impl::checkDeviceFactors() {
  if (!m_normFactorInDevice) {
    md_cryCount = make_cuda_sync_ptr_from_hcopy(m_cryCount, "Copy host from MichNorm_CryCount");
    md_blockFctA = make_cuda_sync_ptr_from_hcopy(m_blockFctA, "Copy host from MichNorm_BlockFctA");
    md_blockFctT = make_cuda_sync_ptr_from_hcopy(m_blockFctT, "Copy host from MichNorm_BlockFctT");
    md_planeFct = make_cuda_sync_ptr_from_hcopy(m_planeFct, "Copy host from MichNorm_PlaneFct");
    md_radialFct = make_cuda_sync_ptr_from_hcopy(m_radialFct, "Copy host from MichNorm_RadialFct");
    md_interferenceFct = make_cuda_sync_ptr_from_hcopy(m_interferenceFct, "Copy host from MichNorm_InterferenceFct");
    md_cryFct = make_cuda_sync_ptr_from_hcopy(m_cryFct, "Copy host from MichNorm_CryFct");
    if (m_selfNormGenerated) {
      md_selfBlockFctA = make_cuda_sync_ptr_from_hcopy(m_selfBlockFctA, "Copy host from MichNorm_SelfBlockFctA");
      md_selfBlockFctT = make_cuda_sync_ptr_from_hcopy(m_selfBlockFctT, "Copy host from MichNorm_SelfBlockFctT");
    } else {
      md_selfBlockFctA = decltype(md_selfBlockFctA){"Clear from MichNorm_SelfBlockFctA"};
      md_selfBlockFctT = decltype(md_selfBlockFctT){"Clear from MichNorm_SelfBlockFctT"};
    }
    if (m_deadTimeGenerated) {
      md_dtComponent = make_cuda_sync_ptr_from_hcopy(m_dtComponent, "Copy host from MichNorm_DTComponent");
    } else {
      md_dtComponent = decltype(md_dtComponent){"Clear from MichNorm_DTComponent"};
    }
    m_normFactorInDevice = true;
  }
}

inline std::unique_ptr<MichNormalization_impl> MichNormalization_impl::copy() {
  std::lock_guard __lock(m_mutex);
  generateNormalization();
  checkOrThrowGenerateFlags();
  auto new_impl = std::make_unique<MichNormalization_impl>(m_michDefine);
  new_impl->m_cryCount = m_cryCount;
  new_impl->m_blockFctA = m_blockFctA;
  new_impl->m_blockFctT = m_blockFctT;
  new_impl->m_planeFct = m_planeFct;
  new_impl->m_radialFct = m_radialFct;
  new_impl->m_interferenceFct = m_interferenceFct;
  new_impl->m_cryFct = m_cryFct;
  new_impl->m_selfBlockFctA = m_selfBlockFctA;
  new_impl->m_selfBlockFctT = m_selfBlockFctT;
  new_impl->m_dtComponent = m_dtComponent;
  new_impl->m_fActCorrCutLow = m_fActCorrCutLow;
  new_impl->m_fActCorrCutHigh = m_fActCorrCutHigh;
  new_impl->m_fCoffCutLow = m_fCoffCutLow;
  new_impl->m_fCoffCutHigh = m_fCoffCutHigh;
  new_impl->m_BadChannelThreshold = m_BadChannelThreshold;
  new_impl->m_radialModuleNumS = m_radialModuleNumS;
  new_impl->m_normFactorGenerated = m_normFactorGenerated;
  new_impl->m_selfNormGenerated = m_selfNormGenerated;
  new_impl->m_deadTimeGenerated = m_deadTimeGenerated;
  // No source binding in the copy, because the source may in other thread.
  new_impl->m_deadTimeTable = m_deadTimeTable;
  // No device factors in the copy, because the device factors may in other thread or stream.
  new_impl->m_normFactorInDevice = false;
  return new_impl;
}

} // namespace openpni::experimental::node