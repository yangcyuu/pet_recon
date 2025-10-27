#pragma once
#include "../process/Attenuation.hpp"
#include "../process/EM.hpp"
#include "../process/Norm.hpp"
#include "../process/Rand.hpp"
#include "PolygonalSystem.hpp"

namespace openpni::example {
inline process::RandDataView<float> generateRandDataView(
    float *out_randMich, float *in_randMich, polygon::PolygonModel &model, unsigned minSectorDifference,
    unsigned radialModuleNumS) {
  return {in_randMich,         out_randMich,    model.polygonSystem(), model.detectorInfo().geometry,
          minSectorDifference, radialModuleNumS};
}

inline void randCorrection(
    process::RandDataView<float> &randDataView) {
  process::rand::_rand{}(randDataView);
}

inline void randCorrection(
    float *out_randMich, float *in_randMich, polygon::PolygonModel &model, unsigned minSectorDifference,
    unsigned radialModuleNumS) {
  auto randDataView = generateRandDataView(out_randMich, in_randMich, model, minSectorDifference, radialModuleNumS);
  randCorrection(randDataView);
}

inline void calAttnCoffWithCTImg(
    float *out_AttnFactor, float *in_AttnMap, example::polygon::PolygonModel &model) {

  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = 0;
  dataView.crystalGeometry = model.crystalGeometry().data();

  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};
  Image3DSpan<const float> imgSpan{imgGeo, in_AttnMap};

  process::attn::cal_attn_coff(dataView, imgSpan, out_AttnFactor, process::attn::attn_model.v0(),
                               openpni::math::ProjectionMethodSiddon(), basic::CpuMultiThread::callWithAllThreads());
}

inline void calAttnCoffWithHUMap(
    float *out_AttnFactor, float *in_AttnMap, example::polygon::PolygonModel &model) {
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = 0;
  dataView.crystalGeometry = model.crystalGeometry().data();

  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};
  Image3DSpan<const float> imgSpan{imgGeo, in_AttnMap};

  process::attn::cal_attn_coff(dataView, imgSpan, out_AttnFactor, process::attn::attn_model.v1(),
                               openpni::math::ProjectionMethodSiddon(), basic::CpuMultiThread::callWithAllThreads());
}

process::normProtocol generateNormDataView(
    example::polygon::PolygonModel &model, float fActCorrCutLow, float fActCorrCutHigh, float fCoffCutLow,
    float fCoffCutHigh, float BadChannelThreshold, int radialModuleNumS) {
  return {model.polygonSystem(), model.detectorInfo().geometry, fActCorrCutLow,  fActCorrCutHigh, fCoffCutLow,
          fCoffCutHigh,          BadChannelThreshold,           radialModuleNumS};
}

inline void normCorrection(
    float *out_normCoff, float *in_normScanMich, float *in_fwdMich, process::normProtocol &normProtocal) {

  auto crystalPerRing = openpni::example::polygon::getCrystalNumOneRing(normProtocal.polygon, normProtocal.detectorGeo);
  auto cryInPlanelT = openpni::example::polygon::getCrystalNumYInPanel(normProtocal.polygon, normProtocal.detectorGeo);
  process::norm::Normalization normCorr(normProtocal);
  normCorr.ringScannerNormFctGenerate(in_normScanMich, in_fwdMich);

  for (auto [lor, bi, vi, sl, cry1, cry2] : normCorr.m_locator.allLORAndBinViewSlicesAndCrystals()) {
    int ring1 = cry1 / crystalPerRing;
    int ring2 = cry2 / crystalPerRing;
    int bv1 = cry1 % crystalPerRing % cryInPlanelT;
    int bv2 = cry2 % crystalPerRing % cryInPlanelT;
    int bv = vi % cryInPlanelT;
    float cryFct = normCorr.m_cryFct[cry1] * normCorr.m_cryFct[cry2];
    float blockFct = normCorr.m_blockFctA[ring1] * normCorr.m_blockFctA[ring2] *
                     normCorr.m_blockFctT[ring1 * cryInPlanelT + bv1] *
                     normCorr.m_blockFctT[ring2 * cryInPlanelT + bv2];
    float radialFct = normCorr.m_radialFct[bi];
    float planeFct = normCorr.m_planeFct[sl];
    float interFct = normCorr.m_interferenceFct[bv * normCorr.m_locator.bins().size() + bi];

    out_normCoff[lor] = cryFct * blockFct * radialFct * planeFct * interFct;
  }

  // // temp save sssnorm file
  // std::vector<float> tempSave(normCorr.m_locator.allLORs().size());
  // for (auto [lor, bi, vi, sl, cry1, cry2] : normCorr.m_locator.allLORAndBinViewSlicesAndCrystals()) {
  //   int ring1 = cry1 / crystalPerRing;
  //   int ring2 = cry2 / crystalPerRing;
  //   int bv1 = cry1 % crystalPerRing % cryInPlanelT;
  //   int bv2 = cry2 % crystalPerRing % cryInPlanelT;
  //   int bv = vi % cryInPlanelT;
  //   float cryFct = normCorr.m_cryFct[cry1] * normCorr.m_cryFct[cry2];
  //   float blockFct = normCorr.m_blockFctA[ring1] * normCorr.m_blockFctA[ring2] *
  //                    normCorr.m_blockFctT[ring1 * cryInPlanelT + bv1] *
  //                    normCorr.m_blockFctT[ring2 * cryInPlanelT + bv2];

  //   tempSave[lor] = cryFct * blockFct;
  // }
  // std::cout << "temp save sssnorm file" << std::endl;
  // std::ofstream outFile("/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0919/sssNormCorrection.bin",
  // std::ios::binary); outFile.write(reinterpret_cast<const char *>(tempSave.data()), tempSave.size() * sizeof(float));
  // outFile.close();
}

inline void normCorrection(
    float *out_normCoff, float *in_normScanMich, float *in_fwdMich, const float *in_delayMich,
    process::normProtocol &normProtocal) // with selfnormalization
{
  std::cout << "do self normalization" << std::endl;
  auto crystalPerRing = openpni::example::polygon::getCrystalNumOneRing(normProtocal.polygon, normProtocal.detectorGeo);
  auto cryInPlanelT = openpni::example::polygon::getCrystalNumYInPanel(normProtocal.polygon, normProtocal.detectorGeo);
  process::norm::Normalization normCorr(normProtocal);
  normCorr.ringScannerNormFctGenerate(in_normScanMich, in_fwdMich);
  normCorr.selfNormalization(in_delayMich);
  for (auto [lor, bi, vi, sl, cry1, cry2] : normCorr.m_locator.allLORAndBinViewSlicesAndCrystals()) {
    int ring1 = cry1 / crystalPerRing;
    int ring2 = cry2 / crystalPerRing;
    int bv1 = cry1 % crystalPerRing % cryInPlanelT;
    int bv2 = cry2 % crystalPerRing % cryInPlanelT;
    int bv = vi % cryInPlanelT;
    float cryFct = normCorr.m_cryFct[cry1] * normCorr.m_cryFct[cry2];
    float blockFct = normCorr.m_blockFctA[ring1] * normCorr.m_blockFctA[ring2] *
                     normCorr.m_blockFctT[ring1 * cryInPlanelT + bv1] *
                     normCorr.m_blockFctT[ring2 * cryInPlanelT + bv2];
    float radialFct = normCorr.m_radialFct[bi];
    float planeFct = normCorr.m_planeFct[sl];
    float interFct = normCorr.m_interferenceFct[bv * normCorr.m_locator.bins().size() + bi];

    out_normCoff[lor] = cryFct * blockFct * radialFct * planeFct * interFct;
    //  cryFct*blockFct * radialFct * planeFct *interFct;
  }
}

// void deadTimeReNormalization(
//     float *out_normCoff, float *in_normScanMich, float *in_fwdMich, double *in_calibrationTable, const int acqNum,
//     double scanTime, const int randomRateMin, process::normProtocol &normProtocal,
//     example::polygon::PolygonModel &model) {
//   auto dtProtocol = process::_deadTimeDataView<double>{
//       in_calibrationTable,           nullptr,  nullptr,       nullptr, nullptr, model.polygonSystem(),
//       model.detectorInfo().geometry, scanTime, randomRateMin, acqNum,  7};
//   process::norm::Normalization normCorr(normProtocal);
//   normCorr.reNormalizedByDT(out_normCoff, in_normScanMich, dtProtocol);
// }

inline void normCorrection(
    float *out_normCoff, float *in_normScanMich, float *in_fwdMich, example::polygon::PolygonModel &model,
    float fActCorrCutLow, float fActCorrCutHigh, float fCoffCutLow, float fCoffCutHigh, float BadChannelThreshold,
    int radialModuleNumS) {
  auto normProtocal = generateNormDataView(model, fActCorrCutLow, fActCorrCutHigh, fCoffCutLow, fCoffCutHigh,
                                           BadChannelThreshold, radialModuleNumS);
  normCorrection(out_normCoff, in_normScanMich, in_fwdMich, normProtocal);
}
inline void normCorrection(
    float *out_normCoff, float *in_normScanMich, float *in_fwdMich, const float *in_delayMich,
    example::polygon::PolygonModel &model, float fActCorrCutLow, float fActCorrCutHigh, float fCoffCutLow,
    float fCoffCutHigh, float BadChannelThreshold,
    int radialModuleNumS) // with selfnormalization
{
  auto normProtocal = generateNormDataView(model, fActCorrCutLow, fActCorrCutHigh, fCoffCutLow, fCoffCutHigh,
                                           BadChannelThreshold, radialModuleNumS);
  normCorrection(out_normCoff, in_normScanMich, in_fwdMich, in_delayMich, normProtocal);
}
// void deadTimeReNormalization(
//     float *out_normCoff, float *in_normScanMich, float *in_fwdMich, float *in_calibrationTable, const int acqNum,
//     double scanTime, const int randomRateMin, example::polygon::PolygonModel &model, float fActCorrCutLow,
//     float fActCorrCutHigh, float fCoffCutLow, float fCoffCutHigh, float BadChannelThreshold, int radialModuleNumS) {
//   auto normProtocal = generateNormDataView(model, fActCorrCutLow, fActCorrCutHigh, fCoffCutLow, fCoffCutHigh,
//                                            BadChannelThreshold, radialModuleNumS);
//   deadTimeReNormalization(out_normCoff, in_normScanMich, in_fwdMich, in_calibrationTable, acqNum, scanTime,
//                           randomRateMin, normProtocal, model);
// }
} // namespace openpni::example