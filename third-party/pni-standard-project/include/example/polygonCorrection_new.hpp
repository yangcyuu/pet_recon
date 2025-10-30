#include "../experimental/node/MichAttn.hpp"
#include "../experimental/node/MichDeadTime.hpp"
#include "../experimental/node/MichNorm.hpp"
#include "../experimental/node/MichRandom.hpp"
#include "../experimental/node/MichScatter.hpp"
namespace pninew = openpni::experimental;
// //======deadTime tables
inline void generateDeadTimeTable(
    pninew::node::DeadTimeTable *__out_dtTable, std::vector<float *> __in_prompMichVecs,
    std::vector<float *> __in_delayMichVecs, std::vector<float> __in_scanTimes, std::vector<float> __in_activities,
    const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichDeadTime cfdt(__michDefine);
}

//======rand
inline void randCorrection(
    float *__out_randMich, float *__in_randMich, unsigned __minSectorDifference, unsigned __radialModuleNumS,
    const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichRandom randCorr(__michDefine);
  randCorr.setMinSectorDifference(__minSectorDifference);
  randCorr.setRadialModuleNumS(__radialModuleNumS);
  randCorr.setBadChannelThreshold(0.0f);
  randCorr.setDelayMich(__in_randMich);
  auto result = randCorr.dumpFactorsAsHMich();
  __out_randMich = result.release();
}
//======attn
inline void calAttnCoffWithCTImg(
    float *__out_AttnFactor, float *__in_AttnMap, pninew::core::Grids<3> attnmap,
    const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichAttn attnCorr(__michDefine);
  attnCorr.setMapSize(attnmap);
  attnCorr.bindHAttnMap(__in_AttnMap);
  auto result = attnCorr.dumpAttnMich();
  __out_AttnFactor = result.release();
}
inline void calAttnCoffWithHUMap(
    float *__out_AttnFactor, float *__in_HUMap, pninew::core::Grids<3> attnmap,
    const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichAttn attnCorr(__michDefine);
  attnCorr.setMapSize(attnmap);
  attnCorr.bindHHUMap(__in_HUMap);
  auto result = attnCorr.dumpAttnMich();
  __out_AttnFactor = result.release();
}
//======norm
inline void E180ActivityFWD() {}

inline void normCorrection(
    float *__out_normMich, float *__in_normScan, float *__in_activity, float __badChannelThreshold,
    float __fActCorrCutLow, float __fActCorrCutHigh, float __fCoffCutLow, float __fCoffCutHigh,
    float __radialModuleNumS, const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichNormalization norm(__michDefine);

  norm.setBadChannelThreshold(__badChannelThreshold);
  norm.setFActCorrCutLow(__fActCorrCutLow);
  norm.setFActCorrCutHigh(__fActCorrCutHigh);
  norm.setFCoffCutLow(__fCoffCutLow);
  norm.setFCoffCutHigh(__fCoffCutHigh);
  norm.setRadialModuleNumS(__radialModuleNumS);
  norm.bindComponentNormScanMich(__in_normScan);
  norm.bindComponentNormIdealMich(__in_activity);

  auto result = norm.dumpNormalizationMich();
  __out_normMich = result.release();
}
inline void selfNormCorrection(
    float *__out_selfNormMich, float *__in_delayMich, float *__in_normScan, float *__in_activity,
    float __badChannelThreshold, float __fActCorrCutLow, float __fActCorrCutHigh, float __fCoffCutLow,
    float __fCoffCutHigh, float __radialModuleNumS, const pninew::core::MichDefine __michDefine) {
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichNormalization norm(__michDefine);

  norm.setBadChannelThreshold(__badChannelThreshold);
  norm.setFActCorrCutLow(__fActCorrCutLow);
  norm.setFActCorrCutHigh(__fActCorrCutHigh);
  norm.setFCoffCutLow(__fCoffCutLow);
  norm.setFCoffCutHigh(__fCoffCutHigh);
  norm.setRadialModuleNumS(__radialModuleNumS);
  norm.bindComponentNormScanMich(__in_normScan);
  norm.bindComponentNormIdealMich(__in_activity);
  norm.bindSelfNormMich(__in_delayMich);

  auto result = norm.dumpNormalizationMich();
  __out_selfNormMich = result.release();
}
inline void selfNormCorrection(
    float *__out_selfNormMich, float *__in_delayMich, std::string __in_factorsPath,
    const pninew::core::MichDefine __michDefine) // get self-norm factors by factors
{
  pninew::node::MichNormalization norm(__michDefine);
  norm.recoverFromFile(__in_factorsPath);
  norm.bindSelfNormMich(__in_delayMich);
  auto result = norm.dumpNormalizationMich();
  __out_selfNormMich = result.release();
}
inline void deadTimeCorrection(
    std::string __in_factorsPath, float *__out_dtMich, pninew::node::DeadTimeTable __dtTable,
    const pninew::core::MichDefine __michDefine) // get deadTime-norm factors by factors
{
  auto michInfo = pninew::core::MichInfoHub::create(__michDefine);
  pninew::node::MichNormalization norm(__michDefine);
  norm.recoverFromFile(__in_factorsPath); // this can be norm factors or self-norm factors
  norm.setDeadTimeTable(__dtTable);
  auto result = norm.dumpNormalizationMich();
  __out_dtMich = result.release();
}
//======scatter
struct sssParams {
  double scatterPointsThreshold{0.00124};
  double taiFittingThreshold{0.95};
  int minSectorDifference{4};
  int radialModuleNumS{6};
  float BadChannelThreshold{0.02};
  pninew::core::Vector<double, 3> scatterEnergyWindow{350.00, 650.00, 0.15};
  pninew::core::Vector<double, 3> scatterEffTableEnergy{0.01, 700.00, 0.01};
  pninew::core::Grids<3> attnMap;
  pninew::core::Grids<3> emissionMap;
};
inline void scatterCorrection(
    float *__out_scatterMich, float *__in_AttnMap, std::string __in_normCoffPath, sssParams __sssParams,
    const pninew::core::MichDefine __michDefine) {
  pninew::node::MichScatter scatter(__michDefine);
  // set sss params
  scatter.setScatterPointsThreshold(__sssParams.scatterPointsThreshold);
  scatter.setTailFittingThreshold(__sssParams.taiFittingThreshold);
  scatter.setScatterEnergyWindow(__sssParams.scatterEnergyWindow);
  scatter.setScatterEffTableEnergy(__sssParams.scatterEffTableEnergy);
  scatter.setMinSectorDifference(__sssParams.minSectorDifference);
  // bind inputs
  pninew::node::MichAttn attnCoff(__michDefine);
  pninew::node::MichNormalization norm(__michDefine);
  pninew::node::MichRandom randCorr(__michDefine);
  attnCoff.setMapSize(__sssParams.attnMap);
  attnCoff.bindHAttnMap(__in_AttnMap);
  norm.recoverFromFile(__in_normCoffPath);
}