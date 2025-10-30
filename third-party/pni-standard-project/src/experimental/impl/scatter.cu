#include <random>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "MichScatterImpl.hpp"
#include "Projection.h"
#include "Test.h"
#include "include/experimental/algorithms/EasyMath.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/tools/Loop.hpp"
namespace openpni::experimental::node::impl {
//==========common sss functions
std::vector<core::Vector<float, 3>> generateAllSSSPoints(
    core::Grids<3> __sssGrids) {
  std::vector<core::Vector<float, 3>> allPoints;
  for (auto z = 0; z < __sssGrids.size.dimSize[2]; ++z)
    for (auto y = 0; y < __sssGrids.size.dimSize[1]; ++y)
      for (auto x = 0; x < __sssGrids.size.dimSize[0]; ++x)
        allPoints.push_back(__sssGrids.voxel_center(core::Vector<int64_t, 3>::create(x, y, z)));
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::mt19937 rng(std::random_device{}());
  auto nextRandomPoint = [&]() { return core::Vector<float, 3>(dist(rng), dist(rng), dist(rng)); };
  std::transform(allPoints.begin(), allPoints.end(), allPoints.begin(),
                 [&](const core::Vector<float, 3> &p) { return p + nextRandomPoint() * __sssGrids.spacing; });
  return allPoints;
}
std::vector<ScatterPoint> isSSSPoints(
    const float *__attnMap, core::Grids<3> __attnMapGrids, const std::vector<core::Vector<float, 3>> &__allPoints,
    float __scatterPointThreshold) {
  std::vector<ScatterPoint> sssPoints;
  for (const auto idx : std::views::iota(0ull, __allPoints.size())) {
    auto randPoint = __allPoints[idx];
    auto indexPointIn = __attnMapGrids.find_index_from_float(randPoint);
    auto index = __attnMapGrids.at(indexPointIn);
    if (__attnMap[index] >= __scatterPointThreshold) {
      sssPoints.push_back({randPoint, __attnMap[index] * 10}); // float mu = __in_attnMapImage3dSpan.ptr[index]*10
    }
  };
  return sssPoints;
}
__PNI_CUDA_MACRO__ float calScannerEFFWithScatterEnergy(
    float energy, core::Vector<float, 3> scatterEnergyWindow) {
  float sigmaTimesSqrt2 = scatterEnergyWindow[2] * 511 / 2.35482f * 1.41421f;
  return core::FMath<float>::gauss_integral(scatterEnergyWindow[0], scatterEnergyWindow[1], energy, sigmaTimesSqrt2);
}
std::vector<float> generateScatterEffTable(
    core::Vector<float, 3> scannerEffTableEnergy, core::Vector<float, 3> scatterEnergyWindow) {
  int energyBinNum = int((scannerEffTableEnergy[1] - scannerEffTableEnergy[0]) / scannerEffTableEnergy[2]) + 1;
  std::vector<float> scannerEffTable(energyBinNum);
  tools::parallel_for_each(energyBinNum, [&](std::size_t energyBinIdx) {
    float energy = scannerEffTableEnergy[0] + energyBinIdx * scannerEffTableEnergy[2];
    scannerEffTable[energyBinIdx] = calScannerEFFWithScatterEnergy(energy, scatterEnergyWindow);
  });
  return scannerEffTable;
}
__PNI_CUDA_MACRO__ float calTotalComptonCrossSection511KeVRelative(
    float scatterEnergy) {
  const float a = scatterEnergy / 511.0;
  // Klein-Nishina formula for a=1 & devided with
  // 0.75 == (40 - 27*log(3)) / 9
  const float prefactor = 9.0 / (-40 + 27 * core::FMath<float>::flog(3.));
  return // checked this in Mathematica
      prefactor * (((-4 - a * (16 + a * (18 + 2 * a))) / ((1 + 2 * a) * (1 + 2 * a)) +
                    ((2 + (2 - a) * a) * core::FMath<float>::flog(1 + 2 * a)) / a) /
                   (a * a));
}
__PNI_CUDA_MACRO__ float calTotalComptonCrossSection(
    float energy) {
  constexpr float PI = 3.1415926535898;
  constexpr float Re = 2.818E-13; // cm
  float k = energy / 511.f;
  float a = core::FMath<float>::flog(1 + 2 * k);
  float prefactor = 2 * PI * Re * Re;
  return prefactor * ((1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - a / k) + a / (2.0 * k) -
                      (1.0 + 3.0 * k) / (1.0 + 2.0 * k) / (1.0 + 2.0 * k));
}
__PNI_CUDA_MACRO__ float calDiffCrossSection(
    float scatCosTheta) {
  // Kelin-Nishina formula. re is classical electron radius
  const float Re = 2.818E-13;               // cm
  float waveRatio = 1 / (2 - scatCosTheta); //  lamda/lamda'
  return 0.5 * Re * Re * waveRatio * waveRatio * (waveRatio + 1 / waveRatio + scatCosTheta * scatCosTheta - 1);
}
__PNI_CUDA_MACRO__ float calTotalAttenInScatterEnergy(
    float totalAtten511, float scatterEnergy) {
  return core::FMath<float>::fpow(totalAtten511, calTotalComptonCrossSection511KeVRelative(scatterEnergy));
}
//========== sss functions
void d_generateSSS_AttnCoff(
    float *__d_out_sssCoff, const ScatterPoint *__d_sssPoints, const core::CrystalGeom *__d_crystalGeom,
    const float *__d_AttnMap, const core::Grids<3> __d_AttnMapGrids, size_t sssPointNum, size_t crystalNum) {
  auto span2 = core::MDSpan<2>::create(crystalNum, sssPointNum);
  tools::parallel_for_each_CUDA(span2, [=] __device__(decltype(span2)::index_type index) {
    auto [cryIndex, sssIndex] = index;
    __d_out_sssCoff[span2[index]] =
        instant_path_integral(core::instant_random_float(cryIndex * sssPointNum + sssIndex),
                              core::TensorDataInput<float, 3>{__d_AttnMapGrids, __d_AttnMap},
                              __d_crystalGeom[cryIndex].O, __d_sssPoints[sssIndex].sssPosition);
    __d_out_sssCoff[span2[index]] = core::FMath<float>::fexp(-__d_out_sssCoff[span2[index]]);
  });
}
void d_generateSSS_EmissionCoff(
    float *__d_out_sssCoff, const ScatterPoint *__d_sssPoints, const core::CrystalGeom *__d_crystalGeom,
    const float *__d_eMap, const core::Grids<3> __d_emapGrids, size_t sssPointNum, size_t crystalNum) {
  auto span2 = core::MDSpan<2>::create(crystalNum, sssPointNum);
  tools::parallel_for_each_CUDA(span2, [=] __device__(decltype(span2)::index_type index) {
    auto [cryIndex, sssIndex] = index;
    __d_out_sssCoff[span2[index]] = instant_path_integral(
        core::instant_random_float(span2[index]), core::TensorDataInput<float, 3>{__d_emapGrids, __d_eMap},
        __d_crystalGeom[cryIndex].O, __d_sssPoints[sssIndex].sssPosition);
  });
}

float calSSSCommonFactors(
    const core::Grids<3> __attnMap, const core::Vector<float, 3> sssGridSize,
    const core::Vector<float, 3> scatterEnergyWindow) {
  auto sssGridNum = __attnMap.spacing * __attnMap.size.dimSize / sssGridSize;
  int sssTotalNum = (int)ceil(sssGridNum[0]) * (int)ceil(sssGridNum[1]) * (int)ceil(sssGridNum[2]);
  const constexpr float PI = 3.1415926535898f;
  const float sssAverageVolume = __attnMap.spacing[0] * __attnMap.spacing[1] * __attnMap.spacing[2] *
                                 __attnMap.size.dimSize[0] * __attnMap.size.dimSize[1] * __attnMap.size.dimSize[2] /
                                 sssTotalNum * 1e3;
  float totalComptonCrossSection511keV = calTotalComptonCrossSection(511.f);
  float scannerEff511KeV = calScannerEFFWithScatterEnergy(511.f, scatterEnergyWindow);
  float commonfactor = 0.25f / PI * sssAverageVolume * scannerEff511KeV / totalComptonCrossSection511keV;
  return commonfactor;
}
std::vector<float> generateGaussianBlurKernel(
    float __systemTimeRes_ns, float __TOFBinWidth) {
  float sigma = __systemTimeRes_ns / 2.355f; // unit: ns
  int validTofBinNumHalf = int(3.0f * sigma / __TOFBinWidth);
  std::vector<float> gauss(2 * validTofBinNumHalf + 1, 0);
  for (auto idx : std::views::iota(0, validTofBinNumHalf)) {
    int idp = validTofBinNumHalf + idx;
    int idn = validTofBinNumHalf - idx;
    gauss[idp] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
    gauss[idn] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
  }
  return gauss;
}
__PNI_CUDA_MACRO__ p3df polylinePosition(
    p3df A, p3df middle, p3df B, float distancePA_PB) {
  const auto mA_mB = algorithms::l2(A - middle) - algorithms::l2(B - middle);
  if (distancePA_PB < mA_mB) { // Thus: point is at line segment A-middle
    const auto PA_PM = distancePA_PB + algorithms::l2(B - middle);
    const auto PA = algorithms::l2(A - middle) / 2;
    return A + algorithms::normalized((middle - A)) * PA;
  } else { // Thus: point is at line segment middle-B
    const auto PM_PB = distancePA_PB - algorithms::l2((A - middle));
    const auto PM = (algorithms::l2((B - middle)) + PM_PB) / 2;
    return middle + algorithms::normalized((B - middle)) * PM;
  }
}
__PNI_CUDA_MACRO__ core::Vector<float, 2> calEmissionIntergralTOF(
    const p3df __cry1Pos, const p3df __cry2Pos, const p3df __scatPos, const float *__emap,
    const core::Grids<3> __emapGrid, float __TOFBinWidth, int __tofBinIdx) {
  auto cry1_cry2 = __cry2Pos - __cry1Pos;
  auto distance_cry1_cry2 = algorithms::l2(cry1_cry2);
  float line_binStart = __tofBinIdx * __TOFBinWidth;
  float line_binEnd = core::FMath<float>::min((__tofBinIdx + 1) * __TOFBinWidth, distance_cry1_cry2);
  // when P is binStart
  float distanceP1A = line_binStart;
  float distanceP1B = distance_cry1_cry2 - line_binStart;
  float distanceP1A_P1B = distanceP1A - distanceP1B;
  auto polylineStart = polylinePosition(__cry1Pos, __scatPos, __cry2Pos, distanceP1A_P1B);
  // when P is binEnd
  float distanceP2A = line_binEnd;
  float distanceP2B = distance_cry1_cry2 - line_binEnd;
  float distanceP2A_P2B = distanceP2A - distanceP2B;
  auto polylineEnd = polylinePosition(__cry1Pos, __scatPos, __cry2Pos, distanceP2A_P2B);
  // SA_SB
  float distanceSA_SB = algorithms::l2((__scatPos - __cry1Pos)) - algorithms::l2((__scatPos - __cry2Pos));
  auto deltaLen1 = distanceP1A_P1B - distanceSA_SB;
  auto deltaLen2 = distanceP2A_P2B - distanceSA_SB;
  float TOFIntergral1;
  float TOFIntergral2;

  if (deltaLen1 < 0 && deltaLen2 < 0) // case1 binstart binend all on sa
  {
    TOFIntergral2 = 0;
    TOFIntergral1 =
        instant_path_integral(core::instant_random_float((__tofBinIdx + 1) * 1000),
                              core::TensorDataInput<float, 3>{__emapGrid, __emap}, polylineStart, polylineEnd);

  } else if (deltaLen1 > 0 && deltaLen2 > 0) // case2 binstart binend all on sb
  {
    TOFIntergral1 = 0;
    TOFIntergral2 =
        instant_path_integral(core::instant_random_float((__tofBinIdx + 1) * 1000),
                              core::TensorDataInput<float, 3>{__emapGrid, __emap}, polylineStart, polylineEnd);
  } else if (deltaLen1 * deltaLen2 < 0) {
    TOFIntergral1 =
        instant_path_integral(core::instant_random_float((__tofBinIdx + 1) * 1000),
                              core::TensorDataInput<float, 3>{__emapGrid, __emap}, polylineStart, __scatPos);
    TOFIntergral2 = instant_path_integral(core::instant_random_float((__tofBinIdx + 1) * 1000),
                                          core::TensorDataInput<float, 3>{__emapGrid, __emap}, __scatPos, polylineEnd);
  }
  return core::Vector<float, 2>::create(TOFIntergral1, TOFIntergral2);
}

void singleScatterSimulation(
    sssDataView __d_sssData) {
  auto michInfo = core::MichInfoHub::create(__d_sssData.__michDefine);
  auto converter = core::IndexConverter::create(__d_sssData.__michDefine);
  auto crystalNum = michInfo.getTotalCrystalNum();
  auto lorNum = michInfo.getLORNum();
  auto binNum = michInfo.getBinNum();
  auto binNumOutFOVOneSide = core::mich::calBinNumOutFOVOneSide(
      __d_sssData.__michDefine.polygon, __d_sssData.__michDefine.detector, __d_sssData.__minSectorDifference);
  const float crystalArea = __d_sssData.__michDefine.detector.crystalSizeU *
                            __d_sssData.__michDefine.detector.crystalSizeV * 0.01; // 单个晶体的面积position,cm2
  auto span2 = core::MDSpan<2>::create(crystalNum, __d_sssData.__countScatter);
  for (const auto [scatterIndexBegin, scatterIndexEnd] :
       tools::chunked_ranges_generator.by_balanced_max_size(0, __d_sssData.__countScatter, 50)) {
    PNI_DEBUG("Calculating scatter index " + std::to_string(scatterIndexBegin) + "~" + std::to_string(scatterIndexEnd) +
              " / " + std::to_string(__d_sssData.__countScatter) + "...\n");
    tools::parallel_for_each_CUDA(lorNum, [=] __device__(size_t __lorIndex) {
      for (int scatterIndex = scatterIndexBegin; scatterIndex < scatterIndexEnd; ++scatterIndex) {
        auto [cry1R, cry2R] = converter.getCrystalIDFromLORID(__lorIndex);
        auto cry1FlatR = core::mich::getFlatIdFromRectangleID(__d_sssData.__michDefine.polygon,
                                                              __d_sssData.__michDefine.detector, cry1R);
        auto cry2FlatR = core::mich::getFlatIdFromRectangleID(__d_sssData.__michDefine.polygon,
                                                              __d_sssData.__michDefine.detector, cry2R);
        int binIndex = __lorIndex % binNum;
        if (binIndex < binNumOutFOVOneSide || binIndex >= binNum - binNumOutFOVOneSide)
          return;

        int index1 = span2[core::Vector<int64_t, 2>::create(cry1FlatR, scatterIndex)];
        int index2 = span2[core::Vector<int64_t, 2>::create(cry2FlatR, scatterIndex)];

        auto s_cry1 =
            (__d_sssData.__d_crystalGeometry[cry1FlatR].O - __d_sssData.__scatterPoints[scatterIndex].sssPosition);
        auto s_cry2 =
            (__d_sssData.__d_crystalGeometry[cry2FlatR].O - __d_sssData.__scatterPoints[scatterIndex].sssPosition);

        float distance_s_cry1_sqare = s_cry1.l22() * 0.01f;     // cm2
        float distance_s_cry2_sqare = s_cry2.l22() * 0.01f;     // cm2
        float scatCosine = -algorithms::cosine(s_cry1, s_cry2); // notice the definition of cosTheta of scatter angle
        float diffCross = calDiffCrossSection(scatCosine);
        float scatterEnergy = 511 / (2 - scatCosine);
        if (scatterEnergy < __d_sssData.m_scatterEffTableEnergy[0] ||
            scatterEnergy > __d_sssData.m_scatterEffTableEnergy[1])
          return;
        int effTableINdex =
            (int)((scatterEnergy - __d_sssData.m_scatterEffTableEnergy[0]) / __d_sssData.m_scatterEffTableEnergy[2]);
        auto scatterEff = __d_sssData.__scannerEffTable[effTableINdex];

        auto n1_vector = algorithms::cross(__d_sssData.__d_crystalGeometry[cry1FlatR].U,
                                           __d_sssData.__d_crystalGeometry[cry1FlatR].V);
        auto n2_vector = algorithms::cross(__d_sssData.__d_crystalGeometry[cry2FlatR].U,
                                           __d_sssData.__d_crystalGeometry[cry2FlatR].V);
        float projectArea1 = algorithms::calculateProjectionArea(crystalArea, n1_vector, s_cry1);
        float projectArea2 = algorithms::calculateProjectionArea(crystalArea, n2_vector, s_cry2);

        float IntergralCry1 = __d_sssData.__sssAttnCoff[index1] * __d_sssData.__sssEmissionCoff[index1] *
                              calTotalAttenInScatterEnergy(__d_sssData.__sssAttnCoff[index2], scatterEnergy);
        float IntergralCry2 = __d_sssData.__sssAttnCoff[index2] * __d_sssData.__sssEmissionCoff[index2] *
                              calTotalAttenInScatterEnergy(__d_sssData.__sssAttnCoff[index1], scatterEnergy);

        float sssValue = 1.f;
        sssValue *= __d_sssData.__commonfactor;
        sssValue *= __d_sssData.__scatterPoints[scatterIndex].mu;
        sssValue *= diffCross;
        sssValue *= projectArea1;
        sssValue *= projectArea2;
        sssValue *= (IntergralCry1 + IntergralCry2);
        sssValue *= scatterEff;
        sssValue *= 1.f / (distance_s_cry1_sqare);
        sssValue *= 1.f / (distance_s_cry2_sqare);

        if (core::FMath<float>::isBad(sssValue) || sssValue < 0) // NaN or negative
          continue;

        __d_sssData.__out_d_scatterValue[__lorIndex] += sssValue;
      }
    });
  }
}
void sssNormalization(
    float *__d_scatterValue, MichNormalization &__norm, core::MichDefine __mich) {
  const auto lorNum = core::MichInfoHub::create(__mich).getLORNum();
  float oldSum = example::d_sum_reduce(__d_scatterValue, lorNum);
  if (oldSum <= 0)
    return;

  float newSum{0};
  LORBatch lorBatch(__mich);
  lorBatch.setSubsetNum(1).setCurrentSubset(0);
  for (auto lors = lorBatch.nextDBatch(); !lors.empty(); lors = lorBatch.nextDBatch()) {
    auto *normFactors = __norm.getDNormFactorsBatch(lors, static_cast<FactorBitMask>(CryFct | BlockFct));
    tools::parallel_for_each_CUDA(lors.size(), [=, lors = lors.data()] __device__(std::size_t i) {
      auto idx = lors[i];
      __d_scatterValue[idx] *= normFactors[i];
    });
    newSum += thrust::transform_reduce(
        thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
        thrust::counting_iterator<std::size_t>(lors.size()),
        [=, lors = lors.data()] __device__(std::size_t i) -> float { return __d_scatterValue[lors[i]]; }, 0.0f,
        thrust::plus<float>());
  }
  float scale = oldSum / newSum;
  example::d_parralel_div(__d_scatterValue, scale, __d_scatterValue, lorNum);
}
void tailFitting(
    float *__d_scatterValue, sssBatchLORGetor &__prompt, MichNormalization &__norm, MichRandom &__rand,
    MichAttn &__attnCutBedCoff, const core::MichDefine __michDefine, int minSectorDifference,
    double __tailFittingThreshold) {
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto sliceNum = michInfo.getSliceNum();
  auto binNum = michInfo.getBinNum();
  auto viewNum = michInfo.getViewNum();
  auto lorNumOneSlice = michInfo.getBinNum() * michInfo.getViewNum();
  auto binNumOutFOVOneSide =
      core::mich::calBinNumOutFOVOneSide(__michDefine.polygon, __michDefine.detector, minSectorDifference);

  cuda_sync_ptr<std::size_t> d_lorBatch{"ScatterTailFitting_lorBatch"};
  cuda_sync_ptr<float> d_XXFirst{"ScatterTailFitting_XXFirst"};
  cuda_sync_ptr<float> d_XYFirst{"ScatterTailFitting_XYFirst"};
  cuda_sync_ptr<float> d_XXSum{"ScatterTailFitting_XXSum"};
  cuda_sync_ptr<float> d_XYSum{"ScatterTailFitting_XYSum"};
  cuda_sync_ptr<float> d_tempXX{"ScatterTailFitting_tempXX"};
  cuda_sync_ptr<float> d_tempXY{"ScatterTailFitting_tempXY"};
  cuda_sync_ptr<float> d_tempReduceKeys{"ScatterTailFitting_tempReduceKeys"};
  for (const auto [sliceBegin, sliceEnd] :
       tools::chunked_ranges_generator.by_max_size(0, sliceNum, 5 * 1024 * 1024 / lorNumOneSlice)) {
    const auto batchSize = (sliceEnd - sliceBegin) * lorNumOneSlice;
    d_lorBatch.reserve(batchSize);
    tools::parallel_for_each_CUDA(
        batchSize, [sliceBegin, lorNumOneSlice, d_lorBatch = d_lorBatch.data()] __device__(std::size_t idx) {
          d_lorBatch[idx] = idx + sliceBegin * lorNumOneSlice;
        });
    // temp
    auto *promptValue = __prompt.getLORBatch(sliceBegin, sliceEnd, __michDefine);
    auto *attnFactors = __attnCutBedCoff.getDAttnFactorsBatch(d_lorBatch.cspan(batchSize));
    auto *normFactors = __norm.getDNormFactorsBatch(d_lorBatch.cspan(batchSize));
    auto *randFactors = __rand.getDRandomFactorsBatch(d_lorBatch.cspan(batchSize));
    // d_print_none_zero_average_value(randFactors, batchSize);
    // d_print_none_zero_average_value(promptValue, batchSize);
    // d_print_none_zero_average_value(randFactors, batchSize);
    // d_print_none_zero_average_value(attnFactors, batchSize);
    // std::cout << "Bin num out of FOV one side: " << binNumOutFOVOneSide << std::endl;
    // tools::parallel_for_each_CUDA(sliceBegin, sliceEnd, [=] __device__(std::size_t sliceId) {
    //   algorithms::LinearFittingHelper<float, algorithms::LinearFittingType::LinearFitting_NoBias> __linearFitModel;
    //   float Y_XSum = 0;
    //   float YSum = 0;
    //   for (size_t lorInsl = 0; lorInsl < lorNumOneSlice; ++lorInsl) {
    //     int binIndex = lorInsl % binNum;
    //     size_t lorIndex = sliceId * lorNumOneSlice + lorInsl;
    //     size_t indexOfFactors = (sliceId - sliceBegin) * lorNumOneSlice + lorInsl;
    //     // __d_scatterValue[lorIndex] = 0;
    //     if (attnFactors[indexOfFactors] >= __tailFittingThreshold && binIndex >= binNumOutFOVOneSide &&
    //         binIndex < binNum - binNumOutFOVOneSide) {
    //       if (normFactors[indexOfFactors] == 0)
    //         continue;
    //       __linearFitModel.add(__d_scatterValue[lorIndex], promptValue[indexOfFactors] -
    //       randFactors[indexOfFactors]); Y_XSum += promptValue[indexOfFactors] - randFactors[indexOfFactors]; YSum +=
    //       promptValue[indexOfFactors];
    //       // __d_scatterValue[lorIndex] = 1;
    //     }
    //   }
    //   // if (sliceId % 100 == 0)
    //   //   printf("Average Y_X: %f, Average Y: %f\n", Y_XSum / __linearFitModel.getCount(),
    //   //          YSum / __linearFitModel.getCount());
    //   for (size_t lorInsl = 0; lorInsl < lorNumOneSlice; ++lorInsl) {
    //     int binIndex = lorInsl % binNum;
    //     size_t lorIndex = sliceId * lorNumOneSlice + lorInsl;
    //     size_t indexOfFactors = (sliceId - sliceBegin) * lorNumOneSlice + lorInsl;
    //     if (binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
    //       if (normFactors[indexOfFactors] == 0) {
    //         __d_scatterValue[lorIndex] = 0;
    //         continue;
    //       }
    //       __d_scatterValue[lorIndex] = __linearFitModel.predict(__d_scatterValue[lorIndex]);
    //     } else
    //       __d_scatterValue[lorIndex] = 0;
    //   }
    // });
    d_XXFirst.reserve(sliceEnd - sliceBegin);
    d_XYFirst.reserve(sliceEnd - sliceBegin);
    d_XXSum.reserve(sliceEnd - sliceBegin);
    d_XYSum.reserve(sliceEnd - sliceBegin);
    d_tempXX.reserve(batchSize);
    d_tempXY.reserve(batchSize);
    d_tempReduceKeys.reserve(batchSize);
    tools::parallel_for_each_CUDA(batchSize, [=, d_tempXX = d_tempXX.data(), d_tempXY = d_tempXY.data(),
                                              d_tempReduceKeys = d_tempReduceKeys.data()] __device__(std::size_t idx) {
      size_t lorIndex = idx + sliceBegin * lorNumOneSlice;
      size_t key = idx / lorNumOneSlice;
      d_tempReduceKeys[idx] = key;
      int binIndex = lorIndex % binNum;
      d_tempXX[idx] = 0;
      d_tempXY[idx] = 0;
      if (attnFactors[idx] >= __tailFittingThreshold && binIndex >= binNumOutFOVOneSide &&
          binIndex < binNum - binNumOutFOVOneSide) {
        if (normFactors[idx] == 0) {
          return;
        }
        d_tempXX[idx] = __d_scatterValue[lorIndex] * __d_scatterValue[lorIndex];
        d_tempXY[idx] = __d_scatterValue[lorIndex] * (promptValue[idx] - randFactors[idx]);
        // printf("lorIndex:%d, scatterValue:%f, promptValue:%f, randFactor:%f, XX:%f, XY:%f\n", int(lorIndex),
        //        __d_scatterValue[lorIndex], promptValue[idx], randFactors[idx], d_tempXX[idx], d_tempXY[idx]);
      }
    });
    thrust::reduce_by_key(
        thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(d_tempReduceKeys.data()),
        thrust::device_pointer_cast(d_tempReduceKeys.data() + batchSize), thrust::device_pointer_cast(d_tempXX.data()),
        thrust::device_pointer_cast(d_XXFirst.data()), thrust::device_pointer_cast(d_XXSum.data()),
        thrust::equal_to<std::size_t>(), thrust::plus<float>());
    thrust::reduce_by_key(
        thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(d_tempReduceKeys.data()),
        thrust::device_pointer_cast(d_tempReduceKeys.data() + batchSize), thrust::device_pointer_cast(d_tempXY.data()),
        thrust::device_pointer_cast(d_XYFirst.data()), thrust::device_pointer_cast(d_XYSum.data()),
        thrust::equal_to<std::size_t>(), thrust::plus<float>());
    tools::parallel_for_each_CUDA(batchSize,
                                  [=, d_XX = d_XXSum.data(), d_XY = d_XYSum.data()] __device__(std::size_t idx) {
                                    size_t lorIndex = idx + sliceBegin * lorNumOneSlice;
                                    size_t key = idx / lorNumOneSlice;
                                    int binIndex = lorIndex % binNum;
                                    if (binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
                                      if (normFactors[idx] == 0) {
                                        __d_scatterValue[lorIndex] = 0;
                                        return;
                                      }
                                      float XX = d_XX[key];
                                      float XY = d_XY[key];
                                      if (XX == 0) {
                                        __d_scatterValue[lorIndex] = 0;
                                        return;
                                      }
                                      float slope = XY / XX;
                                      __d_scatterValue[lorIndex] = __d_scatterValue[lorIndex] * slope;
                                    } else
                                      __d_scatterValue[lorIndex] = 0;
                                  });
  }
}
//======= TOF function
void singleScatterSimulationTOF(
    sssTOFDataView __d_sssData) {
  auto dsMichInfo = core::MichInfoHub::create(__d_sssData.__dsmichDefine);
  auto dsConverter = core::IndexConverter::create(__d_sssData.__dsmichDefine);
  auto dsCrystalNum = dsMichInfo.getTotalCrystalNum();
  auto dslorNum = dsMichInfo.getLORNum();
  auto span2 = core::MDSpan<2>::create(dsCrystalNum, __d_sssData.__countScatter);

  for (const auto [scatterIndexBegin, scatterIndexEnd] :
       tools::chunked_ranges_generator.by_balanced_max_size(0, __d_sssData.__countScatter, 50)) {
    PNI_DEBUG("Calculating scatter index " + std::to_string(scatterIndexBegin) + "~" + std::to_string(scatterIndexEnd) +
              " / " + std::to_string(__d_sssData.__countScatter) + "...\n");
    tools::parallel_for_each_CUDA(dslorNum, [=] __device__(std::size_t dslorIdx) {
      auto [block1R, block2R] = dsConverter.getCrystalIDFromLORID(dslorIdx);
      auto block1FlatR = core::mich::getFlatIdFromRectangleID(__d_sssData.__dsmichDefine.polygon,
                                                              __d_sssData.__dsmichDefine.detector, block1R);
      auto block2FlatR = core::mich::getFlatIdFromRectangleID(__d_sssData.__dsmichDefine.polygon,
                                                              __d_sssData.__dsmichDefine.detector, block2R);

      for (int scatterIndex = 0; scatterIndex < __d_sssData.__countScatter; ++scatterIndex) {
        int index1 = span2[core::Vector<int64_t, 2>::create(block1FlatR, scatterIndex)];
        int index2 = span2[core::Vector<int64_t, 2>::create(block2FlatR, scatterIndex)];

        auto s_block1 =
            (__d_sssData.__d_dsCrystalGeometry[block1FlatR].O - __d_sssData.__scatterPoints[scatterIndex].sssPosition);
        auto s_block2 =
            (__d_sssData.__d_dsCrystalGeometry[block2FlatR].O - __d_sssData.__scatterPoints[scatterIndex].sssPosition);

        float distance_s_block1_sqare = s_block1.l22() * 0.01; // cm2
        float distance_s_block2_sqare = s_block2.l22() * 0.01; // cm2
        float scatCosine =
            -algorithms::cosine(s_block1, s_block2); // notice the definition of cosTheta of scatter angle
        float diffCross = calDiffCrossSection(scatCosine);
        float scatterEnergy = 511 / (2 - scatCosine);
        if (scatterEnergy < __d_sssData.m_scatterEffTableEnergy[0] ||
            scatterEnergy > __d_sssData.m_scatterEffTableEnergy[1])
          return;
        int effTableINdex =
            (int)((scatterEnergy - __d_sssData.m_scatterEffTableEnergy[0]) / __d_sssData.m_scatterEffTableEnergy[2]);
        auto scatterEff = __d_sssData.__scannerEffTable[effTableINdex];

        auto n1_vector = algorithms::cross(__d_sssData.__d_dsCrystalGeometry[block1FlatR].U,
                                           __d_sssData.__d_dsCrystalGeometry[block1FlatR].V);
        auto n2_vector = algorithms::cross(__d_sssData.__d_dsCrystalGeometry[block2FlatR].U,
                                           __d_sssData.__d_dsCrystalGeometry[block2FlatR].V);
        float projectArea1 = algorithms::calculateProjectionArea(__d_sssData.__crystalArea, n1_vector, s_block1);
        float projectArea2 = algorithms::calculateProjectionArea(__d_sssData.__crystalArea, n2_vector, s_block2);

        float AttnIntergralBlock1 = __d_sssData.__sssAttnCoff[index1] *
                                    calTotalAttenInScatterEnergy(__d_sssData.__sssAttnCoff[index2], scatterEnergy);
        float AttnIntergralBlock2 = __d_sssData.__sssAttnCoff[index2] *
                                    calTotalAttenInScatterEnergy(__d_sssData.__sssAttnCoff[index1], scatterEnergy);

        float sssValue = 1;
        sssValue *= __d_sssData.__scatterPoints[scatterIndex].mu;
        sssValue *= projectArea1;
        sssValue *= projectArea2;
        sssValue *= diffCross;
        sssValue *= scatterEff;
        sssValue *= 1.f / (distance_s_block1_sqare);
        sssValue *= 1.f / (distance_s_block2_sqare);
        for (int tofbinIdx = 0; tofbinIdx < __d_sssData.__TOFBinNum; tofbinIdx++) {
          auto [tofIntergral1, tofIntergral2] = calEmissionIntergralTOF(
              __d_sssData.__d_dsCrystalGeometry[block1FlatR].O, __d_sssData.__d_dsCrystalGeometry[block2FlatR].O,
              __d_sssData.__scatterPoints[scatterIndex].sssPosition, __d_sssData.__eMap, __d_sssData.__emapGrids,
              __d_sssData.__TOFBinWidth, tofbinIdx);
          float intergral1 = AttnIntergralBlock1 * tofIntergral1;
          float intergral2 = AttnIntergralBlock2 * tofIntergral2;
          // add gaussBlur
          int startCov = core::FMath<float>::max(0, tofbinIdx - int(__d_sssData.__gaussSize * 0.5));
          int endCov =
              core::FMath<float>::min(__d_sssData.__TOFBinNum - 1, tofbinIdx + int(__d_sssData.__gaussSize * 0.5));
          for (int conv = startCov; conv <= endCov; conv++) {
            sssValue += (intergral1 + intergral2) *
                        __d_sssData.__gaussBlurCoff[conv - tofbinIdx + int(__d_sssData.__gaussSize * 0.5)];
          }
        }
        for (int tofbinIdx = 0; tofbinIdx < __d_sssData.__TOFBinNum; tofbinIdx++)
          __d_sssData.__out_d_dsTOFBinSSSValue[dslorIdx * __d_sssData.__TOFBinNum + tofbinIdx] =
              sssValue * __d_sssData.__commonfactor;
      }
    });
  }
}
void d_sssTOFsumByTOFBin(
    float *__out_d_dsSumTOFBinSSSValue, float *__d_dsTOFBinSSSValue, std::size_t __dslorNum, int __tofbinNum) {
  tools::parallel_for_each_CUDA(__dslorNum * __tofbinNum, [=] __device__(std::size_t idx) {
    size_t lorIdx = idx / __tofbinNum;
    __out_d_dsSumTOFBinSSSValue[lorIdx] += __d_dsTOFBinSSSValue[idx];
  });
}
void d_sssTOFextendSlice(
    float *__d_dsSumTOFBinSSSValueExtend, float *__d_dsSumTOFBinSSSValue, const core::MichDefine __michDefine,
    const core::MichDefine __dsDefine) {
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto dsMichInfo = core::MichInfoHub::create(__dsDefine);

  auto ringNum = michInfo.getRingNum();
  auto sliceNum = michInfo.getSliceNum();
  auto dsRingNum = dsMichInfo.getRingNum();
  auto dsBinNum = dsMichInfo.getBinNum();
  auto dsViewNum = dsMichInfo.getViewNum();

  tools::parallel_for_each_CUDA(dsBinNum * dsViewNum * sliceNum, [=] __device__(std::size_t idx) {
    auto sliceIdx = idx / (dsBinNum * dsViewNum);
    auto dsbivi = idx % (dsBinNum * dsViewNum);
    auto [ring1, ring2] = core::mich::getRing1Ring2FromSlice(__michDefine.polygon, __michDefine.detector, sliceIdx);
    int dsRing1 = ring1 * (dsRingNum / ringNum);
    int dsRing2 = ring2 * (dsRingNum / ringNum);
    int dsSliceIdx = core::mich::getSliceFromRing1Ring2(__dsDefine.polygon, __dsDefine.detector, dsRing1, dsRing2);
    __d_dsSumTOFBinSSSValueExtend[idx] = __d_dsSumTOFBinSSSValue[dsbivi + dsSliceIdx * dsBinNum * dsViewNum];
  });
}
__PNI_CUDA_MACRO__ void chooseNearestBlockAndCalWeight(
    int &__blockNearest, float &__w, int __block, int __cryID, const core::CrystalGeom *__d_crystalGeometry,
    const core::CrystalGeom *__d_dsCrystalGeometry) {
  int block_left = __block - 1;
  int block_right = __block + 1;
  auto cry_block_left = __d_crystalGeometry[__cryID].O - __d_dsCrystalGeometry[block_left].O;
  auto cry_block_right = __d_crystalGeometry[__cryID].O - __d_dsCrystalGeometry[block_right].O;
  auto distance_cry_block_left = algorithms::l2(cry_block_left);
  auto distance_cry_block_right = algorithms::l2(cry_block_right);
  if (distance_cry_block_left > distance_cry_block_right) {
    __blockNearest = block_right;
    __w = distance_cry_block_right;
  } else {
    __blockNearest = block_left;
    __w = distance_cry_block_left;
  }
}
__PNI_CUDA_MACRO__ float get2DInterpolationUpsamplingValue(
    const float *__in_dsSumTOFBinSSSValueExtend, size_t __lorIdx, const core::MichDefine __michDefine,
    const core::MichDefine __dsmichDefine, const core::CrystalGeom *__crystalGeometry,
    const core::CrystalGeom *__dsCrystalGeometry) {
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto dsMichInfo = core::MichInfoHub::create(__dsmichDefine);
  auto coverter = core::IndexConverter::create(__michDefine);
  auto dsCoverter = core::IndexConverter::create(__dsmichDefine);
  auto [cry1R, cry2R] = coverter.getCrystalIDFromLORID(__lorIdx);
  auto cry1FlatR = core::mich::getFlatIdFromRectangleID(__michDefine.polygon, __michDefine.detector, cry1R);
  auto cry2FlatR = core::mich::getFlatIdFromRectangleID(__michDefine.polygon, __michDefine.detector, cry2R);
  // cal where cry1 cry2 in block,this is also the index in ds image
  int block1FlatR = cry1FlatR / (__michDefine.detector.crystalNumU * __michDefine.detector.crystalNumV);
  int block2FlatR = cry2FlatR / (__michDefine.detector.crystalNumU * __michDefine.detector.crystalNumV);
  // cal weight of block-cry which equals to distance from cry to block center
  float wA = algorithms::l2(__crystalGeometry[cry1FlatR].O - __dsCrystalGeometry[block1FlatR].O);
  float wB = algorithms::l2(__crystalGeometry[cry2FlatR].O - __dsCrystalGeometry[block2FlatR].O);
  // find nearest block and cal weight
  float wA_near, wB_near;
  int block1_nearFlatR, block2_nearFlatR;
  chooseNearestBlockAndCalWeight(block1_nearFlatR, wA_near, block1FlatR, cry1FlatR, __crystalGeometry,
                                 __dsCrystalGeometry);
  chooseNearestBlockAndCalWeight(block2_nearFlatR, wB_near, block2FlatR, cry2FlatR, __crystalGeometry,
                                 __dsCrystalGeometry);
  // cal A-B,A-B_near,A_near-B,A_near-B_near's dsLorID
  auto block1R = dsCoverter.getRectangleIdFromFlatId(block1FlatR);
  auto block2R = dsCoverter.getRectangleIdFromFlatId(block2FlatR);
  auto block1_nearR = dsCoverter.getRectangleIdFromFlatId(block1_nearFlatR);
  auto block2_nearR = dsCoverter.getRectangleIdFromFlatId(block2_nearFlatR);
  auto dsLORAB = core::mich::getLORIDFromRectangleID(__dsmichDefine.polygon, __dsmichDefine.detector, block1R, block2R);
  auto dsLORAB_near =
      core::mich::getLORIDFromRectangleID(__dsmichDefine.polygon, __dsmichDefine.detector, block1R, block2_nearR);
  auto dsLORA_nearB =
      core::mich::getLORIDFromRectangleID(__dsmichDefine.polygon, __dsmichDefine.detector, block1_nearR, block2R);
  auto dsLORA_nearB_near =
      core::mich::getLORIDFromRectangleID(__dsmichDefine.polygon, __dsmichDefine.detector, block1_nearR, block2_nearR);
  // because the slice has been extend,so here need to recal dslorId
  auto biviNum = michInfo.getBinNum() * michInfo.getViewNum();
  auto dsbiviNum = dsMichInfo.getBinNum() * dsMichInfo.getViewNum();
  auto sliceNow = __lorIdx / biviNum;
  dsLORAB = dsLORAB % dsbiviNum + sliceNow * dsbiviNum;
  dsLORAB_near = dsLORAB_near % dsbiviNum + sliceNow * dsbiviNum;
  dsLORA_nearB = dsLORA_nearB % dsbiviNum + sliceNow * dsbiviNum;
  dsLORA_nearB_near = dsLORA_nearB_near % dsbiviNum + sliceNow * dsbiviNum;
  // bilinear interpolation
  float value = __in_dsSumTOFBinSSSValueExtend[dsLORAB] * wA * wB +
                __in_dsSumTOFBinSSSValueExtend[dsLORAB_near] * wA * wB_near +
                __in_dsSumTOFBinSSSValueExtend[dsLORA_nearB] * wA_near * wB +
                __in_dsSumTOFBinSSSValueExtend[dsLORA_nearB_near] * wA_near * wB_near;
  float w_All = wA * wB + wA * wB_near + wA_near * wB + wA_near * wB_near;
  return value / w_All;
}
void d_2DIngerpolationUpSampling(
    float *__d_sssValue, const float *__d_dsSumTOFBinSSSValueExtend, const core::MichDefine __michDefine,
    const core::MichDefine __dsmichDefine, const core::CrystalGeom *__d_crystalGeometry,
    const core::CrystalGeom *__d_dsCrystalGeometry) {
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto lorNum = michInfo.getLORNum();
  tools::parallel_for_each_CUDA(lorNum, [=] __device__(size_t lorIdx) {
    __d_sssValue[lorIdx] =
        get2DInterpolationUpsamplingValue(__d_dsSumTOFBinSSSValueExtend, lorIdx, __michDefine, __dsmichDefine,
                                          __d_crystalGeometry, __d_dsCrystalGeometry);
  });
}
void tailFittingTOF(
    float *__out_d_dsfullSliceTOFBinSSSValue, const float *__d_sssValue, const float *__d_dsTOFBinSSSValue,
    const float *__d_prompt, MichNormalization &__norm, MichRandom &__rand, MichAttn &__attnCutBedCoff,
    const core::MichDefine __michDefine, const core::MichDefine __dsmichDefine, int __tofBinNum,
    int __minSectorDifference, double __tailFittingThreshold) {
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto ringNum = michInfo.getRingNum();
  auto binNum = michInfo.getBinNum();
  auto viewNum = michInfo.getViewNum();
  auto sliceNum = michInfo.getSliceNum();
  auto lorNumOneSlice = binNum * viewNum;
  auto binNumOutFOVOneSide =
      core::mich::calBinNumOutFOVOneSide(__michDefine.polygon, __michDefine.detector, __minSectorDifference);
  auto dsMichInfo = core::MichInfoHub::create(__dsmichDefine);
  auto dsRingNum = dsMichInfo.getRingNum();
  auto dsbinNum = dsMichInfo.getBinNum();
  auto dsviewNum = dsMichInfo.getViewNum();

  cuda_sync_ptr<std::size_t> d_lorBatch{"ScatterTailFittingTOF_lorBatch"};
  for (const auto [sliceBegin, sliceEnd] :
       tools::chunked_ranges_generator.by_max_size(0, sliceNum, 5 * 1024 * 1024 / lorNumOneSlice)) {
    const auto batchSize = (sliceEnd - sliceBegin) * lorNumOneSlice;
    d_lorBatch.reserve(batchSize);
    tools::parallel_for_each_CUDA(
        batchSize, [sliceBegin, lorNumOneSlice, d_lorBatch = d_lorBatch.data()] __device__(std::size_t idx) {
          d_lorBatch[idx] = idx + sliceBegin * lorNumOneSlice;
        });
    auto *attnFactors = __attnCutBedCoff.getDAttnFactorsBatch(d_lorBatch.cspan(batchSize));
    auto *normFactors = __norm.getDNormFactorsBatch(d_lorBatch.cspan(batchSize));
    auto *randFactors = __rand.getDRandomFactorsBatch(d_lorBatch.cspan(batchSize));

    tools::parallel_for_each_CUDA(sliceBegin, sliceEnd, [=] __device__(std::size_t sliceId) {
      algorithms::LinearFittingHelper<float, algorithms::LinearFittingType::LinearFitting_NoBias> __linearFitModel;
      auto [ring1, ring2] = core::mich::getRing1Ring2FromSlice(__michDefine.polygon, __michDefine.detector, sliceId);
      int dsRing1 = ring1 * dsRingNum / ringNum;
      int dsRing2 = ring2 * dsRingNum / ringNum;
      auto dsSliceIdx =
          core::mich::getSliceFromRing1Ring2(__dsmichDefine.polygon, __dsmichDefine.detector, dsRing1, dsRing2);

      for (size_t lorInsl = 0; lorInsl < lorNumOneSlice; ++lorInsl) {
        int binIndex = lorInsl % binNum;
        size_t lorIndex = sliceId * lorNumOneSlice + lorInsl;
        size_t indexOfFactors = (sliceId - sliceBegin) * lorNumOneSlice + lorInsl;
        if (attnFactors[indexOfFactors] >= __tailFittingThreshold && binIndex >= binNumOutFOVOneSide &&
            binIndex < binNum - binNumOutFOVOneSide) {
          if (normFactors[indexOfFactors] <= 0)
            continue;

          __linearFitModel.add(__d_sssValue[lorIndex], __d_prompt[lorIndex] - randFactors[indexOfFactors]);
        }
      }

      for (size_t bivi = 0; bivi < dsbinNum * dsviewNum; bivi++) {
        for (int tofbinIdx = 0; tofbinIdx < __tofBinNum; tofbinIdx++) {
          __out_d_dsfullSliceTOFBinSSSValue[bivi + sliceId * dsbinNum * dsviewNum +
                                            tofbinIdx * dsbinNum * dsviewNum * sliceNum] =
              __linearFitModel.predict(
                  __d_dsTOFBinSSSValue[(bivi + dsSliceIdx * dsbinNum * dsviewNum) * __tofBinNum + tofbinIdx]);
        }
      }
    });
  }
}

} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node {
void MichScatter_impl::sssPreGenerate() {
  // 1.generate sss points
  if (!m_scatterPointGrid)
    throw std::runtime_error("MichScatter: scatterPointGrid is not set before sssPreGenerate.");
  auto allPoints = impl::generateAllSSSPoints(*m_scatterPointGrid);
  auto sssPoints =
      impl::isSSSPoints(m_AttnCoff->h_getAttnMap(), m_AttnCoff->getMapGrids(), allPoints, m_scatterPointsThreshold);
  md_scatterPoints = make_cuda_sync_ptr_from_hcopy(sssPoints, "MichScatter_sssPoints_fromHost");
  m_scatterCount = sssPoints.size();
  // 2. generate scatterEffTable
  auto sssEffTable = impl::generateScatterEffTable(m_scatterEffTableEnergy, m_scatterEnergyWindow);
  md_scannerEffTable = make_cuda_sync_ptr_from_hcopy(sssEffTable, "MichScatter_scannerEffTable_fromHost");
  // 3. cal commonFactor
  m_commonFactor = impl::calSSSCommonFactors(m_AttnCoff->getMapGrids(),
                                             core::Vector<float, 3>::create(10.f, 10.f, 10.f), m_scatterEnergyWindow);
  // 4.cal sssAttnCoff & gaussBlur(if TOF)
  if (!m_TOFModel) {
    PNI_DEBUG("MichScatter: Start to generate SSS cofficients.\n");
    auto crystalGeo = m_michCrystal.dumpCrystalsRectangleLayout(); // notice:RectangleID
    auto d_crystalGeo = make_cuda_sync_ptr_from_hcopy(crystalGeo, "MichScatter_crystalGeo_fromHost");
    md_sssAttnCoff.reserve(allPoints.size() * crystalGeo.size());
    impl::d_generateSSS_AttnCoff(md_sssAttnCoff.get(), md_scatterPoints.get(), d_crystalGeo.get(),
                                 m_AttnCoff->d_getAttnMap(), m_AttnCoff->getMapGrids(), sssPoints.size(),
                                 crystalGeo.size());
  } else {
    PNI_DEBUG("MichScatter: Start to generate TOF SSS cofficients.\n");
    auto dsMichCrystal = MichCrystal(m_dsmichDefine);
    auto dsCrystalGeo = dsMichCrystal.dumpCrystalsRectangleLayout(); // notice:RectangleID,dsGeo
    auto d_dsCrystalGeo = make_cuda_sync_ptr_from_hcopy(dsCrystalGeo, "MichScatter_dsCrystalGeo_fromHost");
    md_sssAttnCoff.reserve(allPoints.size() * dsCrystalGeo.size());
    impl::d_generateSSS_AttnCoff(md_sssAttnCoff.get(), md_scatterPoints.get(), d_dsCrystalGeo.get(),
                                 m_AttnCoff->d_getAttnMap(), m_AttnCoff->getMapGrids(), sssPoints.size(),
                                 dsCrystalGeo.size());

    auto gaussBlur = impl::generateGaussianBlurKernel(m_sssTOFParams.m_systemTimeRes_ns, m_sssTOFParams.m_timeBinWidth);
    md_gaussBlurCoff = make_cuda_sync_ptr_from_hcopy(gaussBlur, "MichScatter_gaussBlur_fromHost");
    m_gaussSize = gaussBlur.size();
  }
  PNI_DEBUG("MichScatter: SSS cofficients generated.\n");
  m_preDataGenerated = true;
}
#define DEBUG_SSS_SAVE_FILE 0
cuda_sync_ptr<float> MichScatter_impl::d_generateScatterMich() {
  auto michInfo = core::MichInfoHub::create(m_michDefine);
  auto lorNum = michInfo.getLORNum();
  auto d_scatterValue = make_cuda_sync_ptr<float>(michInfo.getLORNum(), "MichScatter_scatterValueMich");
  if (!d_getEmissionMap()) {
    d_scatterValue.memset(0);
    PNI_DEBUG("MichScatter: No emission map, return zero scatter.\n");
    return d_scatterValue;
  }

  auto crystalGeo = m_michCrystal.dumpCrystalsRectangleLayout(); // notice:RectangleID
  auto d_crystalGeo = make_cuda_sync_ptr_from_hcopy(crystalGeo, "MichScatter_crystalGeo_fromHost");
  // cal emission intergral
  auto d_sssEmssionCoff = make_cuda_sync_ptr<float>(m_scatterCount * crystalGeo.size(), "MichScatter_sssEmssionCoff");
  PNI_DEBUG("MichScatter: Start to generate emission cofficients.\n");
  impl::d_generateSSS_EmissionCoff(d_sssEmssionCoff.get(), md_scatterPoints.get(), d_crystalGeo.get(),
                                   d_getEmissionMap(), m_emissionMapGrids, m_scatterCount, crystalGeo.size());
  //  do singleScatterSimulation
  sssDataView d_sssData{d_scatterValue.get(),  md_sssAttnCoff.get(),   d_sssEmssionCoff.get(), md_scannerEffTable.get(),
                        d_crystalGeo.get(),    md_scatterPoints.get(), m_michDefine,           m_scatterEffTableEnergy,
                        m_minSectorDifference, m_scatterCount,         m_commonFactor};

  impl::singleScatterSimulation(d_sssData);
  PNI_DEBUG("MichScatter: Single scatter simulation done.\n");
  // 同步设备，确保SSS计算完成
  // cudaError_t cudaStatus = cudaDeviceSynchronize();
  // 保存SSS中间结果（tail fitting之前）
  // std::vector<float> sss(lorNum);
  // cudaStatus = cudaMemcpy(sss.data(), d_scatterValue.get(), lorNum * sizeof(float), cudaMemcpyDeviceToHost);
  // std::cout << "saving sss mid file,size:" << sss.size() << std::endl;
  // 保存SSS中间结果到文件
  // std::string sss_mid_file =
  // "/home/ustc/pni_core/new/pni-standard-project/manual-test/Recon/tempFile_debug/SSSNotailFit.bin";
  // std::ofstream sssmidFile(sss_mid_file, std::ios::binary);
  // if (sssmidFile.is_open()) {
  //   sssmidFile.write(reinterpret_cast<const char *>(sss.data()), sss.size() * sizeof(float));
  //   sssmidFile.close();
  //   std::cout << "SSS intermediate result saved to: " << sss_mid_file << std::endl;
  // }
  //  normalization after sss
  impl::sssNormalization(d_scatterValue.get(), *m_norm, m_michDefine);
  PNI_DEBUG("MichScatter: Normalization done.\n");
#if DEBUG_SSS_SAVE_FILE
  // temptest,save sss data
  // 同步设备，确保SSS计算完成
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  // 保存SSS中间结果（tail fitting之前）
  std::vector<float> sss_before_tail(lorNum);
  cudaStatus = cudaMemcpy(sss_before_tail.data(), d_scatterValue.get(), lorNum * sizeof(float),
                          cudaMemcpyDeviceToHost); // 保存SSS中间结果到文件
  std::string sss_midnorm_file = "SSSNotailFit_Norm.bin";
  std::ofstream midFile(sss_midnorm_file, std::ios::binary);
  if (midFile.is_open()) {
    midFile.write(reinterpret_cast<const char *>(sss_before_tail.data()), sss_before_tail.size() * sizeof(float));
    midFile.close();
    std::cout << "SSS intermediate result saved to: " << sss_midnorm_file << std::endl;
  }
#endif
  //  tailFitting
  PNI_DEBUG("MichScatter: Start to do tail fitting.\n");
  impl::tailFitting(d_scatterValue.get(), m_lorGetor, *m_norm, *m_random, *m_AttnCoff, m_michDefine,
                    m_minSectorDifference, m_tailFittingThreshold);
#if DEBUG_SSS_SAVE_FILE
  std::vector<float> sss_after_tail(lorNum);
  cudaStatus = cudaMemcpy(sss_after_tail.data(), d_scatterValue.get(), lorNum * sizeof(float),
                          cudaMemcpyDeviceToHost); // 保存SSS中间结果到文件
  std::string sss_tail_file = "SSSWithTailFit.bin";
  std::ofstream tailFile(sss_tail_file, std::ios::binary);
  if (tailFile.is_open()) {
    tailFile.write(reinterpret_cast<const char *>(sss_after_tail.data()), sss_after_tail.size() * sizeof(float));
    tailFile.close();
    std::cout << "SSS with tail fitting result saved to: " << sss_tail_file << std::endl;
  }
#endif
  PNI_DEBUG("MichScatter: Tail fitting done.\n");
  return d_scatterValue;
}

cuda_sync_ptr<float> MichScatter_impl::d_generateScatterTableTOF() {
  auto michInfo = core::MichInfoHub::create(m_michDefine);
  auto crystalGeo = m_michCrystal.dumpCrystalsRectangleLayout(); // notice:RectangleID
  auto d_crystalGeo = make_cuda_sync_ptr_from_hcopy(crystalGeo, "MichScatter_crystalGeo_fromHost");
  auto dsMichInfo = core::MichInfoHub::create(m_dsmichDefine);
  auto dsMichCrystal = MichCrystal(m_dsmichDefine);
  auto dsCrystalGeo = dsMichCrystal.dumpCrystalsRectangleLayout(); // notice:RectangleID,dsGeo
  auto d_dsCrystalGeo = make_cuda_sync_ptr_from_hcopy(dsCrystalGeo, "MichScatter_dsCrystalGeo_fromHost");
  auto dslorNum = dsMichInfo.getLORNum();
  //  do singleScatterSimulationTOF
  auto __d_dsTOFBinSSSValue =
      make_cuda_sync_ptr<float>(dslorNum * m_sssTOFParams.m_tofBinNum, "MichScatter_dsTOFBinSSSValueMich");
  const float crystalArea =
      m_michDefine.detector.crystalSizeU * m_michDefine.detector.crystalSizeV * 0.01; // 单个晶体的面积position,cm2
  sssTOFDataView d_sssData{__d_dsTOFBinSSSValue.get(),
                           d_getEmissionMap(),
                           md_sssAttnCoff.get(),
                           md_scannerEffTable.get(),
                           md_gaussBlurCoff.get(),
                           d_dsCrystalGeo.get(),
                           md_scatterPoints.get(),
                           m_dsmichDefine,
                           m_scatterEffTableEnergy,
                           m_emissionMapGrids,
                           m_sssTOFParams.m_tofBinNum,
                           m_gaussSize,
                           m_scatterCount,
                           m_sssTOFParams.m_timeBinWidth,
                           crystalArea,
                           m_commonFactor};
  impl::singleScatterSimulationTOF(d_sssData);
  // 7.sum by TOF bin
  auto __d_dsSumTOFBinSSSValue = make_cuda_sync_ptr<float>(dslorNum, "MichScatter_dsSumTOFBinSSSValueMich");
  impl::d_sssTOFsumByTOFBin(__d_dsSumTOFBinSSSValue.get(), __d_dsTOFBinSSSValue.get(), dslorNum,
                            m_sssTOFParams.m_tofBinNum);
  // 8. dsSlice extend to slice
  auto ringNum = michInfo.getRingNum();
  auto dsRingNum = dsMichInfo.getRingNum();
  auto dsBinNum = dsMichInfo.getBinNum();
  auto dsViewNum = dsMichInfo.getViewNum();
  auto sliceNum = michInfo.getSliceNum();
  auto __d_dsSumTOFBinSSSValueExtend =
      make_cuda_sync_ptr<float>(sliceNum * dsBinNum * dsViewNum, "MichScatter_dsSumTOFBinSSSValueExtendMich");

  impl::d_sssTOFextendSlice(__d_dsSumTOFBinSSSValueExtend.get(), __d_dsSumTOFBinSSSValue.get(), m_michDefine,
                            m_dsmichDefine);
  // 9. 2D interpolation
  auto __d_sssValue = make_cuda_sync_ptr<float>(michInfo.getLORNum(), "MichScatter_sssValueMich");
  impl::d_2DIngerpolationUpSampling(__d_sssValue.get(), __d_dsSumTOFBinSSSValueExtend.get(), m_michDefine,
                                    m_dsmichDefine, d_crystalGeo.get(), d_dsCrystalGeo.get());
  // 10. normalization after sss
  impl::sssNormalization(__d_sssValue.get(), *m_norm, m_michDefine);

  // // 11. tailFitting
  auto __out_d_dsfullSliceTOFBinSSSValue = make_cuda_sync_ptr<float>(
      sliceNum * dsBinNum * dsViewNum * m_sssTOFParams.m_tofBinNum, "MichScatter_out_dsfullSliceTOFBinSSSValueMich");
  // impl::tailFittingTOF(__out_d_dsfullSliceTOFBinSSSValue.get(), __d_sssValue.get(), __d_dsTOFBinSSSValue.get(),
  //                      *m_lorGetor, *m_norm, *m_random, *m_AttnCoff, m_michDefine, m_dsmichDefine,
  //                      m_sssTOFParams.m_tofBinNum, minSectorDifference, m_tailFittingThreshold);
  return __out_d_dsfullSliceTOFBinSSSValue;
}

} // namespace openpni::experimental::node