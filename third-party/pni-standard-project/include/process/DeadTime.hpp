#pragma once

#include <ranges>

#include "../example/PolygonalSystem.hpp"
#include "../math/Geometry.hpp"

// temp
#include "Scatter.cuh" // only for use linearFit
//
namespace openpni::process {
template <typename CalculationPresicion>
struct _deadTimeDataView {
  CalculationPresicion *__out_CFDTTable;
  CalculationPresicion *__dMichP;
  CalculationPresicion *__dMichR;
  CalculationPresicion *__scanTime;
  CalculationPresicion *__acqActivity;
  example::PolygonalSystem __polygon;
  basic::DetectorGeometry __detectorGeo;
  double __scanTime_;
  int __randomRateMin;
  int __acquisitionNum;
  int __linearFitPointNum;
  static _deadTimeDataView<CalculationPresicion> defaultE180() {
    static auto E180Polygon = example::E180();
    static auto E180Detector = openpni::device::detectorUnchangable<device::bdm2::BDM2Runtime>();
    return {.__polygon = E180Polygon,
            .__detectorGeo = E180Detector.geometry,
            .__scanTime_ = 1200.09,
            .__randomRateMin = 100,
            .__acquisitionNum = 40,
            .__linearFitPointNum = 7};
  }
};

// struct deadTimeProtocol
// {
//   example::PolygonalSystem &polygon;
//   basic::DetectorGeometry &detectorGeo;
//   double scanTime;
//   int randomRateMin;
//   int acquisitionNum;
//   static deadTimeProtocol defaultE180()
//   {
//     static auto E180Polygon = example::E180();
//     static auto E180Detector = openpni::device::detectorUnchangable<device::bdm2::BDM2Runtime>();
//     return {E180Polygon, E180Detector.geometry, 100, 40, 1};
//   }
// };

namespace deadTime {
struct _MichDownSampling {
  template <typename CalculationPresicion>
  struct _downSamplingByBlock {
    CalculationPresicion *__out_downSampleMich;
    const CalculationPresicion *__in_Mich;
    example::PolygonalSystem __polygon;
    basic::DetectorGeometry __detectorGeo;

    void operator()() {
      auto lorcator = example::polygon::Locator(__polygon, __detectorGeo);
      auto blockRingNum = example::polygon::getBlockRingNum(__polygon, __detectorGeo);

      for (auto [lor, bi, vi, sl] : lorcator.allLORAndBinViewSlices()) {
        auto [ring1, ring2] = example::polygon::calRing1Ring2FromSlice(__polygon, __detectorGeo, sl);
        int blockRing1 = ring1 / __detectorGeo.crystalNumU;
        int blockRing2 = ring2 / __detectorGeo.crystalNumU;
        int dsSliceID = blockRing1 * blockRingNum + blockRing2;
        __out_downSampleMich[dsSliceID] += __in_Mich[lor];
      }
    }
  };
};

template <typename CalculationPresicion>
struct _calReallyRate {
  CalculationPresicion *__out_actualRate;
  CalculationPresicion *__dMichP;
  CalculationPresicion *__dMichR;
  CalculationPresicion *__scanTime;
  int __acquisitionNum;

  void operator()(
      std::size_t idx) {
    std::size_t sliceNow = idx / __acquisitionNum;
    int acqIdx = idx % __acquisitionNum;
    auto rate_p = __dMichP[sliceNow] / __scanTime[acqIdx];
    auto rate_r = __dMichR[sliceNow] / __scanTime[acqIdx];
    __out_actualRate[idx] = rate_p - rate_r;
  }
};

template <typename CalculationPresicion, typename LinearFitModel>
struct _calCFDT {
  CalculationPresicion *__out_CFDT;
  CalculationPresicion *__actualRt;
  CalculationPresicion *__acqActivity;
  int __acquisitionNum;
  int __linearFitPointNum;

  void operator()(
      std::size_t dsslIdx) {
    LinearFitModel __linearFit;
    // linearFit to cal CFDT
    for (auto p : std::views::iota(0, __linearFitPointNum)) {
      __linearFit.add(__acqActivity[__acquisitionNum - p - 1],
                      __actualRt[dsslIdx * __acquisitionNum + __acquisitionNum - p - 1]);
    }
    // cal CFDT by LinearFitRate / actualRate
    for (auto acq : std::views::iota(0, __acquisitionNum)) {
      double idealRt = __linearFit.predict(__acqActivity[acq]); //
      __out_CFDT[dsslIdx * __acquisitionNum + acq] = idealRt / __actualRt[dsslIdx * __acquisitionNum + acq];
    }
  }
};

template <typename CalculationPresicion, typename LinearFitModel>
struct _generateCFDTTable {
  void operator()(
      _deadTimeDataView<CalculationPresicion> &__dtDataview) // CTDTTable_size:2 * m_acquisitionNum * m_dsSlice
  {
    auto blockRingNum = example::polygon::getBlockRingNum(__dtDataview.__polygon, __dtDataview.__detectorGeo);
    auto dsSliceNum = blockRingNum * blockRingNum;
    std::vector<CalculationPresicion> actualRt(dsSliceNum * __dtDataview.__acquisitionNum);
    std::vector<CalculationPresicion> CFDT(dsSliceNum * __dtDataview.acquisitionNum);

    // cal really rate
    for_each(dsSliceNum * __dtDataview.__acquisitionNum,
             _calReallyRate{actualRt.data(), __dtDataview.__dMichP, __dtDataview.__dMichR, __dtDataview.__scanTime});
    // cal  CFDT
    for_each(dsSliceNum, _calCFDT{CFDT.data(), actualRt.data(), __dtDataview.__acqActivity,
                                  __dtDataview.__acquisitionNum, __dtDataview.__linearFitPointNum});
    // generate CFDT table
    for_each(dsSliceNum * __dtDataview.__acquisitionNum, [&](size_t idx) {
      __dtDataview.__out_CFDTTable[idx] = actualRt[idx];
      __dtDataview.__out_CFDTTable[idx + dsSliceNum * __dtDataview.__acquisitionNum] = CFDT[idx];
    });
  }
};
} // namespace deadTime
} // namespace openpni::process
