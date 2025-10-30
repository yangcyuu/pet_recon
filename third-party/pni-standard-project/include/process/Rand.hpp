#pragma once

#include <ranges>

#include "../example/PolygonalSystem.hpp"
#include "../math/Geometry.hpp"
#include "Norm.hpp"

namespace openpni::process {
template <typename CalculationPresicion>
struct RandDataView {
  CalculationPresicion *__in_randMich;
  CalculationPresicion *__out_randMich;
  example::PolygonalSystem __polygon;
  basic::DetectorGeometry __detectorGeo;
  unsigned __minSectorDifference;
  unsigned __radialModuleNumS;

  static RandDataView defaultE180() {
    static auto E180__polygon = example::E180();
    static auto E180Detector = openpni::device::detectorUnchangable<device::bdm2::BDM2Runtime>();
    return {nullptr, nullptr, E180__polygon, E180Detector.geometry, 4, 6};
  }
};

// struct randProtocol
// {
//     example::__polygonalSystem __polygon;
//     basic::__detectorGeometry __detectorGeo;
//     unsigned minSectorDifference;
//     unsigned radialModuleNumS;

//     static randProtocol defaultE180()
//     {
//         static auto E180__polygon = example::E180();
//         static auto E180Detector = openpni::device::detectorUnchangable<device::bdm2::BDM2Runtime>();
//         return {E180__polygon, E180Detector.geometry, 4, 6};
//     }
// };

namespace rand {
struct _rand {
  template <typename CalculationPresicion>
  void operator()(
      RandDataView<CalculationPresicion> &__randDataView) {
    auto cryNum = example::polygon::getTotalCrystalNum(__randDataView.__polygon, __randDataView.__detectorGeo);
    auto crystalPerRing =
        example::polygon::getCrystalNumOneRing(__randDataView.__polygon, __randDataView.__detectorGeo);
    auto cryNumYInPanel =
        example::polygon::getCrystalNumYInPanel(__randDataView.__polygon, __randDataView.__detectorGeo);
    std::vector<float> cryFct(cryNum, 0);

    calCryFctByFanSum(cryFct, __randDataView.__in_randMich, __randDataView.__radialModuleNumS, __randDataView.__polygon,
                      __randDataView.__detectorGeo);

    for (auto LORIndex = 0;
         LORIndex < example::polygon::getLORNum(__randDataView.__polygon, __randDataView.__detectorGeo); LORIndex++) {
      const auto cryPairs = example::polygon::calRectangleFlatCrystalIDFromLORID(
          __randDataView.__polygon, __randDataView.__detectorGeo, LORIndex);
      int panel1 = cryPairs.x % crystalPerRing / cryNumYInPanel;
      int panel2 = cryPairs.y % crystalPerRing / cryNumYInPanel;
      example::polygon::isPairFar(__randDataView.__polygon, panel1, panel2, __randDataView.__minSectorDifference)
          ? __randDataView.__out_randMich[LORIndex] = cryFct[cryPairs.x] * cryFct[cryPairs.y]
          : __randDataView.__out_randMich[LORIndex] = 0;
    }
  }
};
} // namespace rand
} // namespace openpni::process
