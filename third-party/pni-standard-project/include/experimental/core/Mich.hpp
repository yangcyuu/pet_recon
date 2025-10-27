#pragma once

#include "Geometric.hpp"
namespace openpni::experimental::core {
struct UniformID {
  int detectorID; // 探测器ID
  int crystalU;   // U方向晶体ID
  int crystalV;   // V方向晶体ID
};
struct RectangleID {
  int ringID;   // 环ID
  int idInRing; // 环内晶体编号
};
} // namespace openpni::experimental::core

namespace openpni::experimental::core::mich {
__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumOneRing(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __polygon.detectorPerEdge * __polygon.edges * __detector.blockNumV * __detector.crystalNumV;
}
__PNI_CUDA_MACRO__ inline uint32_t getRingNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __polygon.detectorRings * __detector.blockNumU * __detector.crystalNumU;
}

__PNI_CUDA_MACRO__ inline uint32_t getBinNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) - 1;
}

__PNI_CUDA_MACRO__ inline uint32_t getViewNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) / 2;
}

__PNI_CUDA_MACRO__ inline uint32_t getSliceNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getRingNum(__polygon, __detector) * getRingNum(__polygon, __detector);
}

__PNI_CUDA_MACRO__ inline size_t getLORNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return size_t(getBinNum(__polygon, __detector)) * size_t(getViewNum(__polygon, __detector)) *
         size_t(getSliceNum(__polygon, __detector));
}
// 3.基于ringScanner的结构信息,即cry,block,module, panel等信息
__PNI_CUDA_MACRO__ inline uint32_t getPanelNum(
    const PolygonalSystem &__polygon) {
  return __polygon.edges;
}
__PNI_CUDA_MACRO__ inline uint32_t getBlockNumOneRing(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __polygon.detectorPerEdge * __polygon.edges * __detector.blockNumV;
}

__PNI_CUDA_MACRO__ inline uint32_t getTotalCrystalNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) * getRingNum(__polygon, __detector);
}

__PNI_CUDA_MACRO__ inline uint32_t getTotalBlockNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getBlockNumOneRing(__polygon, __detector) * __detector.blockNumU * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumZInModule(
    const DetectorGeom &__detector) // module在ringPETScanner中就是一个探测器
{
  return uint32_t(__detector.blockNumU) * uint32_t(__detector.crystalNumU);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumYInModule(
    const DetectorGeom &__detector) {
  return uint32_t(__detector.blockNumV) * uint32_t(__detector.crystalNumV);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumInModule(
    const DetectorGeom &__detector) {
  return getCrystalNumZInModule(__detector) * getCrystalNumYInModule(__detector);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumZInPanel(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getCrystalNumZInModule(__detector) * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumYInPanel(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return getCrystalNumYInModule(__detector) * __polygon.detectorPerEdge;
}

__PNI_CUDA_MACRO__ inline uint32_t getBlockRingNum(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __polygon.detectorRings * __detector.blockNumU;
}
__PNI_CUDA_MACRO__ inline uint32_t getBlockNumZInPanel(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __detector.blockNumU * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getBlockNumYInPanel(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector) {
  return __detector.blockNumV * __polygon.detectorPerEdge;
}
__PNI_CUDA_MACRO__ inline Vector<uint32_t, 3> getBinViewSliceFromLORID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, size_t LORID) {
  int bin = LORID % getBinNum(__polygon, __detector);
  int view = LORID / getBinNum(__polygon, __detector) % getViewNum(__polygon, __detector);
  int slice = LORID / (getBinNum(__polygon, __detector) * getViewNum(__polygon, __detector));
  return Vector<uint32_t, 3>::create(uint32_t(bin), uint32_t(view), uint32_t(slice));
}
__PNI_CUDA_MACRO__ inline size_t getLORIDFromBinViewSlice(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, uint32_t bin, uint32_t view, uint32_t slice) {
  return size_t(slice) * size_t(getBinNum(__polygon, __detector)) * size_t(getViewNum(__polygon, __detector)) +
         size_t(view) * size_t(getBinNum(__polygon, __detector)) + size_t(bin);
}

// 4.ID计算
__PNI_CUDA_MACRO__ inline Vector<uint32_t, 2> getCrystalIDInRingFromViewBin(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int view, int bin) {
  int crystalOneRing = getCrystalNumOneRing(__polygon, __detector);
  int cry2 = bin / 2 + 1;
  int cry1 = crystalOneRing + (1 - bin % 2) - cry2;

  cry2 = (cry2 + view) % crystalOneRing;
  cry1 = (cry1 + view) % crystalOneRing;
  return Vector<uint32_t, 2>::create(cry1, cry2);
}

__PNI_CUDA_MACRO__ inline Vector<uint32_t, 2> getRing1Ring2FromSlice(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int slice) {
  return Vector<uint32_t, 2>::create(slice / getRingNum(__polygon, __detector),
                                     slice % getRingNum(__polygon, __detector));
}
__PNI_CUDA_MACRO__ inline uint32_t getSliceFromRing1Ring2(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int ring1, int ring2) {
  return ring1 * getRingNum(__polygon, __detector) + ring2;
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumZInBlock(
    const DetectorGeom &__detector) {
  return __detector.crystalNumU;
}
__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumYInBlock(
    const DetectorGeom &__detector) {
  return __detector.crystalNumV;
}

__PNI_CUDA_MACRO__ inline core::Vector<RectangleID, 2> getRectangleCrystalIDFromLORID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, size_t LORID) {
  int crystalNumOneRing = getCrystalNumOneRing(__polygon, __detector);
  int binNum = getBinNum(__polygon, __detector);
  int viewNum = getViewNum(__polygon, __detector);
  int bin = LORID % binNum;
  int view = LORID / binNum % viewNum;
  int slice = LORID / (binNum * viewNum);
  auto cry12 = getCrystalIDInRingFromViewBin(__polygon, __detector, view, bin);
  auto ring12 = getRing1Ring2FromSlice(__polygon, __detector, slice);
  uint32_t crystalID1 = cry12[0] + ring12[0] * crystalNumOneRing;
  uint32_t crystalID2 = cry12[1] + ring12[1] * crystalNumOneRing;
  RectangleID rid1, rid2;
  rid1.ringID = crystalID1 / crystalNumOneRing;
  rid1.idInRing = crystalID1 % crystalNumOneRing;
  rid2.ringID = crystalID2 / crystalNumOneRing;
  rid2.idInRing = crystalID2 % crystalNumOneRing;
  if (crystalID1 > crystalID2)
    return core::Vector<RectangleID, 2>::create(rid1, rid2);
  else
    return core::Vector<RectangleID, 2>::create(rid2, rid1);
}

__PNI_CUDA_MACRO__ inline size_t calLORIDFromRingAndCrystalInRing(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int ring1, int cry1, int ring2, int cry2) {
  if (cry1 == cry2)
    return size_t(-1);
  size_t crystalOneRing = getCrystalNumOneRing(__polygon, __detector);
  size_t view = (cry1 + cry2) % crystalOneRing / 2;
  cry1 -= view;
  cry2 -= view;
  if (cry1 <= 0)
    cry1 += crystalOneRing;
  if (cry2 <= 0)
    cry2 += crystalOneRing;
  int ring1real, ring2real, cry1real, cry2real;
  if (cry1 > cry2) {
    cry1real = cry1;
    cry2real = cry2;
    ring1real = ring1;
    ring2real = ring2;
  } else {
    cry1real = cry2;
    cry2real = cry1;
    ring1real = ring2;
    ring2real = ring1;
  }
  size_t bin = (crystalOneRing - 1) - (cry1real - cry2real);
  size_t slice = ring1real * getRingNum(__polygon, __detector) + ring2real;
  size_t LORID = slice * (crystalOneRing - 1) * (crystalOneRing / 2) + view * (crystalOneRing - 1) + bin;
  return LORID;
}

__PNI_CUDA_MACRO__ inline size_t getLORIDFromRectangleID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, RectangleID crystalID1, RectangleID crystalID2) {
  return calLORIDFromRingAndCrystalInRing(__polygon, __detector, crystalID1.ringID, crystalID1.idInRing,
                                          crystalID2.ringID, crystalID2.idInRing);
}

__PNI_CUDA_MACRO__ inline uint32_t calBlockInRingFromCrystalId(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, RectangleID crystalID) {
  return crystalID.idInRing / __detector.crystalNumV;
}
__PNI_CUDA_MACRO__ inline bool isPairFar(
    const PolygonalSystem &__polygon, int panel1, int panel2, int minSectorDifference) {
  return abs(panel1 - panel2) >= minSectorDifference &&
         getPanelNum(__polygon) - abs(panel1 - panel2) >= minSectorDifference;
}
__PNI_CUDA_MACRO__ inline UniformID getUniformIdFromFlatId(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int flatId) {
  UniformID uid;
  uid.detectorID = flatId / getCrystalNumInModule(__detector);
  int cryIdInModule = flatId % getCrystalNumInModule(__detector);
  uid.crystalU = cryIdInModule % getCrystalNumZInModule(__detector);
  uid.crystalV = cryIdInModule / getCrystalNumZInModule(__detector);
  return uid;
}

__PNI_CUDA_MACRO__ inline RectangleID getRectangleIDFromUniformID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, UniformID __uniformID) {
  __uniformID.crystalV = getCrystalNumYInModule(__detector) - 1 - __uniformID.crystalV; // 由于坐标系问题这里得翻一下
  const auto bdmIDInRing = __uniformID.detectorID % (__polygon.edges * __polygon.detectorPerEdge);
  const auto bdmRingId = __uniformID.detectorID / (__polygon.edges * __polygon.detectorPerEdge);
  const auto crystalRingID = __uniformID.crystalU + getCrystalNumZInModule(__detector) * bdmRingId;
  const auto crystalIdInRing = __uniformID.crystalV + getCrystalNumYInModule(__detector) * bdmIDInRing;
  RectangleID result;
  result.idInRing = crystalIdInRing;
  result.ringID = crystalRingID;
  return result;
}
__PNI_CUDA_MACRO__ inline int getFlatIdFromRectangleID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, RectangleID __rectangleID) {
  return __rectangleID.ringID * getCrystalNumOneRing(__polygon, __detector) + __rectangleID.idInRing;
}
__PNI_CUDA_MACRO__ inline int getFlatIdFromUniformID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, UniformID __uniformID) {
  return __uniformID.detectorID * getCrystalNumInModule(__detector) + __uniformID.crystalU +
         __uniformID.crystalV * getCrystalNumZInModule(__detector);
}

__PNI_CUDA_MACRO__ inline int calBinNumOutFOVOneSide(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int minSectorDifference) {
  int crystalNumYInPanel = getCrystalNumYInPanel(__polygon, __detector);
  if (minSectorDifference == 0)
    return 0;
  return crystalNumYInPanel * minSectorDifference - 1;
}
__PNI_CUDA_MACRO__ inline RectangleID getRectangleIdFromFlatId(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int flatId) {
  RectangleID rid;
  rid.ringID = flatId / getCrystalNumOneRing(__polygon, __detector);
  rid.idInRing = flatId % getCrystalNumOneRing(__polygon, __detector);
  return rid;
}

__PNI_CUDA_MACRO__ inline UniformID getUniformIDFromRectangleID(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, RectangleID __rectangleID) {
  const auto bdmRingID = __rectangleID.ringID / getCrystalNumZInModule(__detector);
  const auto bdmIDInRing = __rectangleID.idInRing / getCrystalNumYInModule(__detector);

  const auto uInModule = __rectangleID.ringID % getCrystalNumZInModule(__detector);
  const auto vInModule = getCrystalNumYInModule(__detector) - 1 -
                         __rectangleID.idInRing % getCrystalNumYInModule(__detector); // 由于坐标系问题这里得翻一下
  const auto bdmID = bdmIDInRing + bdmRingID * (__polygon.edges * __polygon.detectorPerEdge);
  UniformID uid;
  uid.detectorID = bdmID;
  uid.crystalU = uInModule;
  uid.crystalV = vInModule;
  return uid;
}

__PNI_CUDA_MACRO__ inline int calViewNumInSubset(
    const PolygonalSystem &__polygon, const DetectorGeom &__detector, int subsetNum, int subsetIndex) {
  std::size_t viewNum = getViewNum(__polygon, __detector);
  return (viewNum + subsetNum - 1 - subsetIndex) / subsetNum;
}

// __PNI_CUDA_MACRO__ inline std::size_t calLORIDFromSubsetId(
//     const PolygonalSystem &__polygon, const DetectorGeom &__detector, int subsetId, int subsetNum, int binCut,
//     std::size_t id) // only can be used for view_subset
// {
//   std::size_t binNum = getBinNum(__polygon, __detector);
//   std::size_t viewNum = getViewNum(__polygon, __detector);
//   std::size_t sliceNum = getSliceNum(__polygon, __detector);
//   std::size_t viewNumInSubset = calViewNumInSubset(__polygon, __detector, subsetNum, subsetId);

//   std::size_t binIndex = (id % (binNum - 2 * binCut)) + binCut;
//   std::size_t sliceIndex = (id / (binNum - 2 * binCut)) % sliceNum;
//   std::size_t viewIndexInSubset = id / ((binNum - 2 * binCut) * sliceNum);

//   std::size_t result = sliceIndex * binNum * viewNum + (viewIndexInSubset * subsetNum + subsetId) * binNum +
//   binIndex; return result;
// }

// __PNI_CUDA_MACRO__ inline basic::Vec2<uint32_t> calCrystalIDFromSubsetId(
//     const PolygonalSystem &__polygon, const Detector &__detector, int subsetId, int subsetNum, int binCut,
//     std::size_t id) {
//   return calCrystalIDFromLORID(__polygon, __detector,
//                                calLORIDFromSubsetId(__polygon, __detector, subsetId, subsetNum, binCut, id));
// }

// __PNI_CUDA_MACRO__ inline std::size_t calSubsetSizeByView(
//     const PolygonalSystem &__polygon, const Detector &__detector, int __subsetNum, int __subsetIndex, int __binCut) {
//   std::size_t binNum = getBinNum(__polygon, __detector);
//   std::size_t viewNum = getViewNum(__polygon, __detector);
//   std::size_t sliceNum = getSliceNum(__polygon, __detector);

//   std::size_t result = 0;
//   for (std::size_t viewIndex = __subsetIndex; viewIndex < viewNum; viewIndex += __subsetNum)
//     result++;

//   result *= binNum - 2 * __binCut;
//   result *= sliceNum;
//   return result;
// }

} // namespace openpni::experimental::core::mich
#include <ranges>

namespace openpni::experimental::core {

struct IndexConverter;
struct RangeGenerator;
struct MichInfoHub;
struct MichDefine {
  PolygonalSystem polygon; // 多边形系统
  DetectorGeom detector;   // 探测器几何结构
};

struct IndexConverter {
  PolygonalSystem polygon;
  DetectorGeom detector;
  __PNI_CUDA_MACRO__ IndexConverter static create(
      const MichDefine &mich) {
    IndexConverter result;
    result.polygon = mich.polygon;
    result.detector = mich.detector;
    return result;
  }
  __PNI_CUDA_MACRO__ Vector<uint32_t, 3> getBVSFromLOR(
      size_t LORid) const {
    return mich::getBinViewSliceFromLORID(polygon, detector, LORid);
  }
  __PNI_CUDA_MACRO__ std::size_t getLORFromBVS(
      uint32_t bin, uint32_t view, uint32_t slice) const {
    return mich::getLORIDFromBinViewSlice(polygon, detector, bin, view, slice);
  }
  __PNI_CUDA_MACRO__ Vector<uint32_t, 2> getCrystalInRingFromBinView(
      uint32_t bin, uint32_t view) const {
    return mich::getCrystalIDInRingFromViewBin(polygon, detector, int(view), int(bin));
  }
  __PNI_CUDA_MACRO__ Vector<uint32_t, 2> getRing1Ring2FromSlice(
      uint32_t slice) const {
    return mich::getRing1Ring2FromSlice(polygon, detector, int(slice));
  }
  __PNI_CUDA_MACRO__ uint32_t getSliceFromRing1Ring2(
      uint32_t ring1, uint32_t ring2) const {
    return mich::getSliceFromRing1Ring2(polygon, detector, ring1, ring2);
  }
  __PNI_CUDA_MACRO__ core::Vector<RectangleID, 2> getCrystalIDFromLORID(
      std::size_t LORid) const {
    return mich::getRectangleCrystalIDFromLORID(polygon, detector, LORid);
  }
  __PNI_CUDA_MACRO__ std::size_t getLORIDFromRectangleID(
      RectangleID id1, RectangleID id2) const {
    return mich::getLORIDFromRectangleID(polygon, detector, id1, id2);
  }
  __PNI_CUDA_MACRO__ int getBlockIdInRingFromCrystalId(
      RectangleID crystal) const {
    return mich::calBlockInRingFromCrystalId(polygon, detector, crystal);
  }
  __PNI_CUDA_MACRO__ bool isGoodPairMinSector(
      int panel1, int panel2, int minSectorDifference) const {
    return mich::isPairFar(polygon, panel1, panel2, minSectorDifference);
  }
  __PNI_CUDA_MACRO__ RectangleID getRectangleIDFromUniformID(
      UniformID uid) const {
    return mich::getRectangleIDFromUniformID(polygon, detector, uid);
  }
  __PNI_CUDA_MACRO__ UniformID getUniformIDFromRectangleID(
      RectangleID rid) const {
    return mich::getUniformIDFromRectangleID(polygon, detector, rid);
  }
  __PNI_CUDA_MACRO__
  int getFlatIdFromRectangleId(
      RectangleID rId) const {
    return mich::getFlatIdFromRectangleID(polygon, detector, rId);
  }
  __PNI_CUDA_MACRO__
  int getFlatIdFromUniformID(
      UniformID uId) const {
    return mich::getFlatIdFromUniformID(polygon, detector, uId);
  }
  __PNI_CUDA_MACRO__
  UniformID getUniformIdFromFlatId(
      int flatId) const {
    return mich::getUniformIdFromFlatId(polygon, detector, flatId);
  }
  __PNI_CUDA_MACRO__
  RectangleID getRectangleIdFromFlatId(
      int flatId) const {
    return mich::getRectangleIdFromFlatId(polygon, detector, flatId);
  }
};
struct RangeGenerator {
  PolygonalSystem polygon;
  DetectorGeom detector;
  __PNI_CUDA_HOST_ONLY__ RangeGenerator static create(
      const MichDefine &mich) {
    // #ifdef __CUDA_ARCH__
    // #error "DO NOT USE THIS IN CUDA KERNEL"
    // #endif
    RangeGenerator result;
    result.polygon = mich.polygon;
    result.detector = mich.detector;
    return result;
  }
  __PNI_CUDA_HOST_ONLY__ auto allLORs() const {
    return std::views::iota(size_t(0), mich::getLORNum(polygon, detector));
  }
  __PNI_CUDA_HOST_ONLY__ auto allBins() const {
    return std::views::iota(uint32_t(0), mich::getBinNum(polygon, detector));
  }
  __PNI_CUDA_HOST_ONLY__ auto allViews() const {
    return std::views::iota(uint32_t(0), mich::getViewNum(polygon, detector));
  }
  __PNI_CUDA_HOST_ONLY__ auto allRings() const {
    return std::views::iota(uint32_t(0), mich::getRingNum(polygon, detector));
  }
  __PNI_CUDA_HOST_ONLY__ auto allSlices() const { // Exmaple: auto [slice1, slice2, sliceID] : ...
    const uint32_t sliceNum = mich::getSliceNum(polygon, detector);
    return std::views::iota(uint32_t(0), sliceNum) |
           std::views::transform([sliceNum](uint32_t i) { return std::make_tuple(i % sliceNum, i / sliceNum, i); });
  };
  __PNI_CUDA_HOST_ONLY__ auto allFlatCrystalID() const {
    return std::views::iota(uint32_t(0), mich::getTotalCrystalNum(polygon, detector));
  }
  __PNI_CUDA_HOST_ONLY__ auto lorsForGivenBin(
      int bin) const {
    return std::views::iota(0u, mich::getViewNum(polygon, detector) * mich::getSliceNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             return size_t(index) * size_t(mich::getBinNum(polygon, detector)) + size_t(bin);
           });
  }
  __PNI_CUDA_HOST_ONLY__ auto lorsForGivenView(
      int view) const {
    return std::views::iota(0u, mich::getBinNum(polygon, detector) * mich::getSliceNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             const auto binIndex = index % mich::getBinNum(polygon, detector);
             const auto sliceIndex = index / mich::getBinNum(polygon, detector);
             return size_t(sliceIndex) * size_t(mich::getBinNum(polygon, detector)) *
                        size_t(mich::getViewNum(polygon, detector)) +
                    size_t(view) * size_t(mich::getBinNum(polygon, detector)) + size_t(binIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto lorsForGivenSlice(
      int slice) const {
    return std::views::iota(0u, mich::getBinNum(polygon, detector) * mich::getViewNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             const auto binIndex = index % mich::getBinNum(polygon, detector);
             const auto viewIndex = index / mich::getBinNum(polygon, detector);
             return size_t(slice) * size_t(mich::getBinNum(polygon, detector)) *
                        size_t(mich::getViewNum(polygon, detector)) +
                    size_t(viewIndex) * size_t(mich::getBinNum(polygon, detector)) + size_t(binIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto lorsForGivenBinView(
      int bin, int view) const {
    return std::views::iota(0u, mich::getSliceNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             return size_t(index) * size_t(mich::getBinNum(polygon, detector)) *
                        size_t(mich::getViewNum(polygon, detector)) +
                    size_t(view) * size_t(mich::getBinNum(polygon, detector)) + size_t(bin);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto lorsForGivenBinSlice(
      int bin, int slice) const {
    return std::views::iota(0u, mich::getViewNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             return size_t(slice) * size_t(mich::getBinNum(polygon, detector)) *
                        size_t(mich::getViewNum(polygon, detector)) +
                    size_t(index) * size_t(mich::getBinNum(polygon, detector)) + size_t(bin);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto lorsForGivenViewSlice(
      int view, int slice) const {
    return std::views::iota(0u, mich::getBinNum(polygon, detector)) |
           std::views::transform([=, polygon = polygon, detector = detector](auto index) noexcept {
             return size_t(slice) * size_t(mich::getBinNum(polygon, detector)) *
                        size_t(mich::getViewNum(polygon, detector)) +
                    size_t(view) * size_t(mich::getBinNum(polygon, detector)) + size_t(index);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allBinViewSlices() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORAndBinViewSlices() const {
    return allBinViewSlices() | std::views::transform([=, polygon = polygon, detector = detector](auto bvs) noexcept {
             const auto &b{bvs[0]}, &v{bvs[1]}, &s{bvs[2]};
             const auto lorIndex = b + v * mich::getBinNum(polygon, detector) +
                                   s * mich::getBinNum(polygon, detector) * mich::getViewNum(polygon, detector);
             return std::make_tuple(lorIndex, b, v, s);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenBin(
      int bin) const {
    return lorsForGivenBin(bin) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenView(
      int view) const {
    return lorsForGivenView(view) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenSlice(
      int slice) const {
    return lorsForGivenSlice(slice) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenBinView(
      int bin, int view) const {
    return lorsForGivenBinView(bin, view) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenBinSlice(
      int bin, int slice) const {
    return lorsForGivenBinSlice(bin, slice) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto bvsForGivenViewSlice(
      int view, int slice) const {
    return lorsForGivenViewSlice(view, slice) |
           std::views::transform([=, polygon = polygon, detector = detector](auto lorIndex) noexcept {
             return mich::getBinViewSliceFromLORID(polygon, detector, lorIndex);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allFlatCrystals() const { return std::views::iota(0u, mich::getTotalCrystalNum(polygon, detector)); }
  __PNI_CUDA_HOST_ONLY__
  auto allUniformCrystals() const {
    return allFlatCrystals() | std::views::transform([=, polygon = polygon, detector = detector](auto cryID) noexcept {
             return mich::getUniformIdFromFlatId(polygon, detector, cryID);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allRectangleCrystals() const {
    return allFlatCrystals() | std::views::transform([=, polygon = polygon, detector = detector](auto cryID) noexcept {
             return mich::getRectangleIdFromFlatId(polygon, detector, cryID);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allFlatAndUniformCrystals() const {
    return allFlatCrystals() | std::views::transform([=, polygon = polygon, detector = detector](auto cryID) noexcept {
             const auto uid = mich::getUniformIdFromFlatId(polygon, detector, cryID);
             return std::make_tuple(cryID, uid.detectorID, uid.crystalU, uid.crystalV);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allFlatAndRectangleCrystals() const {
    return allFlatCrystals() | std::views::transform([=, polygon = polygon, detector = detector](auto cryID) noexcept {
             const auto rid = mich::getRectangleIdFromFlatId(polygon, detector, cryID);
             return std::make_tuple(cryID, rid.ringID, rid.idInRing);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allFlatAndUniformAndRectangleCrystals() const {
    return allFlatCrystals() | std::views::transform([=, polygon = polygon, detector = detector](auto cryID) noexcept {
             const auto uid = mich::getUniformIdFromFlatId(polygon, detector, cryID);
             const auto rid = mich::getRectangleIdFromFlatId(polygon, detector, cryID);
             return std::make_tuple(cryID, uid.detectorID, uid.crystalU, uid.crystalV, rid.ringID, rid.idInRing);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORRectangleCrystals() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorID) noexcept {
             return mich::getRectangleCrystalIDFromLORID(polygon, detector, lorID);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORAndRectangleCrystals() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorID) noexcept {
             const auto crystalPair = mich::getRectangleCrystalIDFromLORID(polygon, detector, lorID);
             return std::make_tuple(lorID, crystalPair[0], crystalPair[1]);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORAndRectangleCrystalsFlat() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorID) noexcept {
             const auto crystalPair = mich::getRectangleCrystalIDFromLORID(polygon, detector, lorID);
             const auto flatId1 = mich::getFlatIdFromRectangleID(polygon, detector, crystalPair[0]);
             const auto flatId2 = mich::getFlatIdFromRectangleID(polygon, detector, crystalPair[1]);
             return std::make_tuple(lorID, flatId1, crystalPair[0], flatId2, crystalPair[1]);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORAndBinViewSlicesAndCrystals() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorID) noexcept {
             const auto crystalPair = mich::getRectangleCrystalIDFromLORID(polygon, detector, lorID);
             const auto bvs = mich::getBinViewSliceFromLORID(polygon, detector, lorID);
             const auto b{bvs[0]}, v{bvs[1]}, s{bvs[2]};
             return std::make_tuple(lorID, b, v, s, crystalPair[0], crystalPair[1]);
           });
  }
  __PNI_CUDA_HOST_ONLY__
  auto allLORAndBinViewSlicesAndFlatCrystals() const {
    return allLORs() | std::views::transform([=, polygon = polygon, detector = detector](auto lorID) noexcept {
             const auto crystalPair = mich::getRectangleCrystalIDFromLORID(polygon, detector, lorID);
             const auto [b, v, s] = mich::getBinViewSliceFromLORID(polygon, detector, lorID);
             auto [flatId1, flatId2] =
                 algorithms::apply(crystalPair, [=, polygon = polygon, detector = detector](auto rid) {
                   return mich::getFlatIdFromRectangleID(polygon, detector, rid);
                 });
             return std::make_tuple(lorID, b, v, s, flatId1, flatId2);
           });
  }
};
struct MichInfoHub {
  PolygonalSystem polygon;
  DetectorGeom detector;
  __PNI_CUDA_MACRO__ MichInfoHub static create(
      MichDefine mich) {
    MichInfoHub result;
    result.polygon = mich.polygon;
    result.detector = mich.detector;
    return result;
  }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumOneRing() const { return mich::getCrystalNumOneRing(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getRingNum() const { return mich::getRingNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getBinNum() const { return mich::getBinNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getViewNum() const { return mich::getViewNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getSliceNum() const { return mich::getSliceNum(polygon, detector); }
  __PNI_CUDA_MACRO__ std::size_t getLORNum() const { return mich::getLORNum(polygon, detector); }
  __PNI_CUDA_MACRO__ std::size_t getMichSize() const { return getLORNum(); }
  __PNI_CUDA_MACRO__ uint32_t getPanelNum() const { return mich::getPanelNum(polygon); }
  __PNI_CUDA_MACRO__ uint32_t getBlockNumOneRing() const { return mich::getBlockNumOneRing(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getTotalCrystalNum() const { return mich::getTotalCrystalNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getTotalBlockNum() const { return mich::getTotalBlockNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumZInModule() const { return mich::getCrystalNumZInModule(detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumYInModule() const { return mich::getCrystalNumYInModule(detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumYInBlock() const { return mich::getCrystalNumYInBlock(detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumInModule() const { return mich::getCrystalNumInModule(detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumZInPanel() const { return mich::getCrystalNumZInPanel(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumZInBlock() const { return mich::getCrystalNumZInBlock(detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumYInPanel() const { return mich::getCrystalNumYInPanel(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getCrystalNumInPanel() const { return getCrystalNumZInPanel() * getCrystalNumYInPanel(); }
  __PNI_CUDA_MACRO__ uint32_t getBlockRingNum() const { return mich::getBlockRingNum(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getBlockNumZInPanel() const { return mich::getBlockNumZInPanel(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getBlockNumYInPanel() const { return mich::getBlockNumYInPanel(polygon, detector); }
  __PNI_CUDA_MACRO__ uint32_t getBlockNumInPanel() const { return getBlockNumZInPanel() * getBlockNumYInPanel(); }
};

struct MichStandardEvent {
  RectangleID crystal1;
  RectangleID crystal2;
  core::CrystalGeom geo1;
  core::CrystalGeom geo2;
  int16_t tof = 0;                     // in unit of ps, defined as time1 - time2
  float value = 1;                     // if listmode, this is 1.0; if mich, this is the value of the bin
  int16_t cry1_tof_deviation = 0x7fff; // TOF信息的标准差，单位皮秒，没有TOF信息(= inf)
  int16_t cry1_tof_mean = 0;           // TOF信息的平均值，单位皮秒，没有TOF信息(= 0)
  int16_t cry2_tof_deviation = 0x7fff; // TOF信息的标准差，单位皮秒，没有TOF信息(= inf)
  int16_t cry2_tof_mean = 0;           // TOF信息的平均值，单位皮秒，没有TOF信息(= 0)
};

} // namespace openpni::experimental::core