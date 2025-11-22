#pragma once
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <ranges>
#include <tl/expected.hpp>
#include <vector>

#include "../basic/DataView.hpp"
#include "../basic/Model.hpp"
#include "../basic/Point.hpp"
#include "../detector/Detectors.hpp"
#include "../detector/EmptyDetector.hpp"
#include "../math/Geometry.hpp"
#include "../misc/Platform-Independent.hpp"
#include "../misc/Utils.hpp"
using tl::expected;
using tl::unexpected;
namespace openpni::example {
using Detector = basic::DetectorGeometry;
struct PolygonalSystem // 多边形系统，也即同构环系统
{
  unsigned edges;           // 多边形的边数
  unsigned detectorPerEdge; // 每条边有几个探测器
  unsigned detectorLen;     // 探测器中心距离
  double radius;            // 中心到边的距离
  double angleOf1stPerp;    // 第一条垂线对应的角度
  unsigned detectorRings;   // 探测器组成的环数
  float ringDistance;       // 相邻两环之间的中心距离

  unsigned totalDetectorNum() const { return edges * detectorPerEdge * detectorRings; }
};

inline PolygonalSystem E180() {
  PolygonalSystem polygon;
  polygon.edges = 24;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 106.5;
  polygon.angleOf1stPerp = 0;
  polygon.detectorRings = 2;
  polygon.ringDistance = 26.5 * 4 + 2;
  return polygon;
}

/**
 * * @brief 计算多边形的顶点坐标，坐标系如此定义:
 * @note 使用右手坐标系，Z轴垂直于多边形平面
 * @note 以Z轴为法线构建平面直角坐标系，Z轴指向平面外侧，则旋转正方向为逆时针
 * @note 沿负Z方向看去，X轴指向水平右侧，Y轴指向正上方
 * @note 环数从负Z向正Z方向延伸
 * @note 探测器法线方向总是垂直与Z轴，与多边形平面平行
 *         ^Y
 *         |
 *  X      |
 * <-------✖Z
 */

/**
 * 为了方便计算，我们将给出如下定义
 * 1.系统坐标系与GATE坐标系重合，这样重建出的图像可以在Amide中直接显示
 * 2.DetectorGeometry的U方向与系统坐标系Z轴重合，V方向为右手螺旋方向(逆时针方向)
 * 3.我们将根据PETLAB
 * 晶体编号和mich格式给出example中的cryid，bin，view，slice，LORid的定义
 * **cryid: 晶体编号，以detector的晶体条为方式进行记录，这里我们定义V方向为晶体条方向
 * **ringID:环ID，我们定义靠近患者头部的那一环为起始环
 * **bin，view,slice依次见文件定义
 * 根据上述定义我们将给出如下计算
 */

// 1.计算detector坐标系在系统坐标系中的坐标
inline basic::Coordinate3D<float> coordinateFromPolygon(
    const PolygonalSystem &__polygon, unsigned __detectorRingIndex, unsigned __detectorIndex) {
  const float edgeLenth = __polygon.detectorPerEdge * __polygon.detectorLen;
  const float rotateAngle = 2.0f * std::numbers::pi_v<float> / __polygon.edges; // 每条边的旋转角度
  // 计算detectorIndex所在的边和位置
  const unsigned edgeIndex = __detectorIndex / __polygon.detectorPerEdge;
  // 计算该edge的中心
  const float angleNow = __polygon.angleOf1stPerp + edgeIndex * rotateAngle;

  const auto midP = basic::make_vec3<float>( // 从X轴正方向开始逆时针时针旋转
                        std::cos(angleNow), std::sin(angleNow), 0) *
                    float(__polygon.radius);
  // 计算detector在该边上的位置0
  const unsigned positionIndex = __detectorIndex % __polygon.detectorPerEdge;

  auto v = basic::make_vec3<float>(std::cos(angleNow + misc::pi_2), std::sin(angleNow + misc::pi_2), 0);
  // const auto p0 = midP - v * (edgeLenth / 2.0f);
  const auto p0 = midP + v * __polygon.detectorLen * (__polygon.detectorPerEdge / -2.f + .5f);
  // 根据端点求出第一个探测器的坐标
  // const auto dete0 = p0 + v * (__polygon.detectorLen / 2.0f);
  // const auto deteNow = dete0 + v * (__polygon.detectorLen * positionIndex);
  const auto p1 = p0 + v * __polygon.detectorLen * positionIndex;
  const auto p2 =
      p1 + basic::make_vec3<float>(0, 0, 1) * (__polygon.detectorRings / -2.f + .5f) * __polygon.ringDistance;
  const auto p3 = p2 + basic::make_vec3<float>(0, 0, 1) * __polygon.ringDistance * __detectorRingIndex;

  basic::Coordinate3D<float> coord;
  coord.position = p3;
  coord.direction = midP.normalized();                        // 方向与v垂直(法向量)
  coord.rotation = basic::make_vec3<float>(0.0f, 0.0f, 1.0f); // U方向为z轴正方向
  return coord;
}
// 计算晶体坐标
inline std::vector<openpni::basic::Vec3<float>> calCrystalPosition(
    const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
  std::vector<openpni::basic::Vec3<float>> cryPos;
  const auto totalDetectorNum = polygon.totalDetectorNum();
  const auto crystalNumInDetector = detectorGeo.getTotalCrystalNum();
  std::vector<basic::Coordinate3D<float>> detectorCoordinatesWithDirection;
  for (const auto detectorIndex : std::views::iota(0u, totalDetectorNum)) {
    detectorCoordinatesWithDirection.push_back(
        openpni::example::coordinateFromPolygon(polygon, detectorIndex / (polygon.detectorPerEdge * polygon.edges),
                                                detectorIndex % (polygon.detectorPerEdge * polygon.edges)));
    for (const auto crystalIndex : std::views::iota(0u, crystalNumInDetector)) {
      const auto coord =
          openpni::basic::calculateCrystalGeometry(detectorCoordinatesWithDirection.back(), detectorGeo, crystalIndex);
      cryPos.push_back(coord.position);
    }
  }
  return cryPos;
}
// temp for test
inline std::vector<basic::CrystalGeometry> calCrystalGeo(
    const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
  std::vector<basic::CrystalGeometry> cryGeo;
  const auto totalDetectorNum = polygon.totalDetectorNum();
  const auto crystalNumInDetector = detectorGeo.getTotalCrystalNum();
  std::vector<basic::Coordinate3D<float>> detectorCoordinatesWithDirection;
  for (const auto detectorIndex : std::views::iota(0u, totalDetectorNum)) {
    detectorCoordinatesWithDirection.push_back(
        openpni::example::coordinateFromPolygon(polygon, detectorIndex / (polygon.detectorPerEdge * polygon.edges),
                                                detectorIndex % (polygon.detectorPerEdge * polygon.edges)));
    for (const auto crystalIndex : std::views::iota(0u, crystalNumInDetector)) {
      const auto coord =
          openpni::basic::calculateCrystalGeometry(detectorCoordinatesWithDirection.back(), detectorGeo, crystalIndex);
      cryGeo.push_back(coord);
    }
  }
  return cryGeo;
}
//
// 计算blockCenter坐标
inline std::vector<openpni::basic::Vec3<float>> calBlockCenterPosition(
    const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
  std::vector<openpni::basic::Vec3<float>> blockPos;
  const auto totalDetectorNum = polygon.totalDetectorNum();
  const auto blockNumInDetector = detectorGeo.getTotalBlockNum();
  std::vector<basic::Coordinate3D<float>> detectorCoordinatesWithDirection;
  for (const auto detectorIndex : std::views::iota(0u, totalDetectorNum)) {
    detectorCoordinatesWithDirection.push_back(
        openpni::example::coordinateFromPolygon(polygon, detectorIndex / (polygon.detectorPerEdge * polygon.edges),
                                                detectorIndex % (polygon.detectorPerEdge * polygon.edges)));
    for (const auto blockIndex : std::views::iota(0u, blockNumInDetector)) {
      const auto coord =
          openpni::basic::calculateCrystalGeometry(detectorCoordinatesWithDirection.back(), detectorGeo, blockIndex);
      blockPos.push_back(coord.position);
    }
  }
  return blockPos;
}

} // namespace openpni::example

namespace openpni::example::polygon {
// 2.ringScanner的mich信息
__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumOneRing(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __polygon.detectorPerEdge * __polygon.edges * __detector.blockNumV * __detector.crystalNumV;
}

__PNI_CUDA_MACRO__ inline uint32_t getRingNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __polygon.detectorRings * __detector.blockNumU * __detector.crystalNumU;
}

__PNI_CUDA_MACRO__ inline uint32_t getBinNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) - 1;
}

__PNI_CUDA_MACRO__ inline uint32_t getViewNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) / 2;
}

__PNI_CUDA_MACRO__ inline uint32_t getSliceNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getRingNum(__polygon, __detector) * getRingNum(__polygon, __detector);
}

__PNI_CUDA_MACRO__ inline size_t getLORNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return size_t(getBinNum(__polygon, __detector)) * size_t(getViewNum(__polygon, __detector)) *
         size_t(getSliceNum(__polygon, __detector));
}
// 3.基于ringScanner的结构信息,即cry,block,module, panel等信息
__PNI_CUDA_MACRO__ inline uint32_t getPanelNum(
    const PolygonalSystem &__polygon) {
  return __polygon.edges;
}
__PNI_CUDA_MACRO__ inline uint32_t getBlockNumOneRing(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __polygon.detectorPerEdge * __polygon.edges * __detector.blockNumV;
}

__PNI_CUDA_MACRO__ inline uint32_t getTotalCrystalNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getCrystalNumOneRing(__polygon, __detector) * getRingNum(__polygon, __detector);
}

__PNI_CUDA_MACRO__ inline uint32_t getTotalBlockNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getBlockNumOneRing(__polygon, __detector) * __detector.blockNumU * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumZInModule(
    const Detector &__detector) // module在ringPETScanner中就是一个探测器
{
  return uint32_t(__detector.blockNumU) * uint32_t(__detector.crystalNumU);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumYInModule(
    const Detector &__detector) {
  return uint32_t(__detector.blockNumV) * uint32_t(__detector.crystalNumV);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumInModule(
    const Detector &__detector) {
  return getCrystalNumZInModule(__detector) * getCrystalNumYInModule(__detector);
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumZInPanel(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getCrystalNumZInModule(__detector) * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getCrystalNumYInPanel(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return getCrystalNumYInModule(__detector) * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getBlockRingNum(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __polygon.detectorRings * __detector.blockNumU;
}
__PNI_CUDA_MACRO__ inline uint32_t getBlockNumZInPanel(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __detector.blockNumU * __polygon.detectorRings;
}

__PNI_CUDA_MACRO__ inline uint32_t getBlockNumYInPanel(
    const PolygonalSystem &__polygon, const Detector &__detector) {
  return __detector.blockNumV * __polygon.detectorRings;
}
__PNI_CUDA_MACRO__ inline basic::Vec3<int> getBinViewSliceFromLORID(
    const PolygonalSystem &__polygon, const Detector &__detector, size_t LORID) {
  int bin = LORID % getBinNum(__polygon, __detector);
  int view = LORID / getBinNum(__polygon, __detector) % getViewNum(__polygon, __detector);
  int slice = LORID / (getBinNum(__polygon, __detector) * getViewNum(__polygon, __detector));
  return {bin, view, slice};
}
// 4.ID计算
__PNI_CUDA_MACRO__ inline basic::Vec2<int> calCrystalIDInRingFromViewBin(
    const PolygonalSystem &__polygon, const Detector &__detector, int view, int bin) {
  int crystalOneRing = getCrystalNumOneRing(__polygon, __detector);
  int cry2 = bin / 2 + 1;
  int cry1 = crystalOneRing + (1 - bin % 2) - cry2;

  cry2 = (cry2 + view) % crystalOneRing;
  cry1 = (cry1 + view) % crystalOneRing;
  return basic::make_vec2<int>(cry1, cry2);
}

__PNI_CUDA_MACRO__ inline basic::Vec2<int> calRing1Ring2FromSlice(
    const PolygonalSystem &__polygon, const Detector &__detector, int slice) {
  return basic::make_vec2<int>(slice / getRingNum(__polygon, __detector), slice % getRingNum(__polygon, __detector));
}

__PNI_CUDA_MACRO__ inline basic::Vec2<uint32_t> calRectangleFlatCrystalIDFromLORID(
    const PolygonalSystem &__polygon, const Detector &__detector, size_t LORID) {
  int crystalNumOneRing = getCrystalNumOneRing(__polygon, __detector);
  int binNum = getBinNum(__polygon, __detector);
  int viewNum = getViewNum(__polygon, __detector);
  int bin = LORID % binNum;
  int view = LORID / binNum % viewNum;
  int slice = LORID / (binNum * viewNum);
  auto [cry1, cry2] = calCrystalIDInRingFromViewBin(__polygon, __detector, view, bin);
  auto [ring1, ring2] = calRing1Ring2FromSlice(__polygon, __detector, slice);
  uint32_t crystalID1 = cry1 + ring1 * crystalNumOneRing;
  uint32_t crystalID2 = cry2 + ring2 * crystalNumOneRing;
  if (crystalID1 > crystalID2)
    return {uint32_t(crystalID1), uint32_t(crystalID2)};
  else
    return {uint32_t(crystalID2), uint32_t(crystalID1)};
}

__PNI_CUDA_MACRO__ inline size_t calLORIDFromRingAndCrystalInRing(
    const PolygonalSystem &__polygon, const Detector &__detector, int ring1, int cry1, int ring2, int cry2) {
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

__PNI_CUDA_MACRO__ inline size_t calLORIDFromCrystalRectangleID(
    const PolygonalSystem &__polygon, const Detector &__detector, int crystalID1, int crystalID2) {
  int crystalNumOneRing = getCrystalNumOneRing(__polygon, __detector);
  int cry1 = crystalID1 % crystalNumOneRing;
  int cry2 = crystalID2 % crystalNumOneRing;
  int ring1 = crystalID1 / crystalNumOneRing;
  int ring2 = crystalID2 / crystalNumOneRing;
  return calLORIDFromRingAndCrystalInRing(__polygon, __detector, ring1, cry1, ring2, cry2);
}

__PNI_CUDA_MACRO__ inline uint32_t calBlockInRingFromCrystalId(
    const PolygonalSystem &__polygon, const Detector &__detector, int crystalID) {
  int cry = crystalID % getCrystalNumOneRing(__polygon, __detector);
  return cry / __detector.crystalNumV;
}
__PNI_CUDA_MACRO__ inline bool isPairFar(
    const PolygonalSystem &__polygon, int panel1, int panel2, int minSectorDifference) {
  return abs(panel1 - panel2) >= minSectorDifference &&
         getPanelNum(__polygon) - abs(panel1 - panel2) >= minSectorDifference;
}

__PNI_CUDA_MACRO__ inline uint32_t getRectangleIDFromUniformID(
    const PolygonalSystem &__polygon, const Detector &__detector, int __uniformID) {
  const auto bdmID = __uniformID / getCrystalNumInModule(__detector);
  const auto cryIDInModule = __uniformID % getCrystalNumInModule(__detector);
  const auto uInModule = cryIDInModule % getCrystalNumZInModule(__detector);
  const auto vInModule = getCrystalNumYInModule(__detector) - 1 -
                         cryIDInModule / getCrystalNumZInModule(__detector); // 由于坐标系问题这里得翻一下
  const auto bdmIDInRing = bdmID % (__polygon.edges * __polygon.detectorPerEdge);
  const auto bdmRingId = bdmID / (__polygon.edges * __polygon.detectorPerEdge);
  const auto crystalRingID = uInModule + getCrystalNumZInModule(__detector) * bdmRingId;
  const auto crystalIdInRing = vInModule + getCrystalNumYInModule(__detector) * bdmIDInRing;
  return crystalIdInRing + crystalRingID * getCrystalNumOneRing(__polygon, __detector);
}
__PNI_CUDA_MACRO__ inline size_t calLORIDFromCrystalUniformID(
    const PolygonalSystem &__polygon, const Detector &__detector, int crystalID1, int crystalID2) {
  return calLORIDFromCrystalRectangleID(__polygon, __detector,
                                        getRectangleIDFromUniformID(__polygon, __detector, crystalID1),
                                        getRectangleIDFromUniformID(__polygon, __detector, crystalID2));
}
__PNI_CUDA_MACRO__ inline int calBinNumOutFOVOneSide(
    const PolygonalSystem &__polygon, const Detector &__detector, int minSectorDifference) {
  int crystalNumYInPanel = getCrystalNumYInPanel(__polygon, __detector);
  if (minSectorDifference == 0)
    return 0;
  return crystalNumYInPanel * minSectorDifference - 1;
}

__PNI_CUDA_MACRO__ inline uint32_t getUniformIDFromRectangleID(
    const PolygonalSystem &__polygon, const Detector &__detector, uint32_t __rectangleID) {
  const auto crystalNumOneRing = getCrystalNumOneRing(__polygon, __detector);
  // const auto bdmNumOneRing = __polygon.edges * __polygon.detectorPerEdge;
  const auto cryRingID = __rectangleID / crystalNumOneRing;
  const auto bdmRingID = cryRingID / getCrystalNumZInModule(__detector);
  const auto cryIDInRing = __rectangleID % crystalNumOneRing;
  const auto bdmIDInRing = cryIDInRing / getCrystalNumYInModule(__detector);

  const auto uInModule = cryRingID % getCrystalNumZInModule(__detector);
  const auto vInModule = getCrystalNumYInModule(__detector) - 1 -
                         cryIDInRing % getCrystalNumYInModule(__detector); // 由于坐标系问题这里得翻一下
  const auto bdmID = bdmIDInRing + bdmRingID * (__polygon.edges * __polygon.detectorPerEdge);
  return bdmID * getCrystalNumInModule(__detector) + uInModule + vInModule * getCrystalNumZInModule(__detector);
}

__PNI_CUDA_MACRO__ inline int calViewNumInSubset(
    const PolygonalSystem &__polygon, const Detector &__detector, int subsetNum, int subsetIndex) {
  std::size_t viewNum = getViewNum(__polygon, __detector);
  return (viewNum + subsetNum - 1 - subsetIndex) / subsetNum;
}

__PNI_CUDA_MACRO__ inline std::size_t calLORIDFromSubsetId(
    const PolygonalSystem &__polygon, const Detector &__detector, int subsetId, int subsetNum, int binCut,
    std::size_t id) // only can be used for view_subset
{
  std::size_t binNum = getBinNum(__polygon, __detector);
  std::size_t viewNum = getViewNum(__polygon, __detector);
  std::size_t sliceNum = getSliceNum(__polygon, __detector);
  std::size_t viewNumInSubset = calViewNumInSubset(__polygon, __detector, subsetNum, subsetId);

  std::size_t binIndex = (id % (binNum - 2 * binCut)) + binCut;
  std::size_t sliceIndex = (id / (binNum - 2 * binCut)) % sliceNum;
  std::size_t viewIndexInSubset = id / ((binNum - 2 * binCut) * sliceNum);

  std::size_t result = sliceIndex * binNum * viewNum + (viewIndexInSubset * subsetNum + subsetId) * binNum + binIndex;
  return result;
}

__PNI_CUDA_MACRO__ inline basic::Vec2<uint32_t> calCrystalIDFromSubsetId(
    const PolygonalSystem &__polygon, const Detector &__detector, int subsetId, int subsetNum, int binCut,
    std::size_t id) {
  return calRectangleFlatCrystalIDFromLORID(
      __polygon, __detector, calLORIDFromSubsetId(__polygon, __detector, subsetId, subsetNum, binCut, id));
}

__PNI_CUDA_MACRO__ inline std::size_t calSubsetSizeByView(
    const PolygonalSystem &__polygon, const Detector &__detector, int __subsetNum, int __subsetIndex, int __binCut) {
  std::size_t binNum = getBinNum(__polygon, __detector);
  std::size_t viewNum = getViewNum(__polygon, __detector);
  std::size_t sliceNum = getSliceNum(__polygon, __detector);

  std::size_t result = 0;
  for (std::size_t viewIndex = __subsetIndex; viewIndex < viewNum; viewIndex += __subsetNum)
    result++;

  result *= binNum - 2 * __binCut;
  result *= sliceNum;
  return result;
}

struct IndexerOfSubsetForMich {
  PolygonalSystem scanner;
  Detector detector;
  int subsetNum;
  int subsetId;
  int binCut;

  __PNI_CUDA_MACRO__ basic::Vec2<uint32_t> operator()(
      std::size_t __dataIndex) const {
    const auto rectangleID = calCrystalIDFromSubsetId(scanner, detector, subsetId, subsetNum, binCut, __dataIndex);
    return basic::make_vec2<uint32_t>(getUniformIDFromRectangleID(scanner, detector, rectangleID.x),
                                      getUniformIDFromRectangleID(scanner, detector, rectangleID.y));
  }
  __PNI_CUDA_MACRO__ std::size_t operator[](
      std::size_t __dataIndex) const {
    return calLORIDFromSubsetId(scanner, detector, subsetId, subsetNum, binCut, __dataIndex);
  }
  __PNI_CUDA_MACRO__ std::size_t indexInMich(
      std::size_t __cry1, std::size_t __cry2) const {
    return calLORIDFromCrystalUniformID(scanner, detector, __cry1, __cry2);
  }
  __PNI_CUDA_MACRO__ std::size_t count() const {
    return calSubsetSizeByView(scanner, detector, subsetNum, subsetId, binCut);
  }
  __PNI_CUDA_MACRO__ std::size_t crystalNum() const { return getTotalCrystalNum(scanner, detector); }
};

struct Locator {
  PolygonalSystem m_scanner;
  Detector m_detector;
  Locator(
      const PolygonalSystem &__scanner, const Detector &__detector)
      : m_scanner(__scanner)
      , m_detector(__detector) {}

public:
  template <typename T>
  auto setMich(
      T *__mich) const {
    return std::views::transform([=](auto index) -> T & { return __mich[index]; });
  }

public:
  auto allLORs() const { return std::views::iota(0ull, getLORNum(m_scanner, m_detector)); }
  auto bins() const { return std::views::iota(0u, getBinNum(m_scanner, m_detector)); }
  auto views() const { return std::views::iota(0u, getViewNum(m_scanner, m_detector)); }
  auto slices() const { return std::views::iota(0u, getSliceNum(m_scanner, m_detector)); }
  auto rings() const { return std::views::iota(0u, getRingNum(m_scanner, m_detector)); }
  auto crystalsInRing() const {
    return std::views::iota(0u, getCrystalNumOneRing(m_scanner, m_detector) * getRingNum(m_scanner, m_detector));
  }
  auto lorsForGivenBin(
      int bin) const {
    return std::views::iota(0u, getViewNum(m_scanner, m_detector) * getSliceNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             return size_t(index) * size_t(getBinNum(m_scanner, m_detector)) + size_t(bin);
           });
  }
  auto lorsForGivenView(
      int view) const {
    return std::views::iota(0u, getBinNum(m_scanner, m_detector) * getSliceNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             const auto binIndex = index % getBinNum(m_scanner, m_detector);
             const auto sliceIndex = index / getBinNum(m_scanner, m_detector);
             return size_t(sliceIndex) * size_t(getBinNum(m_scanner, m_detector)) *
                        size_t(getViewNum(m_scanner, m_detector)) +
                    size_t(view) * size_t(getBinNum(m_scanner, m_detector)) + size_t(binIndex);
           });
  }
  auto lorsForGivenSlice(
      int slice) const {
    return std::views::iota(0u, getBinNum(m_scanner, m_detector) * getViewNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             const auto binIndex = index % getBinNum(m_scanner, m_detector);
             const auto viewIndex = index / getBinNum(m_scanner, m_detector);
             return size_t(slice) * size_t(getBinNum(m_scanner, m_detector)) *
                        size_t(getViewNum(m_scanner, m_detector)) +
                    size_t(viewIndex) * size_t(getBinNum(m_scanner, m_detector)) + size_t(binIndex);
           });
  }
  auto lorsForGivenBinView(
      int bin, int view) const {
    return std::views::iota(0u, getSliceNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             return size_t(index) * size_t(getBinNum(m_scanner, m_detector)) *
                        size_t(getViewNum(m_scanner, m_detector)) +
                    size_t(view) * size_t(getBinNum(m_scanner, m_detector)) + size_t(bin);
           });
  }
  auto lorsForGivenBinSlice(
      int bin, int slice) const {
    return std::views::iota(0u, getViewNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             return size_t(slice) * size_t(getBinNum(m_scanner, m_detector)) *
                        size_t(getViewNum(m_scanner, m_detector)) +
                    size_t(index) * size_t(getBinNum(m_scanner, m_detector)) + size_t(bin);
           });
  }
  auto lorsForGivenViewSlice(
      int view, int slice) const {
    return std::views::iota(0u, getBinNum(m_scanner, m_detector)) |
           std::views::transform([=, this](auto index) noexcept {
             return size_t(slice) * size_t(getBinNum(m_scanner, m_detector)) *
                        size_t(getViewNum(m_scanner, m_detector)) +
                    size_t(view) * size_t(getBinNum(m_scanner, m_detector)) + size_t(index);
           });
  }
  auto allBinViewSlices() const {
    return allLORs() | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto allLORAndBinViewSlices() const {
    return allBinViewSlices() | std::views::transform([this](auto bvs) noexcept {
             const auto &[b, v, s] = bvs;
             const auto lorIndex = b + v * getBinNum(m_scanner, m_detector) +
                                   s * getBinNum(m_scanner, m_detector) * getViewNum(m_scanner, m_detector);
             return std::make_tuple(lorIndex, b, v, s);
           });
  }
  auto bvsForGivenBin(
      int bin) const {
    return lorsForGivenBin(bin) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto bvsForGivenView(
      int view) const {
    return lorsForGivenView(view) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto bvsForGivenSlice(
      int slice) const {
    return lorsForGivenSlice(slice) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto bvsForGivenBinView(
      int bin, int view) const {
    return lorsForGivenBinView(bin, view) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto bvsForGivenBinSlice(
      int bin, int slice) const {
    return lorsForGivenBinSlice(bin, slice) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto bvsForGivenViewSlice(
      int view, int slice) const {
    return lorsForGivenViewSlice(view, slice) | std::views::transform([this](auto lorIndex) noexcept {
             return getBinViewSliceFromLORID(m_scanner, m_detector, lorIndex);
           });
  }
  auto allCrystals() const { return std::views::iota(0u, getTotalCrystalNum(m_scanner, m_detector)); }
  auto allLocalCrystalAndDetector() const {
    return allCrystals() | std::views::transform([this](auto cryID) noexcept {
             const auto crystalNumPerDetector = m_detector.getTotalCrystalNum();
             return basic::make_vec2<uint32_t>(cryID % crystalNumPerDetector, cryID / crystalNumPerDetector);
           });
  }
  auto allCrystalAndLocalUVAndDetector() const {
    return allCrystals() | std::views::transform([this](auto cryID) noexcept {
             const auto crystalNumPerDetector = m_detector.getTotalCrystalNum();
             const auto crystalNumUInDetector = m_detector.crystalNumU * m_detector.blockNumU;
             const auto crystalLocalIndex = cryID % crystalNumPerDetector;
             return std::make_tuple(cryID, crystalLocalIndex % crystalNumUInDetector,
                                    crystalLocalIndex / crystalNumUInDetector, cryID / crystalNumPerDetector);
           });
  }
  auto allCrystalUV() const {
    return allCrystals() | std::views::transform([this](auto cryID) noexcept {
             return basic::make_vec2<uint32_t>(cryID / getCrystalNumOneRing(m_scanner, m_detector),
                                               cryID % getCrystalNumOneRing(m_scanner, m_detector));
           });
  }
  auto allCrystalRectanglesAndUV() const {
    return allCrystals() | std::views::transform([this](auto cryID) noexcept {
             return std::make_tuple<std::size_t, uint32_t, uint32_t>(
                 cryID, cryID / getCrystalNumOneRing(m_scanner, m_detector),
                 cryID % getCrystalNumOneRing(m_scanner, m_detector));
           });
  }
  auto allCrystalUniformAndRectangleAndUV() const {
    return allCrystals() | std::views::transform([this](auto cryID) noexcept {
             return std::make_tuple<std::size_t, uint32_t, uint32_t, uint32_t>(
                 cryID, getUniformIDFromRectangleID(m_scanner, m_detector, cryID),
                 cryID / getCrystalNumOneRing(m_scanner, m_detector),
                 cryID % getCrystalNumOneRing(m_scanner, m_detector));
           });
  }
  auto allLORCrystals() const {
    return allLORs() | std::views::transform([this](auto lorID) noexcept {
             return calRectangleFlatCrystalIDFromLORID(m_scanner, m_detector, lorID);
           });
  }
  auto allLORAndCrystals() const {
    return allLORs() | std::views::transform([this](auto lorID) noexcept {
             const auto crystalPair = calRectangleFlatCrystalIDFromLORID(m_scanner, m_detector, lorID);
             return basic::make_vec3<std::size_t>(lorID, crystalPair.x, crystalPair.y);
           });
  }
  auto allLORAndBinViewSlicesAndCrystals() const {
    return allLORs() | std::views::transform([this](auto lorID) noexcept {
             const auto crystalPair = calRectangleFlatCrystalIDFromLORID(m_scanner, m_detector, lorID);
             const auto [b, v, s] = getBinViewSliceFromLORID(m_scanner, m_detector, lorID);
             return std::make_tuple(lorID, b, v, s, crystalPair.x, crystalPair.y);
           });
  }
};

} // namespace openpni::example::polygon
namespace openpni::example::polygon {

template <typename DetectorType>
  requires std::is_base_of_v<device::DetectorBase, DetectorType>
class PolygonModelBuilder;
class PolygonModel {
public:
  PolygonModel(
      PolygonalSystem system, std::vector<std::shared_ptr<device::DetectorBase>> &&detectors,
      device::DetectorUnchangable detectorUnchangable)
      : m_systemDefinition(system)
      , m_detectorUnchangable(detectorUnchangable)
      , m_locator(m_systemDefinition, m_detectorUnchangable.geometry)
      , m_detectors(std::move(detectors)) {
    m_crystalNumPrefixSum = misc::rangeToContainer<decltype(m_crystalNumPrefixSum)>(
        std::views::iota(0u, m_systemDefinition.totalDetectorNum() + 1) |
        std::views::transform(
            [this](auto index) noexcept { return m_detectorUnchangable.geometry.getTotalCrystalNum() * index; }));
    m_crystalGeometry = misc::rangeToContainer<decltype(m_crystalGeometry)>(
        m_locator.allCrystalAndLocalUVAndDetector() | std::views::transform([this](auto item) noexcept {
          const auto [cryID, localU, localV, detID] = item;
          return basic::calculateCrystalGeometry(m_detectors.at(detID)->detectorChangable().coordinate,
                                                 m_detectorUnchangable.geometry, localU, localV);
        }));
  }
  ~PolygonModel() = default;

public:
  std::vector<device::DetectorBase *> detectorRuntimes() const noexcept {
    auto range = m_detectors | std::views::transform([](const auto &detector) noexcept { return detector.get(); });
    return std::vector<device::DetectorBase *>(range.begin(), range.end());
  }
  const std::vector<unsigned> &crystalNumPrefixSum() const noexcept { return m_crystalNumPrefixSum; }
  unsigned crystalNum() const noexcept { return m_crystalNumPrefixSum.back(); }
  unsigned detectorNum() const noexcept { return m_detectors.size(); }
  const std::vector<basic::CrystalGeometry> &crystalGeometry() const noexcept { return m_crystalGeometry; }
  process::AcquisitionInfo acquisitionParams(
      uint64_t __maxBufferSize, uint16_t __timeSwitchBuffer_ms) const noexcept {
    process::AcquisitionInfo result;
    for (const auto index : std::views::iota(0ull, m_detectors.size())) {
      const auto &detector = m_detectors.at(index);
      result.channelSettings.push_back({});
      result.channelSettings.back().ipSource = detector->detectorChangable().ipSource;
      result.channelSettings.back().portSource = detector->detectorChangable().portSource;
      result.channelSettings.back().ipDestination = detector->detectorChangable().ipDestination;
      result.channelSettings.back().portDestination = detector->detectorChangable().portDestination;
      result.channelSettings.back().channelIndex = index;
      result.channelSettings.back().quickFilter =
          [detector = m_detectorUnchangable](uint8_t *__udpDatagram, uint16_t __udpLegnth, uint32_t __ipSource,
                                             uint16_t __portSource) noexcept -> bool {
        return __udpLegnth >= detector.minUDPPacketSize && __udpLegnth <= detector.maxUDPPacketSize;
      };
    }
    result.maxBufferSize = __maxBufferSize;
    result.storageUnitSize = m_detectorUnchangable.maxUDPPacketSize;
    result.timeSwitchBuffer_ms = __timeSwitchBuffer_ms;
    result.totalChannelNum = m_detectors.size();
    return result;
  }
  auto &locator() const noexcept { return m_locator; }
  auto michSubsetIndexers(
      int subsetNum, int binCut) const {
    return std::views::iota(0, subsetNum) | std::views::transform([this, binCut, subsetNum](auto index) noexcept {
             IndexerOfSubsetForMich result;
             result.detector = m_detectorUnchangable.geometry;
             result.scanner = m_systemDefinition;
             result.binCut = binCut;
             result.subsetNum = subsetNum;
             result.subsetId = index;
             return result;
           });
  }
  template <typename ImageType>
  expected<ImageType, std::string> readMichFromFile(
      const std::string &filePath) const {
    return ImageType::loadFromFile(filePath).and_then([this](auto &img) {
      const auto binNum = m_locator.bins().size();
      const auto viewNum = m_locator.views().size();
      const auto sliceNum = m_locator.slices().size();
      if (img.getXNum() != binNum || img.getYNum() != viewNum || img.getZNum() != sliceNum)
        return unexpected("Image size mismatch with the current scanner and detector setting.");
      return img;
    });
  }
  const auto &polygonSystem() const noexcept { return m_systemDefinition; }
  const auto &detectorInfo() const noexcept { return m_detectorUnchangable; }
  std::size_t michSize() const noexcept { return m_locator.allLORs().size(); }

private:
  PolygonalSystem m_systemDefinition;
  device::DetectorUnchangable m_detectorUnchangable;
  Locator m_locator;
  std::vector<std::shared_ptr<device::DetectorBase>> m_detectors;
  std::vector<unsigned> m_crystalNumPrefixSum;
  std::vector<basic::CrystalGeometry> m_crystalGeometry;
};

template <typename DetectorType>
  requires std::is_base_of_v<device::DetectorBase, DetectorType>
class PolygonModelBuilder {
public:
  PolygonModelBuilder(
      PolygonalSystem system,
      device::DetectorUnchangable detectorUnchangable = device::detectorUnchangable<DetectorType>()) {
    m_systemDefinition = system;
    m_detectorUnchangable = detectorUnchangable;
    m_detectors = std::vector<std::shared_ptr<device::DetectorBase>>();
    const auto detectorNumOneRing = m_systemDefinition.detectorPerEdge * m_systemDefinition.edges;
    for (const auto index : std::views::iota(0u, m_systemDefinition.totalDetectorNum())) {
      m_detectors.push_back(std::shared_ptr<device::DetectorBase>(new DetectorType()));
      m_detectors.back()->detectorChangable().coordinate =
          coordinateFromPolygon(m_systemDefinition, index / detectorNumOneRing, index % detectorNumOneRing);
    }
  }
  ~PolygonModelBuilder() = default;

public:
  int totalDetectorNum() const { return m_systemDefinition.totalDetectorNum(); }
  void setDetectorIP(
      int detectorIndex, std::string srcIP, uint16_t srcPort, std::string dstIP, uint16_t dstPort) {
    auto detector = m_detectors.at(detectorIndex);
    detector->detectorChangable().ipSource = misc::ipStr2ipInt(srcIP);
    detector->detectorChangable().portSource = srcPort;
    detector->detectorChangable().ipDestination = misc::ipStr2ipInt(dstIP);
    detector->detectorChangable().portDestination = dstPort;
  }
  void loadCalibration(
      int detectorIndex, std::string calibrationFilePath) {
    auto detector = m_detectors.at(detectorIndex);
    detector->loadCalibration(calibrationFilePath);
  }
  std::unique_ptr<PolygonModel> build() {
    return std::make_unique<PolygonModel>(m_systemDefinition, std::move(m_detectors), m_detectorUnchangable);
  }

private:
  PolygonalSystem m_systemDefinition;
  device::DetectorUnchangable m_detectorUnchangable;
  std::vector<std::shared_ptr<device::DetectorBase>> m_detectors;
};

} // namespace openpni::example::polygon
namespace openpni::example::polygon {
enum class QuickBuildFromJsonError {
  NoError = 0,
  JsonParseError = 1,
  EmptyJson = 2,
  InfoNumMismatch = 3,
  UnKnownDetector = 4,
  CaliFileInvalid = 5,
};
const char *getErrorMessage(QuickBuildFromJsonError error) noexcept;
expected<std::shared_ptr<PolygonModel>, QuickBuildFromJsonError>
quickBuildFromJson(const std::string &unparsedJson) noexcept;
} // namespace openpni::example::polygon

namespace openpni::example::polygon::lazy {}